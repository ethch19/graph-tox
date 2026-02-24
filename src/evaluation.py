import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch import optim
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.data_prep import DrugDB
from src.model import FusionModel, GINet
from src.utils import cal_roc_auc, scaffold_split, smiles_to_graph

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models/saved"
EVAL_DIR = BASE_DIR / "eval"

MOLNET_TARGETS = {
    "BBBP": ["p_np"],
    "Tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
    "ClinTox": ["CT_TOX", "FDA_APPROVED"],
    "HIV": ["HIV_active"],
    "BACE": ["Class"],
    "SIDER": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "MUV": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
}


def calculate_hits_at_k(similarity_matrix: torch.Tensor, k: int = 5) -> float:
    """calculates retrieval accuracy (Top-K)"""
    N = similarity_matrix.shape[0]
    _, top_k_indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
    targets = torch.arange(N, device=similarity_matrix.device).view(-1, 1)
    correct_in_top_k = (top_k_indices == targets).any(dim=1)
    return correct_in_top_k.float().mean().item() * 100


def eval_zero_shot_retrieval(
    model: nn.Module,
    device: torch.device,
    target_path: Path,
    batch_size: int = 500,
    heatmap_size: int = 30,
):
    print("\n---Starting Zero-Shot Retrieval---")
    dataset = DrugDB(db_path=str(DATA_DIR / "drugs.db"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    batch = next(iter(loader)).to(device)
    model.eval()

    with torch.no_grad():
        graph_embeds, bio_embeds, _ = model(batch, batch.lincs, batch.chembl)
        sim_matrix = (
            graph_embeds @ bio_embeds.t()
        )  # dot product of normalized vectors = cosine similarity

    # cross-modal retrieval accuracy at top 1/10
    # graph-to-assay, assay-to-graph
    results = {
        "G2A_Hits@1": calculate_hits_at_k(sim_matrix, k=1),
        "G2A_Hits@10": calculate_hits_at_k(sim_matrix, k=10),
        "A2G_Hits@1": calculate_hits_at_k(sim_matrix.t(), k=1),
        "A2G_Hits@10": calculate_hits_at_k(sim_matrix.t(), k=10),
    }

    for k, v in results.items():
        print(f"{k}: {v:.2f}%")

    # heatmap
    save_path = target_path / "fusionsim_heatmap.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix[:heatmap_size, :heatmap_size].cpu().numpy(), cmap="viridis")
    plt.title("Fusion Similarity Heatmap")
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")


def get_predictions(
    gnn_encoder: nn.Module,
    linear_head: nn.Linear,
    loader: DataLoader,
    device: torch.device,
):
    linear_head.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features, _ = gnn_encoder(batch)
            logits = linear_head(features)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    return np.vstack(all_preds), np.vstack(all_targets)


# classification
def eval_downstream(
    gnn_encoder: nn.Module,
    device: torch.device,
    dataset_name: str,  # "BBBP", "Tox21", "ClinTox", "HIV", "BACE", "SIDER", "MUV"
    target_path: Path,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
):
    print(f"\n---Starting {dataset_name} evaluation---")

    if dataset_name not in MOLNET_TARGETS:
        raise ValueError(
            f"Targets for classification dataset '{dataset_name}' are not defined."
        )

    MoleculeNet(root=str(DATA_DIR), name=dataset_name)  # using it as downloader
    raw_dir = DATA_DIR / dataset_name.lower() / "raw"
    csv_paths = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.csv.gz"))
    if not csv_paths:
        raise FileNotFoundError(f"Could not find any .csv or .csv.gz file in {raw_dir}")
    csv_path = csv_paths[0]
    df = pd.read_csv(csv_path)

    label_cols = MOLNET_TARGETS[dataset_name]
    num_tasks = len(label_cols)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]

    print(
        f"[{dataset_name} dataset loaded]: {len(df)} total molecules. {num_tasks} distinct tasks."
    )

    data_list = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        labels = row[label_cols].values.astype(np.float32)

        x, edge_index, edge_attr = smiles_to_graph(smi)
        if x is not None:
            y = torch.tensor(labels, dtype=torch.float).view(1, -1)
            data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smi)
            )

    train_set, val_set, test_set = scaffold_split(data_list)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for param in gnn_encoder.parameters():
        param.requires_grad = False

    dummy_batch = next(iter(train_loader)).to(device)

    with torch.no_grad():
        dummy_feat, _ = gnn_encoder(dummy_batch)
        head_input_dim = dummy_feat.shape[1]

    # MLP head (1 hidden, 256, ReLU, Dropout), same as MolCLR
    linear_head = nn.Sequential(
        nn.Linear(head_input_dim, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_tasks),
    ).to(device)

    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_weights = copy.deepcopy(linear_head.state_dict())
    history = []
    val_history = []

    for epoch in range(epochs):
        linear_head.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features, _ = gnn_encoder(batch)

            logits = linear_head(features)

            is_labeled = ~torch.isnan(batch.y)
            loss = criterion(logits[is_labeled], batch.y[is_labeled])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)

        linear_head.eval()
        val_loss_total = 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                features, _ = gnn_encoder(batch)
                logits = linear_head(features)

                is_labeled = ~torch.isnan(batch.y)
                val_loss = criterion(logits[is_labeled], batch.y[is_labeled])
                val_loss_total += val_loss.item()

                probs = torch.sigmoid(logits)
                all_val_preds.append(probs.cpu().numpy())
                all_val_targets.append(batch.y.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        val_history.append(avg_val_loss)

        val_preds = np.vstack(all_val_preds)
        val_targets = np.vstack(all_val_targets)
        val_auc = cal_roc_auc(val_preds, val_targets, num_tasks)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(linear_head.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"   Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_auc * 100:.2f}%"
            )

    linear_head.load_state_dict(best_weights)
    test_preds, test_targets = get_predictions(
        gnn_encoder, linear_head, test_loader, device
    )
    test_auc = cal_roc_auc(test_preds, test_targets, num_tasks)

    print(f"[{dataset_name}] Validation ROC-AUC: {best_val_auc * 100:.2f}%")
    print(f"[{dataset_name}] Test ROC-AUC: {test_auc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(history, label="Training Loss", color="blue")
    plt.plot(val_history, label="Validation Loss", color="orange")
    plt.title(f"{dataset_name} MLP Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path / f"{dataset_name.lower()}_training_loss.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, col_name in enumerate(label_cols):
        valid_idx = ~np.isnan(test_targets[:, i])
        valid_targets = test_targets[valid_idx, i]
        valid_preds = test_preds[valid_idx, i]

        if len(np.unique(valid_targets)) == 2:
            fpr, tpr, _ = roc_curve(valid_targets, valid_preds)
            task_auc = roc_auc_score(valid_targets, valid_preds)
            plt.plot(
                fpr,
                tpr,
                label=f"{col_name} (AUC = {task_auc:.2f})",
                alpha=0.8,
                linewidth=1.5,
            )

    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{dataset_name} Test ROC Curves\nMacro Average ROC-AUC: {test_auc * 100:.2f}%"
    )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.tight_layout()
    plt.savefig(target_path / f"{dataset_name.lower()}_roc_curves.png")
    plt.close()

    print(f"Training loss + ROC-AUC plot saved to {target_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = EVAL_DIR / f"{timestamp}"
    target_path.mkdir(parents=True, exist_ok=True)

    model_folders = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
    if not model_folders:
        print(
            f"Error: Could not find any model weights matching 'fusion_model_*.pth' in {MODEL_DIR}"
        )
        return
    latest_model_folder = max(model_folders, key=lambda f: f.stat().st_mtime)

    model_files = list(latest_model_folder.glob("fusion_model*.pth"))
    if not model_files:
        print(f"Error: Could not find 'fusion_model*.pth' in {latest_model_folder}")
        return
    latest_model_path = model_files[0]
    print(f"Loading weights from: {latest_model_path.name}")

    raw_gin = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0)
    model = FusionModel(
        gnn_model=raw_gin,
        lincs_input_dim=978,
        chembl_input_dim=1283,
        gnn_feat_dim=512,
        embed_dim=256,
    )

    state_dict = torch.load(latest_model_path, map_location=device)

    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            unwrapped_state_dict[key[7:]] = value
        else:
            unwrapped_state_dict[key] = value
    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    print("Model weights loaded successfully.")

    eval_zero_shot_retrieval(model, device, target_path)

    downstream_datasets = ["BBBP", "Tox21", "ClinTox", "HIV", "BACE", "SIDER", "MUV"]
    for dset in downstream_datasets:
        eval_downstream(model.gnn, device, dset, target_path)


if __name__ == "__main__":
    main()
