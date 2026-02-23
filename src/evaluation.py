from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch import optim
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.data_prep import DrugDB
from src.model import FusionModel, GINet

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "models/saved"


def calculate_hits_at_k(similarity_matrix: torch.Tensor, k: int = 5) -> float:
    """calculates retrieval accuracy (Top-K)"""
    N = similarity_matrix.shape[0]
    _, top_k_indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
    targets = torch.arange(N, device=similarity_matrix.device).view(-1, 1)
    correct_in_top_k = (top_k_indices == targets).any(dim=1)
    return correct_in_top_k.float().mean().item() * 100


def run_zero_shot_retrieval(
    model: nn.Module,
    device: torch.device,
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix[:heatmap_size, :heatmap_size].cpu().numpy(), cmap="viridis")
    plt.title("Fusion Similarity Heatmap")
    plt.savefig(SAVE_DIR / "fusionsim_heatmap.png")
    plt.close()
    print(f"Heatmap saved to {SAVE_DIR / 'fusionsim_heatmap.png'}")


def run_tox21(
    gnn_encoder: nn.Module,
    device: torch.device,
    batch_size: int = 128,
    epochs: int = 15,
    lr: float = 1e-3,
    train_split: float = 0.8,
    head_input_dim: int = 256,
    num_tasks: int = 12,  # tox21 binary classification tasks
):
    """tox21 linear downstream"""
    print("\n---Starting Tox21 evaluation---")

    tox_dataset = MoleculeNet(root=str(DATA_DIR), name="Tox21")
    train_size = int(train_split * len(tox_dataset))
    val_size = len(tox_dataset) - train_size
    train_set, val_set = random_split(tox_dataset, [train_size, val_size])

    loader_args = {"batch_size": batch_size, "drop_last": True}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    for param in gnn_encoder.parameters():  # freeze GIN
        param.requires_grad = False

    linear_head = nn.Linear(head_input_dim, num_tasks).to(device)
    optimizer = optim.Adam(linear_head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(epochs):
        linear_head.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features, _ = gnn_encoder(batch)

            logits = linear_head(features)

            is_labeled = batch.y == batch.y
            loss = criterion(logits[is_labeled], batch.y[is_labeled].float())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_loss:.4f}")

    linear_head.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            features, _ = gnn_encoder(batch)
            logits = linear_head(features)
            probs = torch.sigmoid(logits)  # convert logits to probabilities

            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    roc_aucs = []
    fpr_dict, tpr_dict = {}, {}

    for i in range(num_tasks):
        valid_idx = ~np.isnan(all_targets[:, i])
        valid_targets = all_targets[valid_idx, i]
        valid_preds = all_preds[valid_idx, i]

        if len(np.unique(valid_targets)) == 2:
            score = roc_auc_score(valid_targets, valid_preds)
            roc_aucs.append(score)
            fpr, tpr, _ = roc_curve(valid_targets, valid_preds)
            fpr_dict[i], tpr_dict[i] = fpr, tpr

    macro_roc_auc = np.mean(roc_aucs) * 100  # averaging across all tasks
    print(f"\nTox21 ROC-AUC: {macro_roc_auc:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(history, label="Training Loss", color="blue")
    plt.title("Tox21 Linear Probing Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "tox21_training_loss.png")
    plt.close()
    print(f"Training loss plot saved to {SAVE_DIR / 'tox21_training_loss.png'}")

    plt.figure(figsize=(8, 6))
    for i in fpr_dict.keys():
        plt.plot(fpr_dict[i], tpr_dict[i], alpha=0.3, lw=1)

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Tox21 Validation ROC Curves\nMacro Average ROC-AUC: {macro_roc_auc:.2f}%"
    )
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "tox21_roc_curve.png")
    plt.close()
    print(f"ROC curve plot saved to {SAVE_DIR / 'tox21_roc_curve.png'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model_files = list(SAVE_DIR.glob("fusion_model_*.pth"))
    if not model_files:
        print(
            f"Error: Could not find any model weights matching 'fusion_model_*.pth' in {SAVE_DIR}"
        )
        return
    latest_model_path = max(model_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading latest model weights from: {latest_model_path.name}")

    raw_gin = GINet(num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0.0)
    model = FusionModel(
        gnn_model=raw_gin,
        lincs_input_dim=978,
        chembl_input_dim=1283,
        gnn_feat_dim=256,
        embed_dim=256,
    )

    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.to(device)
    print("Model weights loaded successfully.")

    run_zero_shot_retrieval(model, device)
    run_tox21(model.gnn, device)


if __name__ == "__main__":
    main()
