import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data_prep import DrugDB
from src.model import FusionModel, GINet

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "models/saved"
CHECKPOINT_LATEST = SAVE_DIR / "ckpt_latest.pth"
PRETRAINED_GIN_PATH = BASE_DIR / "models/pretrained/gin_model.pth"


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    val_losses: List[float],
    timestamp: str,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVE_DIR / f"ckpt_{timestamp}_epoch{epoch + 1}.pth"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "timestamp": timestamp,
    }

    torch.save(checkpoint, str(ckpt_path))
    torch.save(checkpoint, str(CHECKPOINT_LATEST))
    print(f"Checkpoint saved: {ckpt_path.name}")


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[int, List[float], List[float], str]:
    if CHECKPOINT_LATEST.exists():
        ckpt = torch.load(str(CHECKPOINT_LATEST))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resuming from {ckpt['timestamp']} at epoch {ckpt['epoch'] + 1}")
        return (
            ckpt["epoch"] + 1,
            ckpt["train_losses"],
            ckpt["val_losses"],
            ckpt["timestamp"],
        )
    return 0, [], [], datetime.now().strftime("%Y%m%d_%H%M%S")


def load_pretrained_gnn(pth_path: Path, gnn_instance: nn.Module) -> nn.Module:
    state_dict = torch.load(pth_path, map_location="cpu")

    gnn_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("projection_head")
    }

    missing_keys, unexpected_keys = gnn_instance.load_state_dict(
        gnn_state_dict, strict=False
    )

    print(f"Successfully loaded weights. Ignored keys: {unexpected_keys}")
    return gnn_instance


def clip_loss(graph_embeds, assay_embeds, logit_scale):  # InfoNCE loss
    # clamp temperature to prevent numerical instability
    logit_scale = torch.clamp(logit_scale, max=100.0)

    # similarity matrix (batch size x batch size)
    logits_per_graph = logit_scale * graph_embeds @ assay_embeds.t()
    logits_per_assay = logits_per_graph.t()

    batch_size = graph_embeds.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long, device=graph_embeds.device)

    # cross entropy losses
    loss_graph = F.cross_entropy(logits_per_graph, labels)
    loss_assay = F.cross_entropy(logits_per_assay, labels)

    return (loss_graph + loss_assay) / 2


def main():
    if not torch.cuda.is_available():
        print("\nError: No CUDA GPU detected")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Training on device: {device}")

    raw_gin = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0)
    model = FusionModel(
        gnn_model=raw_gin,
        lincs_input_dim=978,
        chembl_input_dim=1283,
        gnn_feat_dim=512,
        embed_dim=256,
    ).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch, train_losses, val_losses, run_timestamp = load_checkpoint(
        model, optimiser
    )

    if start_epoch == 0:
        if not PRETRAINED_GIN_PATH.exists():
            print(
                f"\nError: Pre-trained GNN weights not found at: {PRETRAINED_GIN_PATH}"
            )
            sys.exit(1)
        load_pretrained_gnn(PRETRAINED_GIN_PATH, model.gnn)

    dataset = DrugDB(db_path=str(DATA_DIR / "drugs.db"))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    total_epochs = 20
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch"
        )

        if epoch == start_epoch:
            print(f"Profiling Epoch {epoch + 1}...")
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for batch in train_loader:
                    if batch is None:
                        continue
                    batch = batch.to(device)
                    optimiser.zero_grad()

                    with record_function("model_forward"):
                        graph_emb, assay_emb, logit_scale = model(
                            batch, batch.lincs, batch.chembl
                        )

                    with record_function("model_backward"):
                        loss = clip_loss(graph_emb, assay_emb, logit_scale)
                        loss.backward()

                    optimiser.step()
                    epoch_train_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        else:
            for batch in pbar:
                if batch is None:
                    continue
                batch = batch.to(device)
                optimiser.zero_grad()

                graph_emb, assay_emb, logit_scale = model(
                    batch, batch.lincs, batch.chembl
                )

                loss = clip_loss(graph_emb, assay_emb, logit_scale)
                loss.backward()
                optimiser.step()

                epoch_train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)
                graph_emb, assay_emb, logit_scale = model(
                    batch, batch.lincs, batch.chembl
                )
                val_loss = clip_loss(graph_emb, assay_emb, logit_scale)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_duration = time.time() - start_time
        est_remaining = (epoch_duration * (total_epochs - (epoch + 1))) / 60

        print(
            f"Epoch [{epoch + 1}/{total_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"{epoch_duration:.1f}s/epoch | Est. Remaining: {est_remaining:.1f}m"
        )

        save_checkpoint(
            model, optimiser, epoch, train_losses, val_losses, run_timestamp
        )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = SAVE_DIR / f"fusion_model_{run_timestamp}.pth"

    torch.save(model.state_dict(), str(model_save_path))
    print(f"Model weights saved to '{model_save_path}'")

    if CHECKPOINT_LATEST.exists():
        CHECKPOINT_LATEST.unlink()
        print("Latest checkpoint removed")

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(
        range(1, total_epochs + 1), val_losses, label="Validation Loss", marker="x"
    )
    plt.title(f"Contrastive Loss over Epochs ({run_timestamp})")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.legend()
    plt.grid(True)

    plot_save_path = SAVE_DIR / f"training_curve_{run_timestamp}.png"
    plt.savefig(str(plot_save_path))
    plt.close()
    print(f"Training curve saved to '{plot_save_path}'")


if __name__ == "__main__":
    main()
