import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from config import (
    TILES_DIR,
    STATS_PATH,
    MODEL_PATH,
    TRAIN_LOG_PATH,
    SCENE_INDEX_PATH,
    EXPECTED_BANDS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VAL_RATIO,
    SEED,
    ensure_directories,
)
from dataset import GeoAIHeightDataset
from metrics import rmse_from_mse
from unet_model import UNetGeoAI


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, holdout_scene: str | None):
    if holdout_scene is None:
        val_size = max(1, int(VAL_RATIO * len(dataset))) if len(dataset) >= 10 else 0
        train_size = len(dataset) - val_size
        if val_size == 0:
            return dataset, None, "no_validation"
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED),
        )
        return train_ds, val_ds, f"random_tile_split_val_ratio_{VAL_RATIO}"

    if not SCENE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{SCENE_INDEX_PATH} belum ada. Jalankan 03_make_training_tiles.py dulu."
        )

    index_df = pd.read_csv(SCENE_INDEX_PATH)
    train_indices = index_df.index[index_df["scene"] != holdout_scene].tolist()
    val_indices = index_df.index[index_df["scene"] == holdout_scene].tolist()

    if not val_indices:
        available = sorted(index_df["scene"].unique().tolist())
        raise ValueError(f"Holdout scene '{holdout_scene}' tidak ditemukan. Pilihan: {available}")
    if not train_indices:
        raise ValueError("Semua tile masuk holdout. Butuh minimal 1 scene untuk training.")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        f"holdout_scene_{holdout_scene}",
    )


def train_unet(epochs: int, batch_size: int, lr: float, holdout_scene: str | None) -> None:
    ensure_directories()
    seed_everything(SEED)

    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Statistik normalisasi belum ada: {STATS_PATH}")
    if not list(TILES_DIR.glob("img_*.npy")):
        raise FileNotFoundError(f"Tile training belum ada di {TILES_DIR}")

    dataset = GeoAIHeightDataset(TILES_DIR, stats_path=STATS_PATH)
    train_ds, val_ds, split_mode = split_dataset(dataset, holdout_scene)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGeoAI(in_channels=EXPECTED_BANDS, out_channels=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    logs = []

    print("=== Training U-Net Canopy Height Proxy ===")
    print(f"Split mode : {split_mode}")
    print(f"Train tile : {len(train_ds)}")
    print(f"Val tile   : {len(val_ds) if val_ds is not None else 0}")
    print(f"Device     : {device}")
    print("Catatan: metrik adalah terhadap GLAD pseudo-label, bukan ground truth lapangan.")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs} train", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = images.size(0)
            train_loss_sum += loss.item() * n
            train_count += n

        train_mse = train_loss_sum / max(train_count, 1)
        row = {
            "epoch": epoch,
            "split_mode": split_mode,
            "train_mse": train_mse,
            "train_rmse_m": rmse_from_mse(train_mse),
            "val_mse": None,
            "val_rmse_m": None,
        }

        msg = f"Epoch [{epoch:03d}/{epochs}] Train RMSE: {row['train_rmse_m']:.4f} m"

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs} val", leave=False):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(images)
                    loss = criterion(preds, labels)

                    n = images.size(0)
                    val_loss_sum += loss.item() * n
                    val_count += n

            val_mse = val_loss_sum / max(val_count, 1)
            row["val_mse"] = val_mse
            row["val_rmse_m"] = rmse_from_mse(val_mse)
            msg += f" | Val RMSE: {row['val_rmse_m']:.4f} m"

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        logs.append(row)
        print(msg)

    stats = np.load(STATS_PATH)
    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "in_channels": EXPECTED_BANDS,
        "out_channels": 1,
        "normalization": {
            "mean": stats["mean"].astype(np.float32).tolist(),
            "std": stats["std"].astype(np.float32).tolist(),
        },
        "target": "GLAD Forest Height 2019 pseudo-label, meter",
        "split_mode": split_mode,
        "note": (
            "Model trained against GLAD 2019 pseudo-labels. "
            "Predicted 3 m output is a canopy height proxy/downscaled prediction, "
            "not field-validated ground truth."
        ),
    }

    torch.save(checkpoint, MODEL_PATH)
    pd.DataFrame(logs).to_csv(TRAIN_LOG_PATH, index=False)

    print(f"✅ Model tersimpan: {MODEL_PATH}")
    print(f"✅ Log training tersimpan: {TRAIN_LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--holdout-scene",
        type=str,
        default=None,
        help="Nama stem scene untuk validasi holdout. Contoh: scene_2021_01",
    )
    args = parser.parse_args()

    train_unet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        holdout_scene=args.holdout_scene,
    )


if __name__ == "__main__":
    main()
