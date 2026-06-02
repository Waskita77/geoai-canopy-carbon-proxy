from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm

from config import (
    RAW_INFERENCE_DIR,
    HEIGHT_DIR,
    MODEL_PATH,
    EXPECTED_BANDS,
    TILE_SIZE,
    NODATA,
    ensure_directories,
)
from utils_raster import list_geotiffs, is_probably_input_raster, normalize_image, safe_std
from unet_model import UNetGeoAI


HEIGHT_CAP_M = 50.0


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model belum ada: {MODEL_PATH}. Jalankan 04_train_unet.py dulu.")

    # PyTorch 2.6+ default weights_only=True, padahal checkpoint ini menyimpan metadata normalisasi.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise ValueError("Checkpoint tidak valid. Harus berisi model_state_dict dan normalization.")

    in_channels = int(checkpoint.get("in_channels", EXPECTED_BANDS))
    mean = np.array(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = safe_std(np.array(checkpoint["normalization"]["std"], dtype=np.float32))
    state_dict = checkpoint["model_state_dict"]

    model = UNetGeoAI(in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, mean, std, in_channels


def run_spatial_inference() -> None:
    ensure_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, in_channels = load_model(device)

    input_files = [p for p in list_geotiffs(RAW_INFERENCE_DIR) if is_probably_input_raster(p)]
    if not input_files:
        raise FileNotFoundError(f"Tidak ada raster inference .tif di {RAW_INFERENCE_DIR}")

    print(f"=== Spatial Inference: {len(input_files)} raster ===")
    print(f"Device: {device}")
    print(f"Height cap: {HEIGHT_CAP_M} m")

    for input_path in input_files:
        base_name = input_path.stem
        out_raster = HEIGHT_DIR / f"{base_name}_canopy_height_proxy.tif"

        with rasterio.open(input_path) as src:
            if src.count != in_channels:
                raise ValueError(f"{input_path.name} punya {src.count} band, model butuh {in_channels}.")

            meta = src.meta.copy()
            meta.update(
                count=1,
                dtype="float32",
                nodata=NODATA,
                compress="lzw",
                predictor=2,
            )

            with rasterio.open(out_raster, "w", **meta) as dst:
                windows = []
                for y in range(0, src.height, TILE_SIZE):
                    for x in range(0, src.width, TILE_SIZE):
                        w = min(TILE_SIZE, src.width - x)
                        h = min(TILE_SIZE, src.height - y)
                        windows.append((x, y, w, h))

                for x, y, w, h in tqdm(windows, desc=f"Inference {base_name}", leave=False):
                    window = Window(x, y, w, h)
                    img = src.read(window=window).astype(np.float32)

                    valid_mask = np.isfinite(img).all(axis=0)
                    if src.nodata is not None:
                        valid_mask &= ~(img == src.nodata).any(axis=0)
                    valid_mask &= ~(img == 0).all(axis=0)

                    padded = np.zeros((in_channels, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    padded[:, :h, :w] = img
                    padded = normalize_image(padded, mean, std)

                    with torch.no_grad():
                        tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
                        pred = model(tensor).squeeze().cpu().numpy().astype(np.float32)

                    # Cap ke 50 m untuk mengurangi outlier model.
                    # Output tetap canopy height proxy, bukan ground truth tinggi kanopi.
                    pred = np.clip(pred[:h, :w], 0, HEIGHT_CAP_M)
                    pred = np.where(valid_mask, pred, NODATA).astype(np.float32)
                    dst.write(pred, 1, window=window)

        print(f"✅ Height proxy raster: {out_raster}")


if __name__ == "__main__":
    run_spatial_inference()