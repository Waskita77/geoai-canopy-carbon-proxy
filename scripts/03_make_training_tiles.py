from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from config import (
    RAW_TRAIN_DIR,
    ALIGNED_LABEL_DIR,
    TILES_DIR,
    STATS_PATH,
    SCENE_INDEX_PATH,
    EXPECTED_BANDS,
    TILE_SIZE,
    STRIDE,
    NODATA,
    ensure_directories,
)
from utils_raster import list_geotiffs, is_probably_input_raster


def clean_old_tiles() -> None:
    for pattern in ("img_*.npy", "lbl_*.npy"):
        for p in TILES_DIR.glob(pattern):
            p.unlink()


def is_valid_tile(img: np.ndarray, lbl: np.ndarray, ps_nodata) -> bool:
    if img.shape[0] != EXPECTED_BANDS:
        return False
    if not np.isfinite(img).all() or not np.isfinite(lbl).all():
        return False
    if ps_nodata is not None and (img == ps_nodata).any():
        return False
    if (img == 0).all():
        return False
    if (lbl == NODATA).any():
        return False
    if (lbl < 0).any():
        return False
    # Tile dengan semua label 0 cenderung kurang informatif untuk canopy regression.
    # Masih boleh dipakai kalau ingin model kuat mendeteksi non-vegetasi.
    return True


def generate_tiles() -> None:
    ensure_directories()
    clean_old_tiles()

    planet_files = [p for p in list_geotiffs(RAW_TRAIN_DIR) if is_probably_input_raster(p)]
    if not planet_files:
        raise FileNotFoundError(f"Tidak ada PlanetScope/SuperDove training .tif di {RAW_TRAIN_DIR}")

    tile_idx = 0
    band_sum = np.zeros(EXPECTED_BANDS, dtype=np.float64)
    band_sumsq = np.zeros(EXPECTED_BANDS, dtype=np.float64)
    pixel_count = 0

    index_rows = []

    print("=== Membuat tile training 256x256 ===")

    for planet_path in planet_files:
        base_name = planet_path.stem
        glad_path = ALIGNED_LABEL_DIR / f"{base_name}_glad2019_aligned_to_planet.tif"

        if not glad_path.exists():
            print(f"⚠️ Skip {base_name}: label aligned belum ada.")
            continue

        with rasterio.open(planet_path) as src_ps, rasterio.open(glad_path) as src_lbl:
            if src_ps.count != EXPECTED_BANDS:
                raise ValueError(f"{planet_path.name} punya {src_ps.count} band, bukan {EXPECTED_BANDS}.")
            if (src_ps.width, src_ps.height) != (src_lbl.width, src_lbl.height):
                raise ValueError(f"Dimensi PlanetScope dan label tidak sama untuk {base_name}.")
            if src_ps.transform != src_lbl.transform or src_ps.crs != src_lbl.crs:
                raise ValueError(f"Grid/CRS PlanetScope dan label tidak sama untuk {base_name}.")

            max_y = src_ps.height - TILE_SIZE + 1
            max_x = src_ps.width - TILE_SIZE + 1
            windows = [(x, y) for y in range(0, max_y, STRIDE) for x in range(0, max_x, STRIDE)]

            scene_tile_count = 0

            for x, y in tqdm(windows, desc=f"Tiling {base_name}", leave=False):
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                img = src_ps.read(window=window).astype(np.float32)
                lbl = src_lbl.read(1, window=window).astype(np.float32)

                if not is_valid_tile(img, lbl, src_ps.nodata):
                    continue

                img_name = f"img_{tile_idx:06d}.npy"
                lbl_name = f"lbl_{tile_idx:06d}.npy"

                np.save(TILES_DIR / img_name, img)
                np.save(TILES_DIR / lbl_name, lbl)

                flat = img.reshape(EXPECTED_BANDS, -1).astype(np.float64)
                band_sum += flat.sum(axis=1)
                band_sumsq += (flat ** 2).sum(axis=1)
                pixel_count += flat.shape[1]

                index_rows.append({
                    "tile_id": tile_idx,
                    "scene": base_name,
                    "img_file": img_name,
                    "lbl_file": lbl_name,
                    "x": x,
                    "y": y,
                })

                tile_idx += 1
                scene_tile_count += 1

        print(f"✅ {base_name}: {scene_tile_count} tile valid")

    if tile_idx == 0 or pixel_count == 0:
        raise RuntimeError("Tidak ada tile valid. Cek overlap GLAD, nodata, ukuran citra, dan nama file.")

    mean = band_sum / pixel_count
    variance = np.maximum((band_sumsq / pixel_count) - (mean ** 2), 1e-12)
    std = np.sqrt(variance)

    np.savez(STATS_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))

    with SCENE_INDEX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_id", "scene", "img_file", "lbl_file", "x", "y"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"✅ Total tile valid: {tile_idx}")
    print(f"✅ Statistik normalisasi: {STATS_PATH}")
    print(f"✅ Index tile per scene: {SCENE_INDEX_PATH}")
    print(f"mean={mean}")
    print(f"std ={std}")


if __name__ == "__main__":
    generate_tiles()
