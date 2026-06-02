from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from config import (
    GLAD_FILE,
    RAW_TRAIN_DIR,
    ALIGNED_LABEL_DIR,
    EXPECTED_BANDS,
    NODATA,
    ensure_directories,
)
from utils_raster import list_geotiffs, is_probably_input_raster, validate_multiband_raster


def align_glad_to_planetscope() -> None:
    ensure_directories()

    if not GLAD_FILE.exists():
        raise FileNotFoundError(f"File GLAD tidak ditemukan: {GLAD_FILE}")

    planet_files = [p for p in list_geotiffs(RAW_TRAIN_DIR) if is_probably_input_raster(p)]
    if not planet_files:
        raise FileNotFoundError(f"Tidak ada PlanetScope/SuperDove training .tif di {RAW_TRAIN_DIR}")

    with rasterio.open(GLAD_FILE) as src_glad:
        if src_glad.crs is None:
            raise ValueError("File GLAD tidak punya CRS.")

        for planet_path in planet_files:
            validate_multiband_raster(planet_path, EXPECTED_BANDS)
            base_name = planet_path.stem
            out_path = ALIGNED_LABEL_DIR / f"{base_name}_glad2019_aligned_to_planet.tif"

            print(f"✂️ Align GLAD 2019 ke grid PlanetScope: {base_name}")

            with rasterio.open(planet_path) as src_ps:
                dst_array = np.full((src_ps.height, src_ps.width), NODATA, dtype=np.float32)

                reproject(
                    source=rasterio.band(src_glad, 1),
                    destination=dst_array,
                    src_transform=src_glad.transform,
                    src_crs=src_glad.crs,
                    src_nodata=src_glad.nodata,
                    dst_transform=src_ps.transform,
                    dst_crs=src_ps.crs,
                    dst_nodata=NODATA,
                    resampling=Resampling.bilinear,
                )

                meta = src_ps.meta.copy()
                meta.update(
                    count=1,
                    dtype="float32",
                    nodata=NODATA,
                    compress="lzw",
                    predictor=2,
                )

                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(dst_array, 1)

            print(f"✅ Label aligned tersimpan: {out_path}")


if __name__ == "__main__":
    align_glad_to_planetscope()
