from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


def list_geotiffs(folder: Path) -> list[Path]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted({p.resolve() for p in files})


def is_probably_input_raster(path: Path) -> bool:
    name = path.name.lower()
    blocked = ["glad", "prediksi", "prediction", "height_proxy", "carbon"]
    return not any(token in name for token in blocked)


def validate_multiband_raster(path: Path, expected_bands: int) -> None:
    with rasterio.open(path) as src:
        if src.count != expected_bands:
            raise ValueError(f"{path.name} punya {src.count} band, bukan {expected_bands}.")
        if src.crs is None:
            raise ValueError(f"{path.name} tidak punya CRS.")
        if src.transform is None:
            raise ValueError(f"{path.name} tidak punya transform geospasial.")


def safe_std(std: np.ndarray) -> np.ndarray:
    std = std.astype(np.float32)
    return np.where(std <= 0, 1.0, std).astype(np.float32)


def normalize_image(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = safe_std(std)
    return (img.astype(np.float32) - mean[:, None, None]) / std[:, None, None]


def write_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
