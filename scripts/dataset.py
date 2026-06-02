from pathlib import Path
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_raster import safe_std


class GeoAIHeightDataset(Dataset):
    def __init__(self, tiles_dir, stats_path):
        self.tiles_dir = Path(tiles_dir)
        self.img_files = sorted(glob.glob(str(self.tiles_dir / "img_*.npy")))
        self.lbl_files = sorted(glob.glob(str(self.tiles_dir / "lbl_*.npy")))

        if len(self.img_files) != len(self.lbl_files):
            raise ValueError(
                f"Jumlah img dan lbl tidak sama: {len(self.img_files)} vs {len(self.lbl_files)}"
            )

        stats = np.load(stats_path)
        self.mean = stats["mean"].astype(np.float32)
        self.std = safe_std(stats["std"])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(self.img_files[idx]).astype(np.float32)
        lbl = np.load(self.lbl_files[idx]).astype(np.float32)

        if img.shape[0] != len(self.mean):
            raise ValueError(
                f"Band image {self.img_files[idx]} = {img.shape[0]}, statistik = {len(self.mean)}"
            )

        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        lbl = lbl[None, :, :]

        return torch.from_numpy(img), torch.from_numpy(lbl)
