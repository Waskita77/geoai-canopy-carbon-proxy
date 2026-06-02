import numpy as np


def rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.abs(pred.astype(np.float64) - target.astype(np.float64))
    return float(np.mean(diff))


def rmse_from_mse(mse: float) -> float:
    return float(np.sqrt(max(float(mse), 0.0)))
