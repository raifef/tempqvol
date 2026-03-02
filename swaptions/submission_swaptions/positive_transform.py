from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FloorMode = Literal["train_p01", "train_p001", "fixed"]


@dataclass
class PositiveLogTransform:
    """Stable log transform for non-negative targets with train-only floor fitting."""

    floor_mode: FloorMode = "train_p001"
    floor_value: float | None = None
    eps_min: float = 1e-8
    shift: float = 0.0
    floor: float = 1e-8
    eps: float = 1e-8
    fitted: bool = False

    def fit(self, y_train: np.ndarray) -> "PositiveLogTransform":
        y = np.asarray(y_train, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
        if y.size == 0:
            self.shift = 0.0
            self.floor = float(max(self.eps_min, 1e-8))
            self.eps = float(max(self.eps_min, self.floor))
            self.fitted = True
            return self

        min_raw = float(np.min(y))
        self.shift = float(max(0.0, -min_raw + 1e-8))
        y_pos = y + float(self.shift)
        y_pos = np.maximum(y_pos, 0.0)
        pos = y_pos[y_pos > 0.0]

        mode = str(self.floor_mode).strip().lower()
        if mode == "fixed":
            floor = float(max(self.eps_min, 0.0 if self.floor_value is None else self.floor_value))
        else:
            q = 1.0 if mode == "train_p01" else 0.1
            if pos.size == 0:
                floor = float(max(self.eps_min, 1e-8))
            else:
                floor = float(max(self.eps_min, np.percentile(pos, q)))
        self.floor = float(max(self.eps_min, floor))
        self.eps = float(max(self.eps_min, self.floor))
        self.fitted = True
        return self

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("PositiveLogTransform must be fit before use.")

    def forward(self, y: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        y_arr = np.asarray(y, dtype=np.float64)
        y_pos = np.maximum(y_arr + float(self.shift), float(self.floor))
        return np.log(y_pos + float(self.eps))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        z_arr = np.asarray(z, dtype=np.float64)
        y_pos = np.exp(np.clip(z_arr, -60.0, 60.0)) - float(self.eps)
        y_pos = np.maximum(y_pos, float(self.floor))
        return y_pos - float(self.shift)

    def clamp(self, y: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        y_arr = np.asarray(y, dtype=np.float64)
        y_pos = np.maximum(y_arr + float(self.shift), float(self.floor))
        return y_pos - float(self.shift)

