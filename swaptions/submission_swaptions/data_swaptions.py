from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


COL_RE = re.compile(r"^(Strike|Tenor)\s*:\s*([^;]+)\s*;\s*Maturity\s*:\s*(.+)$")


@dataclass
class PreparedDataset:
    level: int
    data_dir: str
    surface_cols: list[str]
    dates: np.ndarray
    observed: np.ndarray
    target: np.ndarray
    filled: np.ndarray
    mask: np.ndarray
    by_maturity: dict[float, list[tuple[float, int]]]
    by_axis: dict[float, list[tuple[float, int]]]
    missing_summary_daily: np.ndarray
    temporal_daily: np.ndarray
    missing_summary_seq: np.ndarray
    temporal_seq: np.ndarray
    x_seq: np.ndarray
    y_seq: np.ndarray
    m_seq: np.ndarray
    starts: np.ndarray
    train_slice: slice
    val_slice: slice
    test_slice: slice
    phase: np.ndarray
    day_index: np.ndarray
    dt_days: np.ndarray
    cycle_mismatch_count: int


def _to_float(token: str, *, default: float = 0.0) -> float:
    t = str(token).strip()
    if not t:
        return float(default)
    tu = t.upper()
    mult = 1.0
    if tu.endswith("Y"):
        tu = tu[:-1]
    elif tu.endswith("M"):
        tu = tu[:-1]
        mult = 1.0 / 12.0
    try:
        return float(tu) * mult
    except Exception:
        return float(default)


def parse_surface_structure(
    surface_cols: list[str],
) -> tuple[dict[float, list[tuple[float, int]]], dict[float, list[tuple[float, int]]], str]:
    by_maturity: dict[float, list[tuple[float, int]]] = defaultdict(list)
    by_axis: dict[float, list[tuple[float, int]]] = defaultdict(list)
    axis_name = ""

    for idx, col in enumerate(surface_cols):
        m = COL_RE.match(col)
        if not m:
            raise ValueError(f"Invalid surface column format: {col}")
        axis_name = m.group(1).strip()
        axis_val = _to_float(m.group(2), default=float(idx))
        maturity = _to_float(m.group(3), default=0.0)
        by_maturity[maturity].append((axis_val, idx))
        by_axis[axis_val].append((maturity, idx))

    for maturity in by_maturity:
        by_maturity[maturity].sort(key=lambda x: x[0])
    for axis in by_axis:
        by_axis[axis].sort(key=lambda x: x[0])
    return by_maturity, by_axis, axis_name


def _dataset_urls(data_dir: str | Path) -> tuple[str, str]:
    data_dir_s = str(data_dir).strip()
    if not data_dir_s:
        data_dir_s = "Quandela/Challenge_Swaptions"
    if data_dir_s.startswith("hf://"):
        data_dir_s = data_dir_s[len("hf://") :]
    if "/" not in data_dir_s:
        data_dir_s = f"Quandela/{data_dir_s}"
    base = f"https://huggingface.co/datasets/{data_dir_s}/resolve/main"
    level1 = f"{base}/level-1_Future_prediction/train.csv?download=true"
    level2 = f"{base}/level-2_Missing_data_prediction/train_level2.csv?download=true"
    return level1, level2


def _read_csv(data_dir: str | Path, rel_path: str) -> pd.DataFrame:
    base = Path(str(data_dir))
    local = base / rel_path
    if local.exists():
        return pd.read_csv(local)
    url1, url2 = _dataset_urls(data_dir)
    if rel_path.endswith("train.csv"):
        return pd.read_csv(url1)
    if rel_path.endswith("train_level2.csv"):
        return pd.read_csv(url2)
    raise FileNotFoundError(f"Unsupported data file: {rel_path}")


def load_level_data(
    data_dir: str | Path,
    level: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    dict[float, list[tuple[float, int]]],
    dict[float, list[tuple[float, int]]],
]:
    level1_rel = "level-1_Future_prediction/train.csv"
    level2_rel = "level-2_Missing_data_prediction/train_level2.csv"

    df_level1 = _read_csv(data_dir, level1_rel)
    if "Date" not in df_level1.columns:
        raise ValueError("Level-1 CSV missing Date column.")

    surface_cols = [c for c in df_level1.columns if c != "Date"]
    by_maturity, by_axis, _ = parse_surface_structure(surface_cols)

    if int(level) == 1:
        df_obs = df_level1.copy()
        target = df_level1[surface_cols].to_numpy(dtype=float)
    elif int(level) == 2:
        df_obs = _read_csv(data_dir, level2_rel)
        if "Date" not in df_obs.columns:
            raise ValueError("Level-2 CSV missing Date column.")
        df_idx = df_level1.set_index("Date")
        target = df_idx.loc[df_obs["Date"].tolist(), surface_cols].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported level: {level}")

    observed = df_obs[surface_cols].to_numpy(dtype=float)
    mask = np.isfinite(observed).astype(float)
    dates = df_obs["Date"].to_numpy(dtype=str)
    return observed, target, mask, surface_cols, dates, by_maturity, by_axis


def compute_missingness_summaries(
    mask_obs: np.ndarray,
    by_maturity: dict[float, list[tuple[float, int]]],
    by_axis: dict[float, list[tuple[float, int]]],
) -> np.ndarray:
    missing = 1.0 - mask_obs
    mats = sorted(by_maturity.keys())
    axes = sorted(by_axis.keys())
    total = missing.mean(axis=1, keepdims=True)

    mat_parts = []
    for mat in mats:
        idxs = [idx for _, idx in by_maturity[mat]]
        mat_parts.append(missing[:, idxs].mean(axis=1, keepdims=True))

    axis_parts = []
    for axis in axes:
        idxs = [idx for _, idx in by_axis[axis]]
        axis_parts.append(missing[:, idxs].mean(axis=1, keepdims=True))

    return np.concatenate([total, *mat_parts, *axis_parts], axis=1)


def impute_ffill_interp(observed: np.ndarray, by_maturity: dict[float, list[tuple[float, int]]]) -> np.ndarray:
    arr = pd.DataFrame(observed).ffill().bfill().to_numpy(dtype=float)

    for day in range(arr.shape[0]):
        row = arr[day]
        for entries in by_maturity.values():
            axis_vals = np.array([a for a, _ in entries], dtype=float)
            idxs = np.array([idx for _, idx in entries], dtype=int)
            vals = row[idxs]
            finite = np.isfinite(vals)
            if finite.all():
                continue
            if finite.sum() == 0:
                continue
            if finite.sum() == 1:
                vals[~finite] = vals[finite][0]
            else:
                vals[~finite] = np.interp(axis_vals[~finite], axis_vals[finite], vals[finite])
            row[idxs] = vals
        arr[day] = row

    col_mean = np.nanmean(arr, axis=0)
    global_mean = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
    col_mean = np.where(np.isfinite(col_mean), col_mean, global_mean)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(col_mean, np.where(missing)[1])
    return arr


def impute_svd_iterative(observed: np.ndarray, rank: int = 12, n_iter: int = 15, seed: int = 0) -> np.ndarray:
    arr = observed.copy()
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr

    col_mean = np.nanmean(arr, axis=0)
    global_mean = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
    col_mean = np.where(np.isfinite(col_mean), col_mean, global_mean)
    arr[missing] = np.take(col_mean, np.where(missing)[1])

    n_components = min(rank, arr.shape[0] - 1, arr.shape[1] - 1)
    if n_components <= 0:
        return arr

    for _ in range(n_iter):
        mean = arr.mean(axis=0, keepdims=True)
        centered = arr - mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            svd = TruncatedSVD(n_components=n_components, random_state=seed)
            z = svd.fit_transform(centered)
            recon = svd.inverse_transform(z) + mean
        arr[missing] = 0.6 * arr[missing] + 0.4 * recon[missing]
    return arr


def build_windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    missing_summary_daily: np.ndarray,
    temporal_daily: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_days, n_features = inputs.shape
    n_windows = n_days - lookback - horizon + 1
    if n_windows <= 0:
        raise ValueError("Not enough rows for requested window/horizon.")

    x_seq = np.zeros((n_windows, lookback, n_features), dtype=float)
    y_seq = np.zeros((n_windows, horizon, n_features), dtype=float)
    m_seq = np.zeros((n_windows, lookback, n_features), dtype=float)
    ms_seq = np.zeros((n_windows, lookback, missing_summary_daily.shape[1]), dtype=float)
    ts_seq = np.zeros((n_windows, lookback, temporal_daily.shape[1]), dtype=float)
    starts = np.arange(n_windows, dtype=int)
    for i in range(n_windows):
        x_seq[i] = inputs[i : i + lookback]
        y_seq[i] = targets[i + lookback : i + lookback + horizon]
        m_seq[i] = mask[i : i + lookback]
        ms_seq[i] = missing_summary_daily[i : i + lookback]
        ts_seq[i] = temporal_daily[i : i + lookback]
    return x_seq, y_seq, m_seq, ms_seq, ts_seq, starts


def _expected_cycle_mask() -> np.ndarray:
    # 19-day cycle:
    # 3 observed days, 1 gap, then 5x(2 observed, 1 gap) => 13 observed in 19.
    return np.array(
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        dtype=int,
    )


def _compute_temporal_features(
    dates: np.ndarray,
    day_has_observation: np.ndarray,
    use_cycle_phase: bool,
    check_cycle_pattern: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    dates_ts = pd.to_datetime(pd.Series(dates), errors="coerce")
    valid_idx = np.where(dates_ts.notna().to_numpy())[0]
    if valid_idx.size > 0:
        date0 = dates_ts.iloc[int(valid_idx[0])]
        day_index = (dates_ts - date0).dt.days.to_numpy(dtype=float)
        if np.isnan(day_index).any():
            fallback = np.arange(day_index.shape[0], dtype=float)
            day_index = np.where(np.isfinite(day_index), day_index, fallback)
    else:
        day_index = np.arange(len(dates), dtype=float)
    day_index = day_index.astype(int)

    phase = np.mod(day_index, 19).astype(int)
    dt_days = np.diff(day_index, prepend=(day_index[0] - 1)).astype(float)
    dt_days = np.clip(np.nan_to_num(dt_days, nan=1.0, posinf=30.0, neginf=1.0), 0.0, 30.0)

    if bool(use_cycle_phase):
        phase_onehot = np.eye(19, dtype=float)[phase]
        phase_sin = np.sin(2.0 * np.pi * phase / 19.0)[:, None]
        phase_cos = np.cos(2.0 * np.pi * phase / 19.0)[:, None]
        temporal_daily = np.concatenate([dt_days[:, None], phase_onehot, phase_sin, phase_cos], axis=1)
    else:
        temporal_daily = dt_days[:, None]

    mismatch_count = 0
    if bool(check_cycle_pattern):
        exp_mask = _expected_cycle_mask()
        n_check = min(int(day_has_observation.shape[0]), int(2 * exp_mask.shape[0]))
        if n_check >= exp_mask.shape[0]:
            tiled = np.tile(exp_mask, int(np.ceil(n_check / exp_mask.shape[0])))[:n_check]
            obs = day_has_observation[:n_check].astype(int)
            mismatch_count = int(np.sum(np.abs(obs - tiled)))
            if mismatch_count > 1:
                warnings.warn(
                    f"Cycle-pattern sanity check: mismatch_count={mismatch_count} over first {n_check} days "
                    "(expected 19-day 13-observed pattern). Proceeding.",
                    RuntimeWarning,
                )
                print(
                    f"[cycle_phase] warning: mismatch_count={mismatch_count} over first {n_check} days "
                    "(expected <=1)."
                )

    return temporal_daily, phase, day_index, dt_days, mismatch_count


def split_windows(n_windows: int, *, lookback: int, horizon: int) -> tuple[slice, slice, slice]:
    """Split window indices into train/val/test with non-overlapping raw time support.

    Windows are indexed by their start day, and each window consumes days in
    [start, start + lookback + horizon - 1]. To avoid leakage across splits,
    we insert a gap of (lookback + horizon - 1) start-indices between adjacent
    splits so no raw day is shared by train/val/test windows.
    """
    gap = int(max(0, int(lookback) + int(horizon) - 1))
    effective = int(n_windows) - 2 * gap
    if effective <= 0:
        raise ValueError(
            "Not enough windows to create leakage-safe train/val/test splits "
            f"with lookback={int(lookback)}, horizon={int(horizon)}."
        )

    n_train = int(0.70 * effective)
    n_val = int(0.15 * effective)
    n_test = effective - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split produced empty train/val/test segment after applying anti-leakage gaps.")

    train_start = 0
    train_stop = n_train
    val_start = train_stop + gap
    val_stop = val_start + n_val
    test_start = val_stop + gap
    test_stop = test_start + n_test
    if test_stop > int(n_windows):
        raise ValueError("Split indexing overflowed total windows; check split configuration.")
    return slice(train_start, train_stop), slice(val_start, val_stop), slice(test_start, test_stop)


def prepare_windowed_dataset(
    data_dir: str | Path,
    level: int,
    lookback: int = 20,
    horizon: int = 10,
    imputer: str = "ffill_interp",
    seed: int = 0,
    use_cycle_phase: bool = True,
) -> PreparedDataset:
    observed, target, mask, surface_cols, dates, by_maturity, by_axis = load_level_data(data_dir, level)

    if imputer == "ffill_interp":
        filled = impute_ffill_interp(observed, by_maturity)
    elif imputer == "svd_iterative":
        filled = impute_svd_iterative(observed, rank=12, n_iter=15, seed=seed)
    else:
        raise ValueError(f"Unknown imputer: {imputer}")

    missing_summary_daily = compute_missingness_summaries(mask, by_maturity, by_axis)
    day_has_observation = (np.nanmean(mask, axis=1) > 0.5).astype(int)
    temporal_daily, phase, day_index, dt_days, cycle_mismatch_count = _compute_temporal_features(
        dates=dates,
        day_has_observation=day_has_observation,
        use_cycle_phase=bool(use_cycle_phase),
        check_cycle_pattern=bool(int(level) == 2),
    )
    x_seq, y_seq, m_seq, ms_seq, ts_seq, starts = build_windows(
        filled,
        target,
        mask,
        missing_summary_daily,
        temporal_daily,
        lookback,
        horizon,
    )
    train_slice, val_slice, test_slice = split_windows(
        x_seq.shape[0],
        lookback=int(lookback),
        horizon=int(horizon),
    )

    return PreparedDataset(
        level=int(level),
        data_dir=str(data_dir),
        surface_cols=surface_cols,
        dates=dates,
        observed=observed,
        target=target,
        filled=filled,
        mask=mask,
        by_maturity=by_maturity,
        by_axis=by_axis,
        missing_summary_daily=missing_summary_daily,
        temporal_daily=temporal_daily,
        missing_summary_seq=ms_seq,
        temporal_seq=ts_seq,
        x_seq=x_seq,
        y_seq=y_seq,
        m_seq=m_seq,
        starts=starts,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        phase=phase,
        day_index=day_index,
        dt_days=dt_days,
        cycle_mismatch_count=int(cycle_mismatch_count),
    )


class FactorZScoreNormalizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, arr_2d: np.ndarray) -> "FactorZScoreNormalizer":
        mean = np.mean(arr_2d, axis=0, keepdims=True)
        std = np.std(arr_2d, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, arr_2d: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted.")
        return (arr_2d - self.mean_) / self.std_

    def inverse_transform(self, arr_2d: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted.")
        return arr_2d * self.std_ + self.mean_

    def transform_seq(self, arr_3d: np.ndarray) -> np.ndarray:
        shp = arr_3d.shape
        return self.transform(arr_3d.reshape(-1, shp[-1])).reshape(shp)

    def inverse_transform_seq(self, arr_3d: np.ndarray) -> np.ndarray:
        shp = arr_3d.shape
        return self.inverse_transform(arr_3d.reshape(-1, shp[-1])).reshape(shp)


def load_surface_schema(data_dir: str | Path) -> tuple[list[str], list[str]]:
    df = _read_csv(data_dir, "level-1_Future_prediction/train.csv")
    if "Date" not in df.columns:
        raise ValueError("Level-1 CSV missing Date column.")
    dates = df["Date"].astype(str).tolist()
    surface_cols = [c for c in df.columns if c != "Date"]
    return dates, surface_cols
