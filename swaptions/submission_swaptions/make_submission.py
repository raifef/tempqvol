from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_swaptions import load_surface_schema


def read_sample_submission(data_dir: str | Path) -> pd.DataFrame | None:
    data_dir_p = Path(str(data_dir))
    sample_path = data_dir_p / "sample_submission.csv"
    if not sample_path.exists():
        return None
    sample_df = pd.read_csv(sample_path)
    if "Date" not in sample_df.columns:
        return None
    if sample_df.shape[1] < 2:
        return None
    return sample_df


def infer_submission_horizon(data_dir: str | Path, default_h: int) -> int:
    sample_df = read_sample_submission(data_dir)
    if sample_df is None:
        return int(default_h)
    h = int(sample_df.shape[0])
    return h if h > 0 else int(default_h)


def _surface_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Date"]


def _align_horizon(pred_df: pd.DataFrame, target_h: int) -> pd.DataFrame:
    if target_h < 1:
        raise ValueError(f"target_h must be >= 1, got {target_h}.")
    n = int(pred_df.shape[0])
    if n == target_h:
        return pred_df
    if n > target_h:
        print(f"[submission] truncating predictions from {n} to {target_h} rows.")
        return pred_df.iloc[:target_h].reset_index(drop=True)
    if n < 1:
        raise ValueError("Predictions are empty; cannot pad to target horizon.")
    pad_n = target_h - n
    print(f"[submission] padding predictions from {n} to {target_h} rows with last prediction.")
    last_row = pred_df.iloc[[-1]].copy()
    pad_df = pd.concat([last_row] * pad_n, ignore_index=True)
    return pd.concat([pred_df.reset_index(drop=True), pad_df], ignore_index=True)


def _make_reference_template(
    data_dir: str | Path,
    target_h: int,
    fallback_surface_cols: list[str] | None = None,
) -> pd.DataFrame:
    sample_df = read_sample_submission(data_dir)
    if sample_df is not None:
        if int(sample_df.shape[0]) == int(target_h):
            return sample_df
        if int(sample_df.shape[0]) > int(target_h):
            return sample_df.iloc[: int(target_h)].reset_index(drop=True)
        if int(sample_df.shape[0]) >= 1:
            pad_n = int(target_h) - int(sample_df.shape[0])
            last = sample_df.iloc[[-1]].copy()
            pad = pd.concat([last] * pad_n, ignore_index=True)
            return pd.concat([sample_df.reset_index(drop=True), pad], ignore_index=True)

    try:
        train_dates, surface_cols = load_surface_schema(data_dir)
    except Exception:
        if not fallback_surface_cols:
            raise
        train_dates = []
        surface_cols = list(fallback_surface_cols)
    last_date = pd.to_datetime(train_dates[-1], errors="coerce") if train_dates else pd.NaT
    if pd.isna(last_date):
        future_dates = [f"T+{i + 1}" for i in range(int(target_h))]
    else:
        future_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(last_date + pd.Timedelta(days=1), periods=int(target_h))]
    out = pd.DataFrame({"Date": future_dates})
    for col in surface_cols:
        out[col] = np.nan
    return out


def _predictions_to_surface_frame(
    predicted_surfaces: np.ndarray | pd.DataFrame,
    sample_surface_cols: list[str],
) -> pd.DataFrame:
    if isinstance(predicted_surfaces, pd.DataFrame):
        pred_df = predicted_surfaces.copy()
        if "Date" in pred_df.columns:
            pred_df = pred_df.drop(columns=["Date"])
        if set(pred_df.columns) == set(sample_surface_cols):
            pred_df = pred_df.reindex(columns=sample_surface_cols)
        elif pred_df.shape[1] == len(sample_surface_cols):
            pred_df.columns = sample_surface_cols
        else:
            raise ValueError(
                "Predicted surface DataFrame columns do not match sample submission surface columns."
            )
    else:
        arr = np.asarray(predicted_surfaces, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"predicted_surfaces must be 2D [H, N_surface], got shape {arr.shape}.")
        if arr.shape[1] != len(sample_surface_cols):
            raise ValueError(
                f"Predicted surface width mismatch: got {arr.shape[1]}, expected {len(sample_surface_cols)}."
            )
        pred_df = pd.DataFrame(arr, columns=sample_surface_cols)

    values = np.nan_to_num(pred_df.to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    values = np.clip(values, 0.0, None)
    return pd.DataFrame(values, columns=sample_surface_cols)


def make_submission_from_predictions(
    data_dir: str | Path,
    predicted_surfaces: np.ndarray | pd.DataFrame,
    out_csv: str | Path,
    expected_surface_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    pred_rows = int(np.asarray(predicted_surfaces).shape[0]) if not isinstance(predicted_surfaces, pd.DataFrame) else int(predicted_surfaces.shape[0])
    if pred_rows < 1:
        raise ValueError("predicted_surfaces has no rows.")
    fallback_cols = list(expected_surface_cols) if expected_surface_cols is not None else None
    sample_df = _make_reference_template(data_dir=data_dir, target_h=pred_rows, fallback_surface_cols=fallback_cols)
    sample_cols = list(sample_df.columns)
    sample_surface_cols = _surface_columns(sample_df)

    if expected_surface_cols is not None:
        expected_set = set(expected_surface_cols)
        sample_set = set(sample_surface_cols)
        if expected_set != sample_set:
            missing = sorted(expected_set - sample_set)
            extra = sorted(sample_set - expected_set)
            raise ValueError(
                "Surface column set mismatch between model output schema and sample submission. "
                f"Missing in sample: {missing[:5]}{'...' if len(missing) > 5 else ''}; "
                f"Extra in sample: {extra[:5]}{'...' if len(extra) > 5 else ''}."
            )

    pred_df = _predictions_to_surface_frame(predicted_surfaces, sample_surface_cols)
    pred_df = _align_horizon(pred_df, target_h=int(sample_df.shape[0]))

    out_df = sample_df.copy()
    out_df.loc[:, sample_surface_cols] = pred_df.to_numpy(dtype=float)

    # Defensive schema checks.
    if list(out_df.columns) != sample_cols:
        raise RuntimeError("Submission column order changed unexpectedly.")
    if set(out_df.columns) != set(sample_cols):
        raise RuntimeError("Submission column set changed unexpectedly.")

    # Defensive value checks.
    surf_values = out_df[sample_surface_cols].to_numpy(dtype=float)
    if np.isnan(surf_values).any():
        raise ValueError("Submission contains NaN values in surface columns.")
    if np.isinf(surf_values).any():
        raise ValueError("Submission contains Inf values in surface columns.")
    if (surf_values < 0.0).any():
        raise ValueError("Submission contains negative values in surface columns.")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build submission CSV from predicted surfaces and sample template.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pred_csv", type=str, required=True, help="CSV containing predicted surface columns.")
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    pred_df = pd.read_csv(args.pred_csv)
    out_df = make_submission_from_predictions(
        data_dir=args.data_dir,
        predicted_surfaces=pred_df,
        out_csv=args.out_csv,
    )
    print(f"Saved submission CSV: {args.out_csv}")
    print(f"Submission shape: {out_df.shape}")


if __name__ == "__main__":
    _main()
