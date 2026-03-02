from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _surface_origin_matrix(bundle: Any, lookback: int, *, split_slice: slice) -> np.ndarray:
    starts_eval = np.asarray(bundle.ds.starts[split_slice], dtype=int)
    origin_idx = starts_eval + int(lookback) - 1
    obs = np.asarray(bundle.ds.filled, dtype=float)
    origin = obs[origin_idx]
    origin = np.clip(np.nan_to_num(origin, nan=0.0, posinf=bundle.clip_cap, neginf=0.0), 0.0, bundle.clip_cap)
    return origin


def run_fast_sanity_checks(
    bundle: Any,
    eval_rows: list[dict[str, Any]],
    y_true_test_surface: np.ndarray,
    lookback: int,
    out_dir: Path,
    level: int,
    eps: float = 1e-8,
    sample_count: int = 25,
    near_zero_threshold: float = 1e-3,
    split: str = "test",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "level": int(level),
        "lookback": int(lookback),
        "eps": float(eps),
        "split": str(split),
        "warnings": [],
    }
    split_norm = str(split).strip().lower()
    if split_norm in {"val", "validation"}:
        split_slice = bundle.ds.val_slice
        x_eval_norm = np.asarray(bundle.x_val_norm, dtype=float)
    elif split_norm == "train":
        split_slice = bundle.ds.train_slice
        x_eval_norm = np.asarray(bundle.x_train_norm, dtype=float)
    else:
        split_slice = bundle.ds.test_slice
        x_eval_norm = np.asarray(bundle.x_test_norm, dtype=float)

    starts_eval = np.asarray(bundle.ds.starts[split_slice], dtype=int)
    n_eval = int(starts_eval.shape[0])
    horizon = int(y_true_test_surface.shape[1])
    if n_eval <= 0 or horizon <= 0:
        raise RuntimeError("Sanity checks failed: empty evaluation windows or horizon.")

    # A) Alignment and leakage check.
    rng = np.random.default_rng(0)
    horizon_choices = [1]
    if horizon >= 3:
        horizon_choices.append(3)
    horizon_choices.append(horizon)
    horizon_choices = sorted(set(int(h) for h in horizon_choices))
    samples = int(min(max(1, sample_count), n_eval * max(1, len(horizon_choices))))
    examples: list[dict[str, Any]] = []
    for _ in range(samples):
        i = int(rng.integers(0, n_eval))
        h = int(horizon_choices[int(rng.integers(0, len(horizon_choices)))])
        start = int(starts_eval[i])
        t = int(start + int(lookback) - 1)
        input_start = int(t - int(lookback) + 1)
        input_end = int(t)
        target_idx = int(t + h)
        if input_start != start:
            raise RuntimeError(
                f"Alignment check failed: input_start={input_start} != start={start} (window={i})."
            )
        if target_idx != int(start + int(lookback) + h - 1):
            raise RuntimeError(
                "Alignment check failed: target index mismatch "
                f"(got {target_idx}, expected {start + int(lookback) + h - 1})."
            )
        if target_idx <= input_end:
            raise RuntimeError(
                f"Leakage detected: target_idx={target_idx} overlaps input_end={input_end}."
            )
        if len(examples) < 10:
            examples.append(
                {
                    "window_idx": i,
                    "h": h,
                    "input_range": [input_start, input_end],
                    "target_idx": target_idx,
                }
            )
    report["alignment_check"] = {"sampled_cases": samples, "examples": examples}
    print(f"[sanity] alignment_check passed on {samples} sampled (t,h) pairs for split={split_norm}.")

    # B) Persistence implementation checks.
    floor = float(getattr(bundle, "surface_floor", 0.0))
    expected_surface = np.asarray(bundle.ds.x_seq[split_slice][:, -1:, :], dtype=float)
    expected_surface = np.clip(
        np.nan_to_num(expected_surface, nan=floor, posinf=bundle.clip_cap, neginf=floor),
        floor,
        bundle.clip_cap,
    )
    expected_surface = np.repeat(expected_surface, horizon, axis=1)
    persist_surface_row = next(
        (r for r in eval_rows if str(r.get("model", "")).strip().lower() == "persistence_surface_naive"),
        None,
    )
    if persist_surface_row is not None:
        pred_surface = np.asarray(persist_surface_row["pred_test_surface"], dtype=float)
        diff_surface = np.abs(pred_surface - expected_surface)
        max_surface_diff = float(np.max(diff_surface))
        report["persistence_surface_check"] = {"max_abs_diff": max_surface_diff}
        if max_surface_diff > 1e-12:
            idx = np.argwhere(diff_surface > 1e-12)[:5]
            mismatches = [
                {
                    "sample": int(i),
                    "h": int(h),
                    "surface_idx": int(k),
                    "pred": float(pred_surface[i, h, k]),
                    "explicit": float(expected_surface[i, h, k]),
                    "abs_diff": float(diff_surface[i, h, k]),
                }
                for i, h, k in idx
            ]
            report["persistence_surface_check"]["mismatches"] = mismatches
            raise RuntimeError(
                "Surface persistence implementation mismatch detected (max_abs_diff > 1e-12). "
                f"First mismatches: {mismatches}"
            )
        print(f"[sanity] persistence_surface_check passed (max_abs_diff={max_surface_diff:.3e}).")
    else:
        msg = "persistence_surface_naive row not found; skipped strict surface persistence equality check."
        report["warnings"].append(msg)
        print(f"[sanity] warning: {msg}")

    persist_row = next((r for r in eval_rows if str(r.get("model", "")).strip().lower() == "persistence_naive"), None)
    if persist_row is not None:
        explicit = np.repeat(x_eval_norm[:, -1:, : bundle.factor_dim], horizon, axis=1)
        pred = np.asarray(persist_row["pred_test_norm"], dtype=float)
        diff = np.abs(pred - explicit)
        max_diff = float(np.max(diff))
        report["persistence_check"] = {"max_abs_diff": max_diff}
        if max_diff > 1e-12:
            idx = np.argwhere(diff > 1e-12)[:5]
            mismatches = [
                {
                    "sample": int(i),
                    "h": int(h),
                    "factor": int(k),
                    "pred": float(pred[i, h, k]),
                    "explicit": float(explicit[i, h, k]),
                    "abs_diff": float(diff[i, h, k]),
                }
                for i, h, k in idx
            ]
            report["persistence_check"]["mismatches"] = mismatches
            raise RuntimeError(
                "Persistence implementation mismatch detected (max_abs_diff > 1e-12). "
                f"First mismatches: {mismatches}"
            )
        print(f"[sanity] persistence_check passed (max_abs_diff={max_diff:.3e}).")

    # C) Delta diagnostic for early/mid/final horizons, plus full-horizon summary.
    origin_surface = _surface_origin_matrix(bundle, lookback=int(lookback), split_slice=split_slice)
    rows = []
    for h in range(1, horizon + 1):
        target_h = np.asarray(y_true_test_surface[:, h - 1, :], dtype=float)
        delta_abs = np.abs(target_h - origin_surface)
        delta_rel = delta_abs / (np.abs(origin_surface) + float(max(1e-12, eps)))
        rows.append(
            {
                "horizon": int(h),
                "abs_delta_median": float(np.median(delta_abs)),
                "abs_delta_p90": float(np.percentile(delta_abs, 90.0)),
                "abs_delta_p99": float(np.percentile(delta_abs, 99.0)),
                "rel_delta_median": float(np.median(delta_rel)),
                "rel_delta_p90": float(np.percentile(delta_rel, 90.0)),
                "rel_delta_p99": float(np.percentile(delta_rel, 99.0)),
            }
        )
    delta_df = pd.DataFrame(rows)
    delta_csv = out_dir / f"level{int(level)}_delta_diagnostic.csv"
    delta_df.to_csv(delta_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = delta_df["horizon"].to_numpy(dtype=int)
    axes[0].plot(x, delta_df["abs_delta_median"], label="median", lw=2.0)
    axes[0].plot(x, delta_df["abs_delta_p90"], label="p90", lw=1.8)
    axes[0].plot(x, delta_df["abs_delta_p99"], label="p99", lw=1.6)
    axes[0].set_title("Absolute Delta vs Horizon")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("|y[t+h]-y[t]|")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(x, delta_df["rel_delta_median"], label="median", lw=2.0)
    axes[1].plot(x, delta_df["rel_delta_p90"], label="p90", lw=1.8)
    axes[1].plot(x, delta_df["rel_delta_p99"], label="p99", lw=1.6)
    axes[1].set_title("Relative Delta vs Horizon")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("|Δ| / (|y|+eps)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    delta_plot = out_dir / f"level{int(level)}_delta_diagnostic.png"
    fig.savefig(delta_plot, dpi=180)
    plt.close(fig)
    report["delta_diagnostic"] = {
        "csv": str(delta_csv),
        "plot": str(delta_plot),
        "selected_h_stats": delta_df[delta_df["horizon"].isin(horizon_choices)].to_dict(orient="records"),
    }
    print(f"[sanity] delta_diagnostic wrote {delta_csv.name} and {delta_plot.name}.")

    # D) MAPE pathology check.
    y_abs = np.abs(np.asarray(y_true_test_surface, dtype=float))
    frac_small = float(np.mean(y_abs < float(max(1e-12, near_zero_threshold))))
    report["mape_pathology"] = {
        "near_zero_threshold": float(near_zero_threshold),
        "fraction_abs_y_below_threshold": frac_small,
    }
    if frac_small > 0.01:
        msg = (
            "MAPE instability risk: more than 1% of targets are near zero. "
            "Prefer sMAPE/WAPE/MAE/RMSE for robust comparisons."
        )
        report["warnings"].append(msg)
        print(f"[sanity] warning: {msg} frac_small={frac_small:.4%}")
    else:
        print(f"[sanity] mape_pathology fraction_small={frac_small:.4%}.")

    report_path = out_dir / f"level{int(level)}_sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sanity] wrote report: {report_path}")
    return report
