from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any
import warnings

# Example commands:
# 1) Quick photonic MC-first sweep (20 modes):
#    python -m submission_swaptions.plot_model_comparison --level 1 --fast --pqrc_sweep \
#      --qrc_sweep_objective surface_mape --pqrc_modes 20 --qrc_mode auto --qrc_target delta \
#      --qrc_baseline persistence --target_transform log --out_dir results/qrc_mc_sweep_l1
# 2) Full evaluation vs all model families:
#    python -m submission_swaptions.plot_model_comparison --level 1 --run_quantum --run_ported \
#      --forecast_horizons 6 --qrc_mode auto --qrc_target delta --qrc_baseline persistence \
#      --pqrc_modes 20 --out_dir results/all_models_l1


def maybe_reexec_with_submission_env() -> None:
    if os.environ.get("SUBMISSION_REEXEC") == "1":
        return

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    preferred = repo_root / ".venv310_arm" / "bin" / "python"

    if preferred.exists() and Path(sys.executable).resolve() != preferred.resolve():
        env = dict(os.environ)
        env["SUBMISSION_REEXEC"] = "1"
        env.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("PYTHONHASHSEED", "0")
        os.execve(str(preferred), [str(preferred), "-m", "submission_swaptions.compare_ported_qml", *sys.argv[1:]], env)


maybe_reexec_with_submission_env()

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constraints import validate_constraints
from .models_ported_hybridai import (
    PORTED_ALGO_SOURCES,
    run_ctrl_classical_reservoir,
    run_ctrl_learned_featuremap,
    run_ctrl_rff,
    run_q_bottleneck,
    run_qrc,
)
from .pipeline import _level_cfg, _load_submission_config, load_and_preprocess


def _to_surface_space(bundle: Any, z_pred_norm: np.ndarray) -> np.ndarray:
    z_pred = bundle.normalizer.inverse_transform_seq(z_pred_norm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        surf_scaled = bundle.pca.inverse_transform_seq(z_pred)
    surf = np.clip(surf_scaled * bundle.price_scale, 0.0, bundle.clip_cap)
    return np.nan_to_num(surf, nan=0.0, posinf=bundle.clip_cap, neginf=0.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse


def _plot_metric_bars(metrics_df: pd.DataFrame, out_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(x - width / 2, metrics_df["factor_mae"], width, label="MAE")
    axes[0].bar(x + width / 2, metrics_df["factor_rmse"], width, label="RMSE")
    axes[0].set_title("Factor-Space Error")
    axes[0].set_ylabel("Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2, metrics_df["surface_mae"], width, label="MAE")
    axes[1].bar(x + width / 2, metrics_df["surface_rmse"], width, label="RMSE")
    axes[1].set_title("Surface-Space Error")
    axes[1].set_ylabel("Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend()

    fig.suptitle("Ported HybridAIQuantum Algorithms on Swaptions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _missingness_vec(missing_seq_norm: np.ndarray) -> np.ndarray:
    last = missing_seq_norm[:, -1, :]
    mean = missing_seq_norm.mean(axis=1)
    std = missing_seq_norm.std(axis=1)
    return np.concatenate([last, mean, std], axis=1)


def run_ported_comparison(
    data_dir: str,
    level: int,
    backend: str,
    seed: int,
    out_dir: str,
    include_quantum: bool,
    fast: bool,
) -> tuple[pd.DataFrame, Path]:
    cfg = _load_submission_config(Path(__file__).with_name("config.yaml"))
    lcfg = _level_cfg(cfg, int(level))
    qcfg = lcfg.get("quantum", {}) if isinstance(lcfg.get("quantum", {}), dict) else {}

    L = int(lcfg.get("L", 20))
    H = int(lcfg.get("H", 10))
    imputer = str(lcfg.get("imputer", "ffill_interp" if int(level) == 1 else "svd_iterative"))
    d_factors = int(lcfg.get("d_factors", 8))
    d_q = int(qcfg.get("d_q", 8))
    d_phi = int(qcfg.get("d_phi", qcfg.get("qrc_feature_dim", 24)))
    shots = int(max(1, int(qcfg.get("shots", 32))))
    qrc_stride = int(qcfg.get("qrc_stride", 1))
    train_qeval_budget = int(qcfg.get("train_qeval_budget", 5000))
    batch_size = int(qcfg.get("batch_size", 32))

    if fast or os.environ.get("SUBMISSION_SMOKE_FAST", "0") == "1":
        imputer = "ffill_interp"
        d_factors = min(d_factors, 6)
        d_q = min(d_q, 6)
        d_phi = min(d_phi, 16)
        shots = max(1, min(shots, 16))
        qrc_stride = max(qrc_stride, 2)
        train_qeval_budget = min(train_qeval_budget, 1000)
        batch_size = max(batch_size, 64)

    bundle = load_and_preprocess(
        data_dir=data_dir,
        level=int(level),
        L=L,
        H=H,
        imputer=imputer,
        d_factors=d_factors,
        seed=int(seed),
    )

    y_true_test_norm = bundle.y_test_norm
    y_true_surface = np.clip(bundle.y_test_surface_scaled * bundle.price_scale, 0.0, bundle.clip_cap)
    x_train_factor = bundle.x_train_norm[:, :, : bundle.factor_dim]
    x_val_factor = bundle.x_val_norm[:, :, : bundle.factor_dim]
    x_test_factor = bundle.x_test_norm[:, :, : bundle.factor_dim]

    m_train = _missingness_vec(bundle.missing_train_norm)
    m_val = _missingness_vec(bundle.missing_val_norm)
    m_test = _missingness_vec(bundle.missing_test_norm)

    eval_rows: list[dict[str, Any]] = []

    # Ported classical controls.
    ctrl_rff = run_ctrl_rff(
        x_train=x_train_factor,
        y_train=bundle.y_train_norm,
        x_val=x_val_factor,
        y_val=bundle.y_val_norm,
        x_test=x_test_factor,
        horizon=H,
        seed=int(seed),
        d_q=d_q,
        d_phi=d_phi,
        use_mask_features=True,
        mask_train_vec=m_train,
        mask_val_vec=m_val,
        mask_test_vec=m_test,
    )
    ctrl_lfm = run_ctrl_learned_featuremap(
        x_train=x_train_factor,
        y_train=bundle.y_train_norm,
        x_val=x_val_factor,
        y_val=bundle.y_val_norm,
        x_test=x_test_factor,
        horizon=H,
        seed=int(seed),
        d_q=d_q,
        d_phi=d_phi,
        use_mask_features=True,
        mask_train_vec=m_train,
        mask_val_vec=m_val,
        mask_test_vec=m_test,
        max_steps=120 if fast else 200,
        batch_size=batch_size,
    )
    ctrl_res = run_ctrl_classical_reservoir(
        x_train=x_train_factor,
        y_train=bundle.y_train_norm,
        x_val=x_val_factor,
        y_val=bundle.y_val_norm,
        x_test=x_test_factor,
        horizon=H,
        seed=int(seed),
        d_phi=d_phi,
        qrc_stride=qrc_stride,
        qrc_pool="summary" if fast else "none",
        reservoir_alpha=0.7,
        reservoir_rho=0.9,
        reservoir_ridge_lambda=1.0,
    )

    forecasts = [ctrl_rff, ctrl_lfm, ctrl_res]

    # Ported quantum models.
    if include_quantum:
        try:
            validate_constraints(input_dim=int(x_train_factor.shape[-1]), backend=str(backend))
            q_b = run_q_bottleneck(
                x_train=x_train_factor,
                y_train=bundle.y_train_norm,
                x_val=x_val_factor,
                y_val=bundle.y_val_norm,
                x_test=x_test_factor,
                horizon=H,
                shots=shots,
                train_qeval_budget=train_qeval_budget,
                batch_size=batch_size,
                seed=int(seed),
                d_q=d_q,
                d_phi=d_phi,
                train_quantum=False,
                grad_qeval_multiplier=1.0,
                use_mask_features=True,
                mask_train_vec=m_train,
                mask_val_vec=m_val,
                mask_test_vec=m_test,
            )
            forecasts.append(q_b)
        except Exception as exc:
            print(f"[ported_qml] skipping q_bottleneck: {exc}")

        try:
            validate_constraints(input_dim=int(x_train_factor.shape[-1]), backend=str(backend))
            q_r = run_qrc(
                x_train=x_train_factor,
                y_train=bundle.y_train_norm,
                x_val=x_val_factor,
                y_val=bundle.y_val_norm,
                x_test=x_test_factor,
                horizon=H,
                shots=shots,
                train_qeval_budget=train_qeval_budget,
                batch_size=batch_size,
                seed=int(seed),
                d_phi=d_phi,
                train_quantum=False,
                grad_qeval_multiplier=1.0,
                qrc_stride=qrc_stride,
                qrc_pool="summary" if fast else "none",
                reservoir_alpha=0.7,
                reservoir_rho=0.9,
                reservoir_ridge_lambda=1.0,
            )
            forecasts.append(q_r)
        except Exception as exc:
            print(f"[ported_qml] skipping qrc: {exc}")

    for fc in forecasts:
        y_pred_test_norm = fc.z_pred_test
        f_mae, f_rmse = _compute_metrics(y_true_test_norm, y_pred_test_norm)
        y_pred_surface = _to_surface_space(bundle, y_pred_test_norm)
        s_mae, s_rmse = _compute_metrics(y_true_surface, y_pred_surface)

        row = {
            "model": str(fc.model),
            "copied_from_hybridaiquantum": True,
            "source_repo": "https://github.com/Quandela/HybridAIQuantum-Challenge",
            "source_path": PORTED_ALGO_SOURCES.get(str(fc.model), "src/qml/models.py"),
            "best_blend_w": float(fc.best_blend_w),
            "factor_mae": float(f_mae),
            "factor_rmse": float(f_rmse),
            "surface_mae": float(s_mae),
            "surface_rmse": float(s_rmse),
        }
        row.update(fc.budget.as_dict())
        row.update(fc.extra)
        eval_rows.append(row)

    metrics_df = pd.DataFrame(eval_rows).sort_values(["surface_mae", "factor_mae"], ascending=True).reset_index(drop=True)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir_p / f"level{int(level)}_ported_hybridai_metrics.csv"
    plot_path = out_dir_p / f"level{int(level)}_ported_hybridai_metrics.png"
    src_path = out_dir_p / f"level{int(level)}_ported_hybridai_sources.csv"

    metrics_df.to_csv(csv_path, index=False)
    _plot_metric_bars(metrics_df, plot_path)

    src_df = pd.DataFrame(
        [{"model": k, "source_path": v, "source_repo": "https://github.com/Quandela/HybridAIQuantum-Challenge"} for k, v in PORTED_ALGO_SOURCES.items()]
    )
    src_df.to_csv(src_path, index=False)

    return metrics_df, out_dir_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ported HybridAIQuantum algorithms on the Swaptions pipeline.")
    parser.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--backend", type=str, choices=["sim", "qpu"], default="sim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--skip_quantum", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.strip() if args.out_dir else ""
    if not out_dir:
        out_dir = str(Path("results") / f"ported_hybridai_level{int(args.level)}")

    metrics_df, out_dir_p = run_ported_comparison(
        data_dir=args.data_dir,
        level=int(args.level),
        backend=str(args.backend),
        seed=int(args.seed),
        out_dir=out_dir,
        include_quantum=not bool(args.skip_quantum),
        fast=bool(args.fast),
    )

    print("Saved ported comparison outputs:")
    print(out_dir_p / f"level{int(args.level)}_ported_hybridai_metrics.csv")
    print(out_dir_p / f"level{int(args.level)}_ported_hybridai_metrics.png")
    print(out_dir_p / f"level{int(args.level)}_ported_hybridai_sources.csv")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
