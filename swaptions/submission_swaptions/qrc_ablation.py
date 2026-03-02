from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


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
        os.execve(str(preferred), [str(preferred), "-m", "submission_swaptions.qrc_ablation", *sys.argv[1:]], env)


if __name__ == "__main__":
    maybe_reexec_with_submission_env()

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import _level_cfg, _load_submission_config, load_and_preprocess
from .plot_model_comparison import (
    PERSISTENCE_SURFACE_NAME,
    _compute_collapse_diagnostics,
    _dataset_signature,
    _horizon_metric,
    _sanitize_objective_name,
    _surface_persistence_prediction,
    _to_surface_space,
    _train_eval_photonic_config,
)
from .qrc_config import QRCConfig, config_to_hash, save_config


def _sample_feature(rng: np.random.Generator) -> str:
    # MAPE-first prior: keep coincidence most likely, while still exploring dense modes.
    options = ["coincidence", "coincidence", "coincidence", "pseudo_coincidence", "clickprob"]
    return str(options[int(rng.integers(0, len(options)))])


def _sample_shots(rng: np.random.Generator) -> int:
    options = [32, 64, 128]
    return int(options[int(rng.integers(0, len(options)))])


def _objective_key(name: str) -> str:
    obj = _sanitize_objective_name(name)
    if obj == "surface_mape":
        return "val_surface_mape"
    if obj == "surface_wape":
        return "val_surface_wape"
    if obj == "surface_mae":
        return "val_surface_mae"
    if obj == "surface_rmse":
        return "val_surface_rmse"
    return "val_surface_mape"


def _build_model_kwargs(
    cfg: QRCConfig,
    *,
    kind: str,
    seed: int,
    dataset_signature: str,
    qrc_config_hash: str,
    pqrc_factor_cap: int,
) -> dict[str, Any]:
    p = cfg.to_dict()
    kwargs = {
        "pqrc_M": int(p["pqrc_modes"]),
        "pqrc_modes": int(p["pqrc_modes"]),
        "pqrc_Nph": int(p["pqrc_nph"]),
        "pqrc_nph": int(p["pqrc_nph"]),
        "pqrc_nphotons": int(p["pqrc_nph"]),
        "pqrc_budget": int(p["pqrc_budget"]),
        "pqrc_gain": (0.0 if str(kind).strip().lower() == "photonic_qrc_no_feedback" else float(p["pqrc_gain"])),
        "pqrc_input_scale": float(p["pqrc_input_scale"]),
        "pqrc_feature": str(p["pqrc_feature"]),
        "pqrc_pseudocount": float(p["pqrc_pseudocount"]),
        "pqrc_higher_order": int(p["pqrc_higher_order"]),
        "pqrc_budget_alloc": str(p["pqrc_budget_alloc"]),
        "pqrc_shots": int(p["pqrc_shots"]),
        "pqrc_ridge": float(p["pqrc_ridge"]),
        "pqrc_in_pca": int(p["pqrc_in_pca"]),
        "pqrc_factor_cap": int(pqrc_factor_cap),
        "qrc_mode": str(p["qrc_mode"]),
        "qrc_target": str(p["qrc_target"]),
        "qrc_baseline": str(p["qrc_baseline"]),
        "target_transform": str(p["target_transform"]),
        "y_floor_mode": str(p["y_floor_mode"]),
        "y_floor_value": p["y_floor_value"],
        "tau": float(p["qrc_gate_tau"]),
        "qrc_gate_tau": float(p["qrc_gate_tau"]),
        "qrc_resid_clip": p["qrc_resid_clip"],
        "qrc_residvar_penalty": float(p["qrc_residvar_penalty"]),
        "qrc_feat_norm": str(p["qrc_feat_norm"]),
        "qrc_config_hash": str(qrc_config_hash),
        "dataset_signature": str(dataset_signature),
        "seed": int(seed),
        "level": int(p["level"]),
        "lookback": int(p["lookback"]),
        "horizons": int(p["horizons"]),
    }
    return kwargs


def _plot_objective_vs_trial(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    trial = pd.to_numeric(df["trial"], errors="coerce").to_numpy(dtype=float)
    objective = pd.to_numeric(df["objective_value"], errors="coerce").to_numpy(dtype=float)
    status = df["status"].astype(str).to_numpy()
    finite = np.isfinite(trial) & np.isfinite(objective)
    if np.any(finite):
        ax.plot(trial[finite], objective[finite], color="#999999", alpha=0.35, lw=1.2)
    for label, color in [("success", "#1f77b4"), ("rejected", "#ff7f0e"), ("failed", "#d62728")]:
        m = finite & (status == label)
        if np.any(m):
            ax.scatter(trial[m], objective[m], s=26, color=color, alpha=0.9, label=label)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective")
    ax.set_title("QRC Ablation Objective vs Trial")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_feature_collapse_vs_objective(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ok = df[df["status"].astype(str) == "success"].copy()
    if ok.empty:
        return
    x = pd.to_numeric(ok["feature_collapse_rate_train"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ok["objective_value"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(x[m], y[m], s=26, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Feature Collapse Rate (train)")
    ax.set_ylabel("Objective")
    ax.set_title("Feature Collapse vs Objective")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_residual_ratio_vs_objective(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ok = df[df["status"].astype(str) == "success"].copy()
    if ok.empty:
        return
    x = pd.to_numeric(ok["residual_std_ratio"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ok["objective_value"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(x[m], y[m], s=26, color="#9467bd", alpha=0.85)
    ax.axvline(1.0, color="#666666", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel("Residual Std Ratio")
    ax.set_ylabel("Objective")
    ax.set_title("Residual Dispersion vs Objective")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random ablation runner for photonic QRC (surface-objective selection).")
    p.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    p.add_argument("--level", type=int, choices=[1, 2], default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--forecast_horizons", type=int, default=6)
    p.add_argument("--objective", type=str, choices=["surface_mape", "surface_wape", "surface_mae", "surface_rmse"], default="surface_mape")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--kind", type=str, choices=["photonic_qrc_feedback", "photonic_qrc_no_feedback"], default="photonic_qrc_feedback")
    p.add_argument("--out_dir", type=str, default="results/qrc_sweep")
    p.add_argument("--mape_eps", type=float, default=1e-8)
    p.add_argument("--feature_std_eps", type=float, default=1e-8)
    p.add_argument("--collapse_std_eps", type=float, default=1e-10)
    p.add_argument("--sweep_shots", action="store_true", help="If set, sample shots from {32,64,128}; otherwise keep --pqrc_shots fixed.")
    p.add_argument("--pqrc_shots", type=int, default=32)
    p.add_argument("--pqrc_budget", type=int, default=32)
    p.add_argument("--pqrc_nphotons", type=int, default=2)
    p.add_argument("--pqrc_in_pca", type=int, default=4)
    p.add_argument("--pqrc_factor_cap", type=int, default=0)
    p.add_argument("--pqrc_higher_order", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--pqrc_budget_alloc", type=str, choices=["uniform", "weighted"], default="uniform")
    p.add_argument("--qrc_mode", type=str, choices=["residual", "gated_residual", "absolute", "auto"], default="auto")
    p.add_argument("--qrc_target", type=str, choices=["level", "delta", "log_return", "norm_resid"], default="delta")
    p.add_argument("--qrc_baseline", type=str, choices=["persistence", "seasonal13"], default="persistence")
    p.add_argument("--target_transform", type=str, choices=["none", "log"], default="log")
    p.add_argument("--y_floor_mode", type=str, choices=["train_p01", "train_p001", "fixed"], default="train_p001")
    p.add_argument("--y_floor_value", type=float, default=None)
    p.add_argument("--qrc_feat_norm", type=str, choices=["none", "standard"], default="standard")
    p.add_argument("--qrc_resid_clip", type=float, default=None)
    p.add_argument("--qrc_residvar_penalty", type=float, default=0.1)
    p.add_argument(
        "--vary_trial_seed",
        action="store_true",
        help="If set, vary model seed per trial. Default keeps seed fixed for reproducible MAPE-first selection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_all = _load_submission_config(Path(__file__).with_name("config.yaml"))
    lcfg = _level_cfg(cfg_all, int(args.level))
    lookback = int(args.lookback if int(args.lookback) > 0 else int(lcfg.get("L", 20)))
    horizon = int(max(1, int(args.forecast_horizons)))
    d_factors = int(lcfg.get("d_factors", 8))
    imputer = str(lcfg.get("imputer", "ffill_interp" if int(args.level) == 1 else "svd_iterative"))
    use_cycle_phase = bool(lcfg.get("use_cycle_phase", True))
    objective = _sanitize_objective_name(str(args.objective))
    objective_key = _objective_key(objective)

    bundle = load_and_preprocess(
        data_dir=str(args.data_dir),
        level=int(args.level),
        L=int(lookback),
        H=int(horizon),
        imputer=str(imputer),
        d_factors=int(d_factors),
        seed=int(args.seed),
        use_cycle_phase=bool(use_cycle_phase),
    )
    dataset_signature = _dataset_signature(bundle, level=int(args.level), lookback=int(lookback), horizon=int(horizon))

    x_train_in = np.concatenate([bundle.x_train_norm, bundle.missing_train_norm], axis=2)
    x_val_in = np.concatenate([bundle.x_val_norm, bundle.missing_val_norm], axis=2)
    y_train = np.asarray(bundle.y_train_norm, dtype=float)
    y_val = np.asarray(bundle.y_val_norm, dtype=float)
    floor = float(getattr(bundle, "surface_floor", 0.0))
    y_true_val_surface = np.clip(bundle.y_val_surface_scaled * bundle.price_scale, floor, bundle.clip_cap)

    persistence_surface = _surface_persistence_prediction(bundle, bundle.ds.val_slice, int(horizon))
    persistence_row = {
        "model": PERSISTENCE_SURFACE_NAME,
        "pred_test_surface": np.asarray(persistence_surface, dtype=float),
        "pred_test_norm": np.zeros_like(y_val, dtype=float),
        "horizon_surface_mape": _horizon_metric(y_true_val_surface, persistence_surface, metric="mape", eps=float(args.mape_eps)),
        "model_hash": "persistence",
        "qrc_config_hash": "",
        "dataset_signature": str(dataset_signature),
    }

    base_cfg = QRCConfig.from_dict(
        {
            "qrc_mode": str(args.qrc_mode),
            "qrc_target": str(args.qrc_target),
            "qrc_baseline": str(args.qrc_baseline),
            "target_transform": str(args.target_transform),
            "y_floor_mode": str(args.y_floor_mode),
            "y_floor_value": (None if args.y_floor_value is None else float(args.y_floor_value)),
            "qrc_feat_norm": str(args.qrc_feat_norm),
            "qrc_gate_tau": 0.05,
            "qrc_resid_clip": (None if args.qrc_resid_clip is None else float(args.qrc_resid_clip)),
            "qrc_residvar_penalty": float(args.qrc_residvar_penalty),
            "qrc_sweep_objective": str(objective),
            "pqrc_modes": 8,
            "pqrc_gain": 0.8,
            "pqrc_input_scale": 0.75,
            "pqrc_ridge": 1e-2,
            "pqrc_shots": int(args.pqrc_shots),
            "pqrc_budget": int(args.pqrc_budget),
            "pqrc_nph": int(args.pqrc_nphotons),
            "pqrc_in_pca": int(args.pqrc_in_pca),
            "pqrc_feature": "coincidence",
            "pqrc_pseudocount": 0.5,
            "pqrc_higher_order": int(args.pqrc_higher_order),
            "pqrc_budget_alloc": str(args.pqrc_budget_alloc),
            "seed": int(args.seed),
            "level": int(args.level),
            "lookback": int(lookback),
            "horizons": int(horizon),
        }
    ).validate()

    search_space = {
        "pqrc_feature": ["coincidence", "pseudo_coincidence", "clickprob"],
        "pqrc_pseudocount": [0.0, 0.25, 0.5, 1.0],
        "pqrc_ridge_log10": [-4.0, -1.0],
        "qrc_gate_tau": [0.02, 0.15],
        "pqrc_gain": [0.4, 1.0],
        "pqrc_input_scale": [0.5, 1.0],
        "pqrc_modes": [8, 12, 16, 20],
        "pqrc_shots": ([32, 64, 128] if bool(args.sweep_shots) else [int(args.pqrc_shots)]),
    }

    rng = np.random.default_rng(int(args.seed))
    anchor_cfgs: list[dict[str, Any]] = []
    # Anchor 0: the exact baseline config so sweep never regresses below default without evidence.
    anchor_cfgs.append(dict(base_cfg.to_dict()))
    # Anchor 1: prior strong setting used in earlier all-model runs.
    anchor_legacy = dict(base_cfg.to_dict())
    anchor_legacy.update(
        {
            "pqrc_feature": "coincidence",
            "pqrc_gain": 0.8,
            "pqrc_input_scale": 0.75,
            "pqrc_ridge": 1e-2,
            "qrc_gate_tau": 0.05,
            "pqrc_modes": 8,
            "qrc_mode": "auto",
        }
    )
    anchor_cfgs.append(anchor_legacy)
    rows: list[dict[str, Any]] = []
    for trial in range(int(args.trials)):
        trial_seed = int(args.seed + 7919 * (trial + 1)) if bool(args.vary_trial_seed) else int(args.seed)
        t0 = time.perf_counter()
        if trial < len(anchor_cfgs):
            cfg_trial = dict(anchor_cfgs[trial])
        else:
            cfg_trial = dict(base_cfg.to_dict())
            cfg_trial["pqrc_feature"] = _sample_feature(rng)
            cfg_trial["pqrc_pseudocount"] = float(rng.choice(np.asarray([0.0, 0.25, 0.5, 1.0], dtype=float)))
            cfg_trial["pqrc_ridge"] = float(10.0 ** rng.uniform(-4.0, -1.0))
            cfg_trial["qrc_gate_tau"] = float(rng.uniform(0.02, 0.15))
            cfg_trial["pqrc_gain"] = float(rng.uniform(0.4, 1.0))
            cfg_trial["pqrc_input_scale"] = float(rng.uniform(0.5, 1.0))
            cfg_trial["pqrc_modes"] = int(rng.choice(np.asarray([8, 12, 16, 20], dtype=int)))
            cfg_trial["pqrc_shots"] = int(_sample_shots(rng) if bool(args.sweep_shots) else int(args.pqrc_shots))
        cfg_trial["seed"] = int(trial_seed)
        cfg_trial["qrc_sweep_objective"] = str(objective)
        if str(cfg_trial["pqrc_feature"]).strip().lower() == "clickprob":
            cfg_trial["pqrc_higher_order"] = 0
        if str(args.kind).strip().lower() == "photonic_qrc_no_feedback":
            cfg_trial["pqrc_gain"] = 0.0
        try:
            qrc_cfg = QRCConfig.from_dict(cfg_trial).validate()
        except Exception as exc:
            rows.append(
                {
                    "trial": int(trial),
                    "status": "failed",
                    "error": f"config_validation: {exc}",
                    "runtime_sec": float(time.perf_counter() - t0),
                    "dataset_signature": str(dataset_signature),
                    "objective_name": str(objective),
                    "objective_value": float("inf"),
                }
            )
            continue

        qrc_hash = config_to_hash(qrc_cfg)
        model_kwargs = _build_model_kwargs(
            qrc_cfg,
            kind=str(args.kind),
            seed=int(trial_seed),
            dataset_signature=str(dataset_signature),
            qrc_config_hash=str(qrc_hash),
            pqrc_factor_cap=int(args.pqrc_factor_cap),
        )

        try:
            rec = _train_eval_photonic_config(
                kind=str(args.kind),
                seed=int(trial_seed),
                model_kwargs=model_kwargs,
                x_train=x_train_in,
                y_train=y_train,
                x_val=x_val_in,
                y_val=y_val,
                x_test=x_val_in,
                y_true_test_norm=y_val,
                bundle=bundle,
                y_true_val_surface=y_true_val_surface,
                y_true_test_surface=y_true_val_surface,
                mape_eps=float(args.mape_eps),
                residual_penalty_weight=float(args.qrc_residvar_penalty),
                sweep_objective=str(objective),
            )
            runtime_sec = float(time.perf_counter() - t0)
            model_obj = rec.get("model_obj", None)
            pred_val_norm = np.asarray(rec["y_pred_test"], dtype=float)
            pred_val_surface = _to_surface_space(bundle, pred_val_norm)
            nonfinite = int(np.size(pred_val_surface) - np.isfinite(pred_val_surface).sum())
            nonpositive = int(np.sum(pred_val_surface <= 0.0))

            fs = rec.get("feature_stats")
            feat_std_train = np.asarray(fs.get("feat_std_train", []), dtype=float) if isinstance(fs, dict) else np.array([], dtype=float)
            feat_std_val = np.asarray(fs.get("feat_std_val", []), dtype=float) if isinstance(fs, dict) else np.array([], dtype=float)
            collapse_rate_train = float(np.mean(feat_std_train < float(args.feature_std_eps))) if feat_std_train.size else 1.0
            collapse_rate_val = float(np.mean(feat_std_val < float(args.feature_std_eps))) if feat_std_val.size else 1.0
            feature_median_std = float(np.median(feat_std_train)) if feat_std_train.size else 0.0

            diag_row = {
                "model": str(args.kind),
                "pred_test_surface": pred_val_surface,
                "pred_test_norm": pred_val_norm,
                "qrc_gate": rec.get("qrc_gate_test"),
                "model_hash": str(rec.get("model_hash", "")),
                "qrc_config_hash": str(qrc_hash),
                "dataset_signature": str(dataset_signature),
                "horizon_surface_mape": _horizon_metric(y_true_val_surface, pred_val_surface, metric="mape", eps=float(args.mape_eps)),
            }
            diag = _compute_collapse_diagnostics(
                [persistence_row, diag_row],
                y_true_test_surface=y_true_val_surface,
                eps=float(args.mape_eps),
            )
            diag_m = sorted(
                [d for d in diag if str(d.get("model", "")) == str(args.kind)],
                key=lambda d: int(d.get("horizon", 0)),
            )
            corr_h = np.array([float(d.get("corr_h", np.nan)) for d in diag_m], dtype=float)
            std_ratio_h = np.array([float(d.get("residual_std_ratio", np.nan)) for d in diag_m], dtype=float)
            mape_delta_h = np.array([float(d.get("mape_vs_persistence_delta_h", np.nan)) for d in diag_m], dtype=float)

            reject_reason = ""
            if nonfinite > 0 or nonpositive > 0:
                reject_reason = f"invalid_surface_output(nonfinite={nonfinite},nonpositive={nonpositive})"
            elif (not np.isfinite(feature_median_std)) or (feature_median_std <= float(args.collapse_std_eps)):
                reject_reason = f"feature_collapse(feature_median_std={feature_median_std:.3e})"

            row = {
                "trial": int(trial),
                "status": ("rejected" if reject_reason else "success"),
                "error": reject_reason,
                "runtime_sec": float(runtime_sec),
                "dataset_signature": str(dataset_signature),
                "qrc_config_hash": str(qrc_hash),
                "model_hash": str(rec.get("model_hash", "")),
                "objective_name": str(rec.get("objective_name", objective)),
                "objective_value": float(rec.get("objective_value", np.inf)),
                "score": float(rec.get("score", np.inf)),
                "val_surface_mape": float(rec.get("val_surface_mape", np.nan)),
                "val_surface_wape": float(rec.get("val_surface_wape", np.nan)),
                "val_surface_mae": float(rec.get("val_surface_mae", np.nan)),
                "val_surface_rmse": float(rec.get("val_surface_rmse", np.nan)),
                "val_delta_persist": float(rec.get("val_delta_persist", np.nan)),
                "residual_std_ratio": float(rec.get("residual_std_ratio", np.nan)),
                "corr_h_mean": float(np.nanmean(corr_h)) if corr_h.size else np.nan,
                "residual_std_ratio_h_mean": float(np.nanmean(std_ratio_h)) if std_ratio_h.size else np.nan,
                "corr_h": json.dumps([float(v) for v in corr_h], separators=(",", ":")),
                "residual_std_ratio_h": json.dumps([float(v) for v in std_ratio_h], separators=(",", ":")),
                "mape_vs_persistence_delta_h": json.dumps([float(v) for v in mape_delta_h], separators=(",", ":")),
                "feature_dim": int(feat_std_train.size),
                "feature_median_std": float(feature_median_std),
                "feature_collapse_rate_train": float(collapse_rate_train),
                "feature_collapse_rate_val": float(collapse_rate_val),
                "surface_nonfinite_count": int(nonfinite),
                "surface_nonpositive_count": int(nonpositive),
                "train_qevals": int(getattr(model_obj, "train_qevals", 0)),
                "infer_qevals": int(getattr(model_obj, "infer_qevals", 0)),
                "train_total_shots": int(getattr(model_obj, "train_total_shots", 0)),
                "infer_total_shots": int(getattr(model_obj, "infer_total_shots", 0)),
                "total_qevals": int(getattr(model_obj, "train_qevals", 0) + getattr(model_obj, "infer_qevals", 0)),
                "total_shots": int(getattr(model_obj, "train_total_shots", 0) + getattr(model_obj, "infer_total_shots", 0)),
                "shots_per_eval": int(getattr(model_obj, "shots_per_eval", int(model_kwargs["pqrc_shots"]))),
                "qrc_mode_used": str(rec.get("qrc_mode_used", model_kwargs["qrc_mode"])),
                "qrc_target_used": str(rec.get("qrc_target_used", model_kwargs["qrc_target"])),
                "qrc_baseline_used": str(rec.get("qrc_baseline_used", model_kwargs["qrc_baseline"])),
                "pqrc_feature": str(model_kwargs["pqrc_feature"]),
                "pqrc_pseudocount": float(model_kwargs["pqrc_pseudocount"]),
                "pqrc_ridge": float(model_kwargs["pqrc_ridge"]),
                "qrc_gate_tau": float(model_kwargs["qrc_gate_tau"]),
                "pqrc_gain": float(model_kwargs["pqrc_gain"]),
                "pqrc_input_scale": float(model_kwargs["pqrc_input_scale"]),
                "pqrc_modes": int(model_kwargs["pqrc_modes"]),
                "pqrc_shots": int(model_kwargs["pqrc_shots"]),
                "pqrc_budget": int(model_kwargs["pqrc_budget"]),
                "pqrc_nph": int(model_kwargs["pqrc_Nph"]),
                "pqrc_in_pca": int(model_kwargs["pqrc_in_pca"]),
                "pqrc_factor_cap": int(model_kwargs["pqrc_factor_cap"]),
                "pqrc_higher_order": int(model_kwargs["pqrc_higher_order"]),
                "pqrc_budget_alloc": str(model_kwargs["pqrc_budget_alloc"]),
                "seed": int(trial_seed),
                "level": int(args.level),
                "lookback": int(lookback),
                "horizons": int(horizon),
                "kind": str(args.kind),
                "config": json.dumps(model_kwargs, sort_keys=True, default=str, separators=(",", ":")),
            }
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "trial": int(trial),
                    "status": "failed",
                    "error": str(exc),
                    "runtime_sec": float(time.perf_counter() - t0),
                    "dataset_signature": str(dataset_signature),
                    "qrc_config_hash": str(qrc_hash),
                    "objective_name": str(objective),
                    "objective_value": float("inf"),
                    "kind": str(args.kind),
                    "seed": int(trial_seed),
                    "level": int(args.level),
                    "lookback": int(lookback),
                    "horizons": int(horizon),
                }
            )

    df = pd.DataFrame(rows).sort_values("trial", ascending=True, ignore_index=True)
    trials_csv = out_dir / "trials.csv"
    results_csv = out_dir / "results.csv"
    df.to_csv(trials_csv, index=False)
    df.to_csv(results_csv, index=False)

    ok = df[df["status"].astype(str) == "success"].copy()
    if ok.empty:
        raise RuntimeError("qrc_ablation found no successful configurations. Check trials.csv for rejection/failure reasons.")
    ok = ok.sort_values(
        by=["objective_value", "val_surface_mape", "val_surface_mae", "score"],
        ascending=[True, True, True, True],
        ignore_index=True,
    )
    best = ok.iloc[0].to_dict()

    best_mode_used = str(best.get("qrc_mode_used", args.qrc_mode))
    best_cfg_dict = {
        # Keep auto when requested so replay re-runs mode selection on the same objective.
        "qrc_mode": (str(args.qrc_mode) if str(args.qrc_mode).strip().lower() == "auto" else best_mode_used),
        "qrc_target": str(best.get("qrc_target_used", args.qrc_target)),
        "qrc_baseline": str(best.get("qrc_baseline_used", args.qrc_baseline)),
        "target_transform": str(args.target_transform),
        "y_floor_mode": str(args.y_floor_mode),
        "y_floor_value": (None if args.y_floor_value is None else float(args.y_floor_value)),
        "qrc_feat_norm": str(args.qrc_feat_norm),
        "qrc_gate_tau": float(best["qrc_gate_tau"]),
        "qrc_resid_clip": (None if args.qrc_resid_clip is None else float(args.qrc_resid_clip)),
        "qrc_residvar_penalty": float(args.qrc_residvar_penalty),
        "qrc_sweep_objective": str(objective),
        "pqrc_modes": int(best["pqrc_modes"]),
        "pqrc_gain": float(best["pqrc_gain"]),
        "pqrc_input_scale": float(best["pqrc_input_scale"]),
        "pqrc_ridge": float(best["pqrc_ridge"]),
        "pqrc_shots": int(best["pqrc_shots"]),
        "pqrc_budget": int(best["pqrc_budget"]),
        "pqrc_nph": int(best["pqrc_nph"]),
        "pqrc_in_pca": int(best["pqrc_in_pca"]),
        "pqrc_feature": str(best["pqrc_feature"]),
        "pqrc_pseudocount": float(best["pqrc_pseudocount"]),
        "pqrc_higher_order": int(best["pqrc_higher_order"]),
        "pqrc_budget_alloc": str(best["pqrc_budget_alloc"]),
        # Replay with the exact winning trial seed unless caller explicitly varies seeds.
        "seed": int(best.get("seed", args.seed)),
        "level": int(args.level),
        "lookback": int(lookback),
        "horizons": int(horizon),
    }
    best_cfg = QRCConfig.from_dict(best_cfg_dict).validate()
    save_config(best_cfg, out_dir / "best_config.json")
    with (out_dir / "best_config_full.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **_build_model_kwargs(
                    best_cfg,
                    kind=str(args.kind),
                    seed=int(best.get("seed", args.seed)),
                    dataset_signature=str(dataset_signature),
                    qrc_config_hash=str(config_to_hash(best_cfg)),
                    pqrc_factor_cap=int(args.pqrc_factor_cap),
                ),
                "kind": str(args.kind),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    run_meta = {
        "seed": int(args.seed),
        "level": int(args.level),
        "lookback": int(lookback),
        "horizons": int(horizon),
        "objective": str(objective),
        "objective_key": str(objective_key),
        "trials": int(args.trials),
        "kind": str(args.kind),
        "dataset_signature": str(dataset_signature),
        "search_space": search_space,
        "base_config": base_cfg.to_dict(),
        "best_trial": int(best["trial"]),
        "best_objective": float(best["objective_value"]),
        "best_qrc_mode_used": best_mode_used,
        "best_seed": int(best.get("seed", args.seed)),
        "vary_trial_seed": bool(args.vary_trial_seed),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    _plot_objective_vs_trial(df, out_dir / "objective_vs_trial.png")
    _plot_feature_collapse_vs_objective(df, out_dir / "feature_collapse_vs_objective.png")
    _plot_residual_ratio_vs_objective(df, out_dir / "residual_std_ratio_vs_objective.png")

    print("Saved ablation artifacts:")
    print(trials_csv)
    print(results_csv)
    print(out_dir / "best_config.json")
    print(out_dir / "best_config_full.json")
    print(out_dir / "run_metadata.json")
    print(out_dir / "objective_vs_trial.png")
    print(out_dir / "feature_collapse_vs_objective.png")
    print(out_dir / "residual_std_ratio_vs_objective.png")
    print()
    cols = [
        "trial",
        "status",
        "objective_value",
        "val_surface_mape",
        "feature_collapse_rate_train",
        "residual_std_ratio",
        "pqrc_feature",
        "pqrc_pseudocount",
        "pqrc_modes",
        "pqrc_shots",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
