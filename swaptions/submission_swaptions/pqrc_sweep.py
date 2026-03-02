from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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
        os.execve(str(preferred), [str(preferred), "-m", "submission_swaptions.pqrc_sweep", *sys.argv[1:]], env)


maybe_reexec_with_submission_env()

from .plot_model_comparison import run_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Photonic QRC hyperparameter sweep runner (surface-objective selection).")
    p.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    p.add_argument("--level", type=int, choices=[1, 2], default=1)
    p.add_argument("--backend", type=str, choices=["sim", "qpu"], default="sim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--lookback", type=int, default=26)
    p.add_argument("--forecast_horizons", type=int, default=6)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--pqrc_M", type=int, default=12)
    p.add_argument("--pqrc_modes", type=int, default=0, help="Alias for photonic modes (0 uses --pqrc_M).")
    p.add_argument("--pqrc_Nph", type=int, default=2)
    p.add_argument("--pqrc_nphotons", type=int, default=None, help="Alias for --pqrc_Nph.")
    p.add_argument("--pqrc_budget", type=int, default=32)
    p.add_argument("--pqrc_gain", type=float, default=None)
    p.add_argument("--pqrc_input_scale", type=float, default=None)
    p.add_argument("--pqrc_feature", type=str, choices=["coincidence", "clickprob", "pseudo_coincidence"], default="coincidence")
    p.add_argument("--pqrc_pseudocount", type=float, default=0.5, help="Additive pseudocount smoothing for shot-derived feature histograms (0 disables).")
    p.add_argument("--pqrc_higher_order", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--pqrc_budget_alloc", type=str, choices=["uniform", "weighted"], default="uniform")
    p.add_argument("--pqrc_shots", type=int, default=128)
    p.add_argument("--pqrc_ridge", type=float, default=100.0)
    p.add_argument("--pqrc_in_pca", type=int, default=4)
    p.add_argument(
        "--pqrc_factor_cap",
        type=int,
        default=0,
        help="Photonic input factor cap (<=0 disables cap and uses all available factors).",
    )
    p.add_argument("--qrc_mode", type=str, choices=["residual", "gated_residual", "absolute", "auto"], default="auto")
    p.add_argument("--qrc_target", type=str, choices=["level", "delta", "log_return", "norm_resid"], default="delta")
    p.add_argument("--qrc_baseline", type=str, choices=["persistence", "seasonal13"], default="persistence")
    p.add_argument("--target_transform", type=str, choices=["none", "log"], default="log")
    p.add_argument("--y_floor_mode", type=str, choices=["train_p01", "train_p001", "fixed"], default="train_p001")
    p.add_argument("--y_floor_value", type=float, default=None)
    p.add_argument("--persist_qrc_alpha", type=float, default=0.25)
    p.add_argument("--persist_qrc_rclip", type=float, default=0.05)
    p.add_argument("--persist_qrc_drive_scale", type=float, default=0.25)
    p.add_argument("--qrc_gate_tau", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=None, help="Legacy alias for --qrc_gate_tau.")
    p.add_argument("--qrc_resid_clip", type=float, default=2.5)
    p.add_argument("--qrc_residvar_penalty", type=float, default=0.1)
    p.add_argument("--qrc_feat_norm", type=str, choices=["none", "standard"], default="standard")
    p.add_argument("--qrc_residual_penalty", type=float, default=0.1)
    p.add_argument(
        "--qrc_sweep_objective",
        type=str,
        choices=["surface_mape", "surface_wape", "surface_mae", "surface_rmse", "mape", "mc_then_mape"],
        default="surface_mape",
    )
    p.add_argument(
        "--sweep_objective",
        type=str,
        choices=["surface_mape", "surface_wape", "surface_mae", "surface_rmse", "mape", "mc_then_mape"],
        default="",
        help="Alias for --qrc_sweep_objective (surface-based objective for sweep selection).",
    )
    p.add_argument(
        "--performance_preset",
        action="store_true",
        help="Use higher-capacity photonic defaults when knobs are not explicitly set (modes=20, nphotons=10).",
    )
    p.add_argument("--sweep_modes", action="store_true", help="Allow modes grid search (otherwise locked if provided).")
    p.add_argument("--sweep_gain", action="store_true", help="Allow gain grid search (otherwise locked if provided).")
    p.add_argument("--sweep_input_scale", action="store_true", help="Allow input_scale grid search (otherwise locked if provided).")
    p.add_argument("--sweep_ridge", action="store_true", help="Allow ridge grid search (otherwise locked if provided).")
    p.add_argument("--pqrc_sweep_subset", type=int, default=256)
    p.add_argument("--pqrc_sweep_topk", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.strip() if args.out_dir else ""
    if not out_dir:
        out_dir = str(Path("results") / f"pqrc_sweep_level{int(args.level)}")

    sweep_modes = bool(args.sweep_modes) or (int(args.pqrc_modes) <= 0)
    sweep_gain = bool(args.sweep_gain) or (args.pqrc_gain is None)
    sweep_input_scale = bool(args.sweep_input_scale) or (args.pqrc_input_scale is None)
    sweep_ridge = bool(args.sweep_ridge) or (args.pqrc_ridge is None)
    sweep_objective = str(args.sweep_objective).strip() or str(args.qrc_sweep_objective)
    pqrc_nphotons = None if args.pqrc_nphotons is None else int(args.pqrc_nphotons)

    run_comparison(
        data_dir=str(args.data_dir),
        level=int(args.level),
        backend=str(args.backend),
        seed=int(args.seed),
        out_dir=out_dir,
        include_quantum=False,
        include_ported=False,
        include_classical=True,
        classical_models=["persistence_surface_naive", "persistence_naive", "photonic_qrc_feedback", "photonic_qrc_no_feedback"],
        fast=bool(args.fast),
        lookback=int(args.lookback) if int(args.lookback) > 0 else None,
        forecast_horizons=int(args.forecast_horizons),
        pqrc_M=int(args.pqrc_M),
        pqrc_modes=(int(args.pqrc_modes) if int(args.pqrc_modes) > 0 else None),
        pqrc_Nph=int(args.pqrc_Nph),
        pqrc_nphotons=pqrc_nphotons,
        pqrc_budget=int(args.pqrc_budget),
        pqrc_gain=(None if args.pqrc_gain is None else float(args.pqrc_gain)),
        pqrc_input_scale=(None if args.pqrc_input_scale is None else float(args.pqrc_input_scale)),
        pqrc_feature=str(args.pqrc_feature),
        pqrc_pseudocount=float(args.pqrc_pseudocount),
        pqrc_higher_order=int(args.pqrc_higher_order),
        pqrc_budget_alloc=str(args.pqrc_budget_alloc),
        pqrc_shots=int(args.pqrc_shots),
        pqrc_ridge=(None if args.pqrc_ridge is None else float(args.pqrc_ridge)),
        pqrc_in_pca=int(args.pqrc_in_pca),
        pqrc_factor_cap=int(args.pqrc_factor_cap),
        qrc_mode=str(args.qrc_mode),
        qrc_target=str(args.qrc_target),
        qrc_baseline=str(args.qrc_baseline),
        target_transform=str(args.target_transform),
        y_floor_mode=str(args.y_floor_mode),
        y_floor_value=(None if args.y_floor_value is None else float(args.y_floor_value)),
        persist_qrc_alpha=float(args.persist_qrc_alpha),
        persist_qrc_rclip=float(args.persist_qrc_rclip),
        persist_qrc_drive_scale=float(args.persist_qrc_drive_scale),
        tau=(None if args.tau is None else float(args.tau)),
        qrc_gate_tau=float(args.qrc_gate_tau),
        qrc_resid_clip=(None if args.qrc_resid_clip is None else float(args.qrc_resid_clip)),
        qrc_residvar_penalty=float(args.qrc_residvar_penalty),
        qrc_feat_norm=str(args.qrc_feat_norm),
        qrc_residual_penalty=float(args.qrc_residual_penalty),
        qrc_sweep_objective=str(sweep_objective),
        sweep_objective=str(sweep_objective),
        pqrc_sweep=True,
        sweep_modes=bool(sweep_modes),
        sweep_gain=bool(sweep_gain),
        sweep_input_scale=bool(sweep_input_scale),
        sweep_ridge=bool(sweep_ridge),
        pqrc_sweep_subset=int(args.pqrc_sweep_subset),
        pqrc_sweep_topk=int(args.pqrc_sweep_topk),
        performance_preset=bool(args.performance_preset),
    )
    print(f"Saved photonic sweep outputs under: {out_dir}")


if __name__ == "__main__":
    main()
