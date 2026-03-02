from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
        os.execve(str(preferred), [str(preferred), "-m", "submission_swaptions.solution", *sys.argv[1:]], env)


maybe_reexec_with_submission_env()

from .pipeline import run_end_to_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Swaptions Q-ANNOT end-to-end pipeline.")
    parser.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    parser.add_argument("--level", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--backend", type=str, choices=["sim", "qpu"], default="sim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=0, help="Lookback window in observed samples (0 uses config).")
    parser.add_argument("--use_cycle_phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--classical_kind",
        type=str,
        choices=[
            "mlp",
            "gru",
            "lstm",
            "factor_ar",
            "reservoir",
            "persistence",
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "persist_qrc_weak",
            "photonic_memory",
            "photonic_memory_no_feedback",
        ],
        default="",
        help="Override classical forecaster from config.yaml.",
    )
    parser.add_argument("--pqrc_M", type=int, default=8)
    parser.add_argument("--pqrc_modes", type=int, default=0, help="Alias for photonic mode count (0 uses --pqrc_M).")
    parser.add_argument("--pqrc_Nph", type=int, default=2)
    parser.add_argument("--pqrc_nphotons", type=int, default=None, help="Alias for --pqrc_Nph.")
    parser.add_argument("--pqrc_budget", type=int, default=32)
    parser.add_argument("--pqrc_gain", type=float, default=0.7565800181198181)
    parser.add_argument("--pqrc_feature", type=str, choices=["coincidence", "clickprob"], default="clickprob")
    parser.add_argument("--pqrc_higher_order", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--pqrc_budget_alloc", type=str, choices=["uniform", "weighted"], default="uniform")
    parser.add_argument("--pqrc_shots", type=int, default=32)
    parser.add_argument("--pqrc_ridge", type=float, default=0.0051830988569115735)
    parser.add_argument("--pqrc_in_pca", type=int, default=4)
    parser.add_argument(
        "--pqrc_factor_cap",
        type=int,
        default=0,
        help="Photonic input factor cap (<=0 disables cap and uses all available factors).",
    )
    parser.add_argument("--qrc_mode", type=str, choices=["residual", "gated_residual", "absolute", "auto"], default="residual")
    parser.add_argument("--qrc_target", type=str, choices=["level", "delta", "log_return", "norm_resid"], default="delta")
    parser.add_argument("--qrc_baseline", type=str, choices=["persistence", "seasonal13"], default="persistence")
    parser.add_argument("--target_transform", type=str, choices=["none", "log"], default="log")
    parser.add_argument("--y_floor_mode", type=str, choices=["train_p01", "train_p001", "fixed"], default="train_p001")
    parser.add_argument("--y_floor_value", type=float, default=None)
    parser.add_argument("--persist_qrc_alpha", type=float, default=0.25)
    parser.add_argument("--persist_qrc_rclip", type=float, default=0.05)
    parser.add_argument("--persist_qrc_drive_scale", type=float, default=0.25)
    parser.add_argument("--qrc_gate_tau", type=float, default=0.06184302083987248)
    parser.add_argument("--tau", type=float, default=None, help="Legacy alias for --qrc_gate_tau.")
    parser.add_argument("--qrc_resid_clip", type=float, default=None)
    parser.add_argument("--qrc_residvar_penalty", type=float, default=0.1)
    parser.add_argument("--qrc_feat_norm", type=str, choices=["none", "standard"], default="standard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv.strip() if isinstance(args.out_csv, str) else ""
    if not out_csv:
        out_csv = str(Path("results") / f"swaptions_submission_level{int(args.level)}.csv")

    _, payload = run_end_to_end(
        data_dir=args.data_dir,
        level=int(args.level),
        out_csv=out_csv,
        backend=str(args.backend),
        seed=int(args.seed),
        use_cycle_phase=bool(args.use_cycle_phase),
        classical_kind=(str(args.classical_kind).strip().lower() or None),
        lookback=(int(args.lookback) if int(args.lookback) > 0 else None),
        pqrc_M=int(args.pqrc_M),
        pqrc_modes=(int(args.pqrc_modes) if int(args.pqrc_modes) > 0 else None),
        pqrc_Nph=int(args.pqrc_Nph),
        pqrc_nphotons=(None if args.pqrc_nphotons is None else int(args.pqrc_nphotons)),
        pqrc_budget=int(args.pqrc_budget),
        pqrc_gain=float(args.pqrc_gain),
        pqrc_feature=str(args.pqrc_feature),
        pqrc_higher_order=int(args.pqrc_higher_order),
        pqrc_budget_alloc=str(args.pqrc_budget_alloc),
        pqrc_shots=int(args.pqrc_shots),
        pqrc_ridge=float(args.pqrc_ridge),
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
    )
    metrics = payload.get("metrics", {})
    print("Run complete.")
    if isinstance(metrics, dict):
        show = [
            "classical_kind",
            "quantum_mode",
            "selected_dim",
            "w_star",
            "test_factor_mae_blend",
            "test_surface_mae_blend",
            "total_qevals",
            "total_shots",
        ]
        compact = {k: metrics[k] for k in show if k in metrics}
        print(compact)


if __name__ == "__main__":
    main()
