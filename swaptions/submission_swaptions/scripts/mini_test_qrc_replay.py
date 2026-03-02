from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

def maybe_reexec_with_submission_env() -> None:
    if os.environ.get("SUBMISSION_REEXEC") == "1":
        return
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
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
        os.execve(
            str(preferred),
            [str(preferred), "-m", "submission_swaptions.scripts.mini_test_qrc_replay", *sys.argv[1:]],
            env,
        )


maybe_reexec_with_submission_env()

from submission_swaptions.plot_model_comparison import run_comparison


def _extract_qrc_curve(csv_path: Path, model_name: str = "photonic_qrc_feedback") -> np.ndarray:
    df = pd.read_csv(csv_path)
    row = df[df["model"].astype(str) == str(model_name)]
    if row.empty:
        raise RuntimeError(f"Missing {model_name} in {csv_path}")
    arr = row.iloc[0]["horizon_surface_mape"]
    if isinstance(arr, str):
        vals = np.asarray(json.loads(arr), dtype=float)
    else:
        vals = np.asarray(arr, dtype=float)
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Mini-test: sweep best config replay consistency for photonic QRC.")
    p.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    p.add_argument("--level", type=int, choices=[1, 2], default=1)
    p.add_argument("--out_root", type=str, default="results/qrc_replay_minitest")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_sweep = out_root / "sweep"
    out_replay = out_root / "replay"
    out_sweep.mkdir(parents=True, exist_ok=True)
    out_replay.mkdir(parents=True, exist_ok=True)

    run_comparison(
        data_dir=args.data_dir,
        level=int(args.level),
        backend="sim",
        seed=int(args.seed),
        out_dir=out_sweep,
        include_quantum=False,
        include_ported=False,
        include_classical=True,
        classical_models=["persistence_naive", "photonic_qrc_feedback"],
        fast=True,
        lookback=26,
        forecast_horizons=6,
        qrc_mode="auto",
        qrc_target="delta",
        qrc_baseline="persistence",
        qrc_sweep_objective="mc_then_mape",
        pqrc_sweep=True,
        sweep_modes=True,
        sweep_gain=True,
        sweep_input_scale=True,
        sweep_ridge=True,
        pqrc_sweep_subset=96,
        pqrc_sweep_topk=2,
        sanity_checks=False,
    )

    hist_path = out_sweep / f"level{int(args.level)}_qrc_sweep_history.csv"
    if not hist_path.exists():
        raise RuntimeError(f"Missing sweep history CSV: {hist_path}")
    hist_df = pd.read_csv(hist_path)
    for col in ["pqrc_modes", "pqrc_gain", "pqrc_input_scale", "pqrc_ridge", "pqrc_shots", "pqrc_budget"]:
        if col not in hist_df.columns:
            raise RuntimeError(f"Sweep history missing expected hyperparameter column: {col}")
        vals = pd.to_numeric(hist_df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Sweep history contains non-finite values in {col}")

    best_cfg = out_sweep / "best_config.json"
    if not best_cfg.exists():
        raise RuntimeError(f"Missing best_config.json: {best_cfg}")

    run_comparison(
        data_dir=args.data_dir,
        level=int(args.level),
        backend="sim",
        seed=int(args.seed),
        out_dir=out_replay,
        include_quantum=False,
        include_ported=False,
        include_classical=True,
        classical_models=["persistence_naive", "photonic_qrc_feedback"],
        fast=True,
        lookback=26,
        forecast_horizons=6,
        qrc_load_from=str(out_sweep),
        sanity_checks=False,
    )

    s_curve = _extract_qrc_curve(out_sweep / f"level{int(args.level)}_metrics_summary.csv")
    r_curve = _extract_qrc_curve(out_replay / f"level{int(args.level)}_metrics_summary.csv")
    if s_curve.shape != r_curve.shape:
        raise RuntimeError(f"Replay mismatch shape: sweep={s_curve.shape} replay={r_curve.shape}")
    if not np.array_equal(s_curve, r_curve):
        raise RuntimeError("Replay mismatch: QRC horizon curve differs between sweep winner and qrc_load_from replay.")

    print("[mini-test] PASS: sweep winner replay is identical and sweep hyperparameters contain no NaNs.")


if __name__ == "__main__":
    main()
