from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import warnings

# Benchmark examples:
# 1) Baseline comparison with raw-surface persistence anchor:
#    python -m submission_swaptions.plot_model_comparison --level 1 --forecast_horizons 6 \
#      --classical_models persistence_surface_naive,persistence_naive,factor_ar,mlp,gru,lstm,reservoir \
#      --out_dir results/baseline_surface_persistence
# 2) Quantum/photonic sweep optimizing surface objective at high capacity:
#    python -m submission_swaptions.plot_model_comparison --level 1 --forecast_horizons 6 --pqrc_sweep \
#      --performance_preset --pqrc_modes 20 --pqrc_nphotons 10 --sweep_objective surface_mape \
#      --classical_models persistence_surface_naive,photonic_qrc_feedback,photonic_qrc_no_feedback \
#      --out_dir results/qrc_surface_sweep_l1
# 3) Full comparison replaying best swept config:
#    python -m submission_swaptions.plot_model_comparison --level 1 --forecast_horizons 6 \
#      --run_quantum --run_ported --qrc_load_from results/qrc_surface_sweep_l1 \
#      --out_dir results/all_models_surface_l1

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
        os.execve(str(preferred), [str(preferred), "-m", "submission_swaptions.plot_model_comparison", *sys.argv[1:]], env)


if __name__ == "__main__":
    maybe_reexec_with_submission_env()

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constraints import validate_constraints
from .feature_selection import ridge_validation_mae_score, wrapper_forward_selection
from .logging_utils import append_jsonl, stable_hash_array, stable_hash_config, summarize_matrix
from .models_classical import train_classical_forecaster
from .pipeline import _level_cfg, _load_submission_config, load_and_preprocess
from .positive_transform import PositiveLogTransform
from .qrc_config import QRCConfig, config_to_hash, load_config, save_config
from .sanity_checks import run_fast_sanity_checks

try:
    from .models_ported_hybridai import (
        run_ported_model,
    )

    _HAS_PORTED = True
    _PORTED_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency path
    run_ported_model = None  # type: ignore[assignment]
    _HAS_PORTED = False
    _PORTED_IMPORT_ERROR = str(exc)

try:
    from .models_quantum import QuantumAnnotatorResidual

    _HAS_QUANTUM = True
except Exception as exc:  # pragma: no cover - optional dependency path
    QuantumAnnotatorResidual = None  # type: ignore[assignment]
    _HAS_QUANTUM = False
    _QUANTUM_IMPORT_ERROR = str(exc)


def _to_surface_space(bundle: Any, z_pred_norm: np.ndarray) -> np.ndarray:
    z_pred = bundle.normalizer.inverse_transform_seq(z_pred_norm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        surf_scaled = bundle.pca.inverse_transform_seq(z_pred)
    floor = float(getattr(bundle, "surface_floor", 0.0))
    surf = np.clip(surf_scaled * bundle.price_scale, floor, bundle.clip_cap)
    return np.nan_to_num(surf, nan=floor, posinf=bundle.clip_cap, neginf=floor)


PERSISTENCE_SURFACE_NAME = "persistence_surface_naive"
PERSISTENCE_FACTOR_NAME = "persistence_naive"


@dataclass
class SurfacePersistencePredictor:
    """Raw-surface persistence forecaster used as the default comparison anchor."""

    horizon: int
    floor: float
    clip_cap: float

    def fit(self, x_seq_surface: np.ndarray, y_surface: np.ndarray | None = None) -> "SurfacePersistencePredictor":
        _ = np.asarray(x_seq_surface, dtype=float)
        _ = None if y_surface is None else np.asarray(y_surface, dtype=float)
        return self

    def predict(self, x_seq_surface: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq_surface, dtype=float)
        if x.ndim != 3:
            raise ValueError(f"x_seq_surface must be [N,L,D], got {x.shape}.")
        last_surface = x[:, -1, :]
        last_surface = np.clip(
            np.nan_to_num(last_surface, nan=float(self.floor), posinf=float(self.clip_cap), neginf=float(self.floor)),
            float(self.floor),
            float(self.clip_cap),
        )
        return np.repeat(last_surface[:, None, :], int(self.horizon), axis=1)


def _surface_persistence_prediction(bundle: Any, split_slice: slice, horizon: int) -> np.ndarray:
    x_seq_surface = np.asarray(bundle.ds.x_seq[split_slice], dtype=float)
    predictor = SurfacePersistencePredictor(
        horizon=int(horizon),
        floor=float(getattr(bundle, "surface_floor", 0.0)),
        clip_cap=float(bundle.clip_cap),
    )
    predictor.fit(x_seq_surface)
    return predictor.predict(x_seq_surface)


def _surface_to_factor_norm(bundle: Any, y_surface: np.ndarray) -> np.ndarray:
    floor = float(getattr(bundle, "surface_floor", 0.0))
    y_s = np.asarray(y_surface, dtype=float)
    y_s = np.clip(np.nan_to_num(y_s, nan=floor, posinf=bundle.clip_cap, neginf=floor), floor, bundle.clip_cap)
    y_scaled = np.clip(y_s / float(max(1e-12, bundle.price_scale)), floor / float(max(1e-12, bundle.price_scale)), 100.0)
    n, h, d_surface = y_scaled.shape
    z = bundle.pca.transform(y_scaled.reshape(-1, d_surface)).reshape(n, h, -1)
    z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
    return np.asarray(bundle.normalizer.transform_seq(z), dtype=float)


def _persistence_anchor_name(eval_rows: list[dict[str, Any]]) -> str:
    names = {str(r.get("model", "")).strip().lower() for r in eval_rows}
    if PERSISTENCE_SURFACE_NAME in names:
        return PERSISTENCE_SURFACE_NAME
    if PERSISTENCE_FACTOR_NAME in names:
        return PERSISTENCE_FACTOR_NAME
    return PERSISTENCE_FACTOR_NAME


def _sanitize_objective_name(name: str) -> str:
    n = str(name).strip().lower()
    aliases = {
        "mape": "surface_mape",
        "mae": "surface_mae",
        "rmse": "surface_rmse",
        "wape": "surface_wape",
    }
    n = aliases.get(n, n)
    if n not in {"surface_mape", "surface_wape", "surface_mae", "surface_rmse", "mc_then_mape"}:
        return "surface_mape"
    return n


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    den = np.maximum(np.abs(y_true), float(max(1e-12, eps)))
    return float(np.mean(np.abs(y_pred - y_true) / den) * 100.0)


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    den = np.abs(y_true) + np.abs(y_pred) + float(max(1e-12, eps))
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def _compute_wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    num = float(np.sum(np.abs(y_pred - y_true)))
    den = float(np.sum(np.abs(y_true) + float(max(1e-12, eps))))
    return float((num / max(den, float(max(1e-12, eps)))) * 100.0)


def _compute_metric_block(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mape": _compute_mape(y_true, y_pred, eps=eps),
        "smape": _compute_smape(y_true, y_pred, eps=eps),
        "wape": _compute_wape(y_true, y_pred, eps=eps),
    }


def _horizon_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, eps: float) -> np.ndarray:
    m = str(metric).strip().lower()
    err = y_pred - y_true
    if m == "mae":
        out = np.mean(np.abs(err), axis=(0, 2))
    elif m == "rmse":
        out = np.sqrt(np.mean(err * err, axis=(0, 2)))
    elif m == "mape":
        den = np.maximum(np.abs(y_true), float(max(1e-12, eps)))
        out = np.mean(np.abs(err) / den, axis=(0, 2)) * 100.0
    elif m == "smape":
        den = np.abs(y_true) + np.abs(y_pred) + float(max(1e-12, eps))
        out = np.mean(2.0 * np.abs(err) / den, axis=(0, 2)) * 100.0
    elif m == "wape":
        num = np.sum(np.abs(err), axis=(0, 2))
        den = np.sum(np.abs(y_true) + float(max(1e-12, eps)), axis=(0, 2))
        out = (num / np.maximum(den, float(max(1e-12, eps)))) * 100.0
    else:
        raise ValueError(f"Unsupported horizon metric: {metric}")
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)


def _compute_delta_to_next_days(dates: np.ndarray) -> np.ndarray:
    dates_ts = pd.to_datetime(pd.Series(dates), errors="coerce")
    if dates_ts.notna().sum() < 2:
        return np.ones((len(dates),), dtype=float)
    dt = np.diff(dates_ts.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[D]").astype(float)
    dt = np.nan_to_num(dt, nan=1.0, posinf=30.0, neginf=1.0)
    dt = np.clip(dt, 0.0, 30.0)
    if dt.size == 0:
        return np.ones((len(dates),), dtype=float)
    return np.concatenate([dt, dt[-1:]], axis=0)


def _dataset_signature(bundle: Any, *, level: int, lookback: int, horizon: int) -> str:
    import hashlib

    payload = {
        "level": int(level),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "dates_head": [str(x) for x in np.asarray(bundle.ds.dates[:16]).tolist()],
        "dates_tail": [str(x) for x in np.asarray(bundle.ds.dates[-16:]).tolist()],
        "starts_hash": hashlib.sha1(np.asarray(bundle.ds.starts, dtype=np.int64).tobytes()).hexdigest(),
        "x_shape": [int(v) for v in np.asarray(bundle.x_test_norm).shape],
        "y_shape": [int(v) for v in np.asarray(bundle.y_test_norm).shape],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _select_row_by_hash(df: pd.DataFrame, model_hash: str | None = None) -> pd.Series:
    if df.empty:
        raise RuntimeError("No sweep rows available for qrc_load_from.")
    if model_hash:
        hit = df[df["model_hash"].astype(str) == str(model_hash)]
        if hit.empty:
            raise RuntimeError(f"qrc_model_hash={model_hash} not found in sweep rows.")
        return hit.iloc[0]
    sort_cols = []
    if "collapsed" in df.columns:
        sort_cols.append("collapsed")
    if "objective_value" in df.columns:
        sort_cols.append("objective_value")
    elif "score" in df.columns:
        sort_cols.append("score")
    if "val_mape" in df.columns:
        sort_cols.append("val_mape")
    if "val_mae" in df.columns:
        sort_cols.append("val_mae")
    if not sort_cols:
        return df.iloc[0]
    return df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).iloc[0]


def _resolve_sweep_history_path(src: Path, *, level: int) -> Path | None:
    hist = src / f"level{int(level)}_qrc_sweep_history.csv"
    if hist.exists():
        return hist
    cands = sorted(src.glob(f"level{int(level)}_qrc_sweep_history_*.csv"))
    if cands:
        return cands[-1]
    return None


def _objective_series(df: pd.DataFrame, objective_name: str) -> pd.Series:
    obj = _sanitize_objective_name(str(objective_name))
    col_map = {
        "surface_mape": "val_surface_mape",
        "surface_wape": "val_surface_wape",
        "surface_mae": "val_surface_mae",
        "surface_rmse": "val_surface_rmse",
    }
    col = col_map.get(obj, "")
    if col and col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(s).any():
            return s
    if "objective_value" in df.columns:
        s = pd.to_numeric(df["objective_value"], errors="coerce")
        if np.isfinite(s).any():
            return s
    if "score" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce")
        if np.isfinite(s).any():
            return s
    if "val_mape" in df.columns:
        return pd.to_numeric(df["val_mape"], errors="coerce")
    return pd.Series(np.full((len(df),), np.inf, dtype=float), index=df.index, dtype=float)


def _select_best_sweep_row(
    df: pd.DataFrame,
    *,
    objective_name: str,
    preferred_models: tuple[str, ...] = ("photonic_qrc_feedback",),
) -> pd.Series:
    if df.empty:
        raise RuntimeError("No sweep rows available for qrc_load_from.")
    cand = df.copy()
    if preferred_models and "model" in cand.columns:
        pref = {str(m).strip().lower() for m in preferred_models if str(m).strip()}
        model_norm = cand["model"].astype(str).str.strip().str.lower()
        sub = cand.loc[model_norm.isin(pref)].copy()
        if not sub.empty:
            cand = sub
    if "failed" in cand.columns:
        failed = cand["failed"].astype(str).str.lower().isin({"1", "true", "yes"})
        cand = cand.loc[~failed].copy()
    if cand.empty:
        raise RuntimeError("Sweep history contains only failed rows.")
    obj = _sanitize_objective_name(str(objective_name))
    if "objective_name" in cand.columns:
        oname = cand["objective_name"].astype(str).str.lower()
        obj_rows = cand.loc[oname == obj]
        if not obj_rows.empty:
            cand = obj_rows.copy()
    cand["_objective_for_sort"] = _objective_series(cand, obj)
    if "collapsed" in cand.columns:
        cand["_collapsed_for_sort"] = cand["collapsed"].astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        cand["_collapsed_for_sort"] = False
    if "score" in cand.columns:
        cand["_score_for_sort"] = pd.to_numeric(cand["score"], errors="coerce").fillna(np.inf)
    else:
        cand["_score_for_sort"] = np.inf
    if "val_mape" in cand.columns:
        cand["_val_mape_for_sort"] = pd.to_numeric(cand["val_mape"], errors="coerce").fillna(np.inf)
    else:
        cand["_val_mape_for_sort"] = np.inf
    if "val_mae" in cand.columns:
        cand["_val_mae_for_sort"] = pd.to_numeric(cand["val_mae"], errors="coerce").fillna(np.inf)
    else:
        cand["_val_mae_for_sort"] = np.inf
    cand = cand.sort_values(
        by=["_collapsed_for_sort", "_objective_for_sort", "_score_for_sort", "_val_mape_for_sort", "_val_mae_for_sort"],
        ascending=[True, True, True, True, True],
    )
    return cand.iloc[0]


def _row_to_qrc_cfg(row: pd.Series) -> tuple[dict[str, Any], str]:
    cfg_blob = row.get("config")
    if not isinstance(cfg_blob, str) or not cfg_blob.strip():
        raise RuntimeError("Sweep row has empty config payload.")
    cfg_dict = dict(json.loads(cfg_blob))
    cfg = QRCConfig.from_dict(cfg_dict).validate().to_dict()
    h = str(row.get("model_hash", "")).strip() or str(cfg.get("config_hash", config_to_hash(cfg)))
    return cfg, h


def _load_qrc_config_from_source(
    source: str | Path,
    *,
    level: int,
    model_hash: str | None = None,
    objective_name: str = "surface_mape",
    enforce_optimal: bool = True,
    allow_nonoptimal_hash: bool = False,
    preferred_models: tuple[str, ...] = ("photonic_qrc_feedback",),
) -> tuple[dict[str, Any], str]:
    src = Path(source)
    if src.is_file():
        cfg = load_config(src).to_dict()
        cfg_hash = str(cfg.get("config_hash", config_to_hash(cfg)))
        if bool(enforce_optimal):
            hist = _resolve_sweep_history_path(src.parent, level=int(level))
            if hist is not None:
                df = pd.read_csv(hist)
                best_row = _select_best_sweep_row(
                    df,
                    objective_name=str(objective_name),
                    preferred_models=preferred_models,
                )
                best_cfg, best_hash = _row_to_qrc_cfg(best_row)
                if str(best_hash) != str(cfg_hash):
                    print(
                        "[model_compare] qrc_optimal_guard: supplied config file is not the best sweep row "
                        f"for objective={_sanitize_objective_name(str(objective_name))}; "
                        f"using model_hash={best_hash} instead of {cfg_hash}."
                    )
                    return best_cfg, str(best_hash)
        return cfg, cfg_hash
    if not src.exists():
        raise FileNotFoundError(f"qrc_load_from path not found: {src}")

    hist = _resolve_sweep_history_path(src, level=int(level))
    if hist is None:
        best_cfg = src / "best_config.json"
        if best_cfg.exists() and model_hash is None:
            cfg = load_config(best_cfg).to_dict()
            return cfg, str(cfg.get("config_hash", config_to_hash(cfg)))
        raise RuntimeError(f"Could not find sweep history in {src}")

    df = pd.read_csv(hist)
    best_row = _select_best_sweep_row(
        df,
        objective_name=str(objective_name),
        preferred_models=preferred_models,
    )
    best_cfg, best_hash = _row_to_qrc_cfg(best_row)
    best_cfg_path = src / "best_config.json"
    if best_cfg_path.exists():
        saved_cfg = load_config(best_cfg_path).to_dict()
        saved_hash = str(saved_cfg.get("config_hash", config_to_hash(saved_cfg)))
        if saved_hash != str(best_hash):
            print(
                "[model_compare] qrc_optimal_guard: best_config.json is stale relative to sweep history "
                f"(saved={saved_hash}, best={best_hash}); replaying best sweep row."
            )

    if model_hash:
        row = _select_row_by_hash(df, model_hash=model_hash)
        cfg, cfg_hash = _row_to_qrc_cfg(row)
        if bool(enforce_optimal) and (str(cfg_hash) != str(best_hash)) and (not bool(allow_nonoptimal_hash)):
            raise RuntimeError(
                f"qrc_model_hash={model_hash} is not optimal for objective={_sanitize_objective_name(str(objective_name))}. "
                f"Best available hash is {best_hash}. Pass --allow_nonoptimal_qrc_hash to override."
            )
        return cfg, str(cfg_hash)

    if bool(enforce_optimal):
        return best_cfg, str(best_hash)

    # Back-compat fallback when optimal guard disabled.
    best_cfg_path = src / "best_config.json"
    if best_cfg_path.exists():
        cfg = load_config(best_cfg_path).to_dict()
        return cfg, str(cfg.get("config_hash", config_to_hash(cfg)))
    return best_cfg, str(best_hash)


def _kernel_gram_summary(gram: np.ndarray) -> dict[str, float]:
    g = np.asarray(gram, dtype=float)
    g = np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
    diag = np.diag(g)
    if g.shape[0] > 1:
        mask = ~np.eye(g.shape[0], dtype=bool)
        off = g[mask]
    else:
        off = np.array([0.0], dtype=float)
    return {
        "diag_mean": float(np.mean(diag)),
        "diag_std": float(np.std(diag)),
        "offdiag_mean": float(np.mean(off)),
        "offdiag_std": float(np.std(off)),
    }


def _model_family(model_name: str) -> str:
    m = str(model_name).lower()
    if m.startswith("persistence"):
        return "persistence"
    if m.startswith("seasonal_naive"):
        return "persistence"
    if m in {
        "mlp",
        "gru",
        "lstm",
        "factor_ar",
        "reservoir",
        "photonic_qrc_feedback",
        "photonic_qrc_no_feedback",
        "persist_qrc_weak",
        "photonic_memory",
        "photonic_memory_no_feedback",
    }:
        return "classical"
    if m.startswith("ported_"):
        return "ported"
    return "quantum"


def _is_quantum_named(model_name: str) -> bool:
    m = str(model_name).lower()
    if m.startswith("photonic_qrc"):
        return True
    return ("qrc" in m) or ("q_bottleneck" in m) or ("quantum" in m)


def _line_style_for_model(model_name: str, color_idx: int) -> dict[str, Any]:
    m = str(model_name).lower()
    # Keep MLP visually dominant and fixed green so it never disappears under overlaps.
    if m == "mlp":
        return {"color": "#00a651", "lw": 3.2, "marker": "o", "ms": 5.5, "alpha": 1.0, "zorder": 7, "ls": "-"}
    if m.startswith("persistence"):
        return {"color": "#555555", "lw": 2.2, "marker": "s", "ms": 4.5, "alpha": 0.95, "zorder": 6, "ls": "--"}
    if m.startswith("seasonal_naive"):
        return {"color": "#7f7f7f", "lw": 2.0, "marker": "P", "ms": 4.5, "alpha": 0.92, "zorder": 6, "ls": "-."}
    if m in {"gru", "lstm", "factor_ar"}:
        palette = {"gru": "#1f77b4", "lstm": "#ff7f0e", "factor_ar": "#9467bd"}
        return {
            "color": palette[m],
            "lw": 2.6,
            "marker": "o",
            "ms": 4.5,
            "alpha": 0.95,
            "zorder": 6,
            "ls": "-",
        }
    if m == "reservoir":
        return {
            "color": "#8c564b",
            "lw": 2.6,
            "marker": "D",
            "ms": 4.6,
            "alpha": 0.95,
            "zorder": 6,
            "ls": "-",
        }
    if m == "photonic_qrc_feedback":
        return {"color": "#d62728", "lw": 2.7, "marker": "v", "ms": 4.8, "alpha": 0.95, "zorder": 6, "ls": "-"}
    if m == "photonic_qrc_no_feedback":
        return {"color": "#e377c2", "lw": 2.3, "marker": "v", "ms": 4.4, "alpha": 0.9, "zorder": 6, "ls": "--"}
    if m == "persist_qrc_weak":
        return {"color": "#bcbd22", "lw": 2.4, "marker": ">", "ms": 4.6, "alpha": 0.95, "zorder": 6, "ls": "-"}
    if m == "photonic_memory":
        return {"color": "#17becf", "lw": 2.7, "marker": "^", "ms": 4.8, "alpha": 0.95, "zorder": 6, "ls": "-"}
    if m == "photonic_memory_no_feedback":
        return {"color": "#2ca02c", "lw": 2.3, "marker": "^", "ms": 4.4, "alpha": 0.9, "zorder": 6, "ls": "--"}

    cmap = plt.get_cmap("tab20")
    color = cmap(int(color_idx) % 20)
    fam = _model_family(model_name)
    if fam == "ported":
        return {"color": color, "lw": 1.5, "marker": "^", "ms": 4.0, "alpha": 0.75, "zorder": 4, "ls": "-."}
    return {"color": color, "lw": 1.7, "marker": "x", "ms": 4.2, "alpha": 0.78, "zorder": 5, "ls": "--"}


def _ordered_rows_for_lines(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classical = [r for r in eval_rows if _model_family(str(r["model"])) in {"persistence", "classical"}]
    others = [r for r in eval_rows if _model_family(str(r["model"])) not in {"persistence", "classical"}]
    return others + classical


def _parse_metrics_arg(metrics_arg: str | None, default_metric: str) -> list[str]:
    allowed = ["mape", "mae", "rmse", "smape", "wape"]
    if metrics_arg is None or not str(metrics_arg).strip():
        requested = allowed
    else:
        requested = [t.strip().lower() for t in str(metrics_arg).split(",") if t.strip()]
    out: list[str] = []
    for m in requested:
        if m in allowed and m not in out:
            out.append(m)
    if not out:
        dm = str(default_metric).strip().lower()
        out = [dm] if dm in allowed else ["mape"]
    return out


def _evaluate_model(
    name: str,
    y_pred_test_norm: np.ndarray,
    y_true_test_norm: np.ndarray,
    y_true_test_surface: np.ndarray,
    bundle: Any,
    mape_eps: float,
    y_pred_test_surface: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred_surface = (
        np.asarray(y_pred_test_surface, dtype=float)
        if y_pred_test_surface is not None
        else _to_surface_space(bundle, y_pred_test_norm)
    )
    floor = float(getattr(bundle, "surface_floor", 0.0))
    nonfinite = int(np.size(pred_surface) - np.isfinite(pred_surface).sum())
    nonpositive = int(np.sum(pred_surface <= 0.0))
    if nonfinite > 0 or nonpositive > 0:
        warnings.warn(
            f"{name}: surface output sanity check failed (nonfinite={nonfinite}, nonpositive={nonpositive}, floor={floor:.3e}).",
            RuntimeWarning,
        )
    fac = _compute_metric_block(y_true_test_norm, y_pred_test_norm, eps=mape_eps)
    surf = _compute_metric_block(y_true_test_surface, pred_surface, eps=mape_eps)

    row = {
        "model": name,
        "factor_mae": fac["mae"],
        "factor_rmse": fac["rmse"],
        "factor_mape": fac["mape"],
        "factor_smape": fac["smape"],
        "factor_wape": fac["wape"],
        "surface_mae": surf["mae"],
        "surface_rmse": surf["rmse"],
        "surface_mape": surf["mape"],
        "surface_smape": surf["smape"],
        "surface_wape": surf["wape"],
        "horizon_factor_mae": _horizon_metric(y_true_test_norm, y_pred_test_norm, metric="mae", eps=mape_eps),
        "horizon_factor_rmse": _horizon_metric(y_true_test_norm, y_pred_test_norm, metric="rmse", eps=mape_eps),
        "horizon_factor_mape": _horizon_metric(y_true_test_norm, y_pred_test_norm, metric="mape", eps=mape_eps),
        "horizon_factor_smape": _horizon_metric(y_true_test_norm, y_pred_test_norm, metric="smape", eps=mape_eps),
        "horizon_factor_wape": _horizon_metric(y_true_test_norm, y_pred_test_norm, metric="wape", eps=mape_eps),
        "horizon_surface_mae": _horizon_metric(y_true_test_surface, pred_surface, metric="mae", eps=mape_eps),
        "horizon_surface_rmse": _horizon_metric(y_true_test_surface, pred_surface, metric="rmse", eps=mape_eps),
        "horizon_surface_mape": _horizon_metric(y_true_test_surface, pred_surface, metric="mape", eps=mape_eps),
        "horizon_surface_smape": _horizon_metric(y_true_test_surface, pred_surface, metric="smape", eps=mape_eps),
        "horizon_surface_wape": _horizon_metric(y_true_test_surface, pred_surface, metric="wape", eps=mape_eps),
        "pred_test_surface": pred_surface,
        "pred_test_norm": y_pred_test_norm,
        "surface_floor": floor,
        "surface_nonfinite_count": nonfinite,
        "surface_nonpositive_count": nonpositive,
    }
    if isinstance(extra, dict):
        row.update(extra)
    return row


def _split_surface_mape_extra(
    *,
    y_true_train_surface: np.ndarray,
    y_pred_train_surface: np.ndarray,
    y_true_val_surface: np.ndarray,
    y_pred_val_surface: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    return {
        "surface_mape_train": _compute_mape(y_true_train_surface, y_pred_train_surface, eps=eps),
        "surface_mape_val": _compute_mape(y_true_val_surface, y_pred_val_surface, eps=eps),
        "horizon_surface_mape_train": _horizon_metric(y_true_train_surface, y_pred_train_surface, metric="mape", eps=eps),
        "horizon_surface_mape_val": _horizon_metric(y_true_val_surface, y_pred_val_surface, metric="mape", eps=eps),
    }


def _train_fit_tune_slices(
    n_train: int,
    *,
    tune_fraction: float = 0.15,
    min_train: int = 16,
    min_tune: int = 8,
) -> tuple[slice, slice]:
    n = int(n_train)
    n_tune = int(max(min_tune, round(float(tune_fraction) * n)))
    n_tune = int(min(max(1, n_tune), max(1, n - min_train)))
    split = int(n - n_tune)
    if split <= 0 or split >= n:
        raise ValueError(
            "Unable to build internal train/tune split: "
            f"n={n}, split={split}, min_train={min_train}, min_tune={min_tune}."
        )
    return slice(0, split), slice(split, n)


def _build_model_hash(row: dict[str, Any], *, seed: int, level: int, backend: str) -> str:
    cfg: dict[str, Any] = {
        "model": str(row.get("model", "")),
        "seed": int(seed),
        "level": int(level),
        "backend": str(backend),
    }
    if isinstance(row.get("config_fingerprint"), dict):
        cfg["config_fingerprint"] = row.get("config_fingerprint")
    scalar_keys = {
        "quantum_mode",
        "selected_dim",
        "w_star",
        "w_min",
        "shots_per_eval",
        "total_qevals",
        "total_shots",
        "pqrc_M",
        "pqrc_modes",
        "pqrc_Nph",
        "pqrc_budget",
        "pqrc_gain",
        "pqrc_feature",
        "pqrc_pseudocount",
        "pqrc_higher_order",
        "pqrc_budget_alloc",
        "pqrc_shots",
        "pqrc_ridge",
        "pqrc_in_pca",
        "pqrc_factor_cap",
        "qrc_mode",
        "qrc_target",
        "qrc_baseline",
        "target_transform",
        "y_floor_mode",
        "y_floor_value",
        "tau",
        "qrc_gate_tau",
        "qrc_resid_clip",
        "qrc_residvar_penalty",
        "qrc_feat_norm",
        "qrc_residual_penalty",
        "qrc_mode_used",
        "pqrc_input_scale",
        "pqrc_min_residual_scale",
        "persist_qrc_alpha",
        "persist_qrc_rclip",
        "persist_qrc_drive_scale",
        "qrc_config_hash",
        "dataset_signature",
    }
    for k in scalar_keys:
        v = row.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            cfg[k] = v
    return stable_hash_config(cfg, name=f"model:{cfg['model']}")


def _compute_collapse_diagnostics(
    eval_rows: list[dict[str, Any]],
    *,
    y_true_test_surface: np.ndarray,
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    if not eval_rows:
        return []
    anchor_name = _persistence_anchor_name(eval_rows)
    persistence_row = next((r for r in eval_rows if str(r.get("model", "")).strip().lower() == anchor_name), None)
    if persistence_row is None:
        return []

    y_persist = np.asarray(persistence_row["pred_test_surface"], dtype=float)
    y_true = np.asarray(y_true_test_surface, dtype=float)
    true_res = y_true - y_persist
    h = int(y_persist.shape[1])
    records: list[dict[str, Any]] = []
    for row in eval_rows:
        y_pred = np.asarray(row["pred_test_surface"], dtype=float)
        pred_res = y_pred - y_persist
        gate_arr_raw = row.get("qrc_gate")
        gate_arr = None
        if gate_arr_raw is not None:
            gate_arr = np.asarray(gate_arr_raw, dtype=float)
            y_pred_norm_ref = np.asarray(row.get("pred_test_norm", np.empty((0, 0, 0))), dtype=float)
            if gate_arr.shape != y_pred_norm_ref.shape:
                gate_arr = None
        mean_diffs: list[float] = []
        mape_deltas: list[float] = []
        for h_idx in range(h):
            diff_h = np.abs(y_pred[:, h_idx, :] - y_persist[:, h_idx, :]).reshape(-1)
            pred_h = pred_res[:, h_idx, :].reshape(-1)
            true_h = true_res[:, h_idx, :].reshape(-1)
            std_pred = float(np.std(pred_h))
            std_true = float(np.std(true_h))
            std_ratio = float(std_pred / (std_true + 1e-12))
            if std_pred > 1e-12 and std_true > 1e-12:
                corr = float(np.corrcoef(pred_h, true_h)[0, 1])
            else:
                corr = 0.0
            y_h = y_true[:, h_idx, :]
            yp_h = y_pred[:, h_idx, :]
            yp_p_h = y_persist[:, h_idx, :]
            mape_h = _compute_mape(y_h, yp_h, eps=eps)
            mape_p_h = _compute_mape(y_h, yp_p_h, eps=eps)
            mape_delta = float(mape_h - mape_p_h)
            mean_abs = float(np.mean(diff_h))
            mean_diffs.append(mean_abs)
            mape_deltas.append(mape_delta)
            if gate_arr is not None:
                gate_h = gate_arr[:, h_idx, :].reshape(-1)
                gate_mean = float(np.mean(gate_h))
                gate_std = float(np.std(gate_h))
            else:
                gate_mean = float("nan")
                gate_std = float("nan")
            records.append(
                {
                    "model": str(row["model"]),
                    "model_hash": str(row.get("model_hash", "")),
                    "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                    "dataset_signature": str(row.get("dataset_signature", "")),
                    "horizon": int(h_idx + 1),
                    "max_abs_diff_h": float(np.max(diff_h)),
                    "mean_abs_diff_h": mean_abs,
                    "corr_h": float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)),
                    "residual_std_pred_h": std_pred,
                    "residual_std_true_h": std_true,
                    "residual_std_ratio": std_ratio,
                    "gate_mean": gate_mean,
                    "gate_std": gate_std,
                    "mape_vs_persistence_delta_h": mape_delta,
                }
            )
        collapse_mean = float(np.mean(mean_diffs)) if mean_diffs else 0.0
        row["collapse_mean_abs_diff"] = collapse_mean
        model_name_l = str(row.get("model", "")).lower()
        is_qrc_family = ("qrc" in model_name_l) or ("q_bottleneck" in model_name_l) or ("quantum" in model_name_l)
        row["collapse_failed"] = bool(is_qrc_family and collapse_mean < 1e-6)
        row["no_gain"] = bool(is_qrc_family and (float(np.mean(mape_deltas)) >= 0.0 if mape_deltas else False))
    return records


def _write_feature_stats_csv(
    eval_rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in eval_rows:
        fs = row.get("feature_stats")
        if not isinstance(fs, dict):
            continue
        train_std = np.asarray(fs.get("feat_std_train", []), dtype=float)
        val_std = np.asarray(fs.get("feat_std_val", []), dtype=float)
        if train_std.size == 0 or val_std.size == 0:
            continue
        records.append(
            {
                "model": str(row.get("model", "")),
                "model_hash": str(row.get("model_hash", "")),
                "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                "dataset_signature": str(row.get("dataset_signature", "")),
                "feat_std_train_min": float(np.min(train_std)),
                "feat_std_train_median": float(np.median(train_std)),
                "feat_std_train_max": float(np.max(train_std)),
                "feat_std_val_min": float(np.min(val_std)),
                "feat_std_val_median": float(np.median(val_std)),
                "feat_std_val_max": float(np.max(val_std)),
                "feat_low_frac_train": float(np.mean(train_std < 1e-8)),
                "feat_low_frac_val": float(np.mean(val_std < 1e-8)),
                "feat_norm": str(fs.get("feat_norm", "")),
                "pqrc_pseudocount": float(fs.get("pqrc_pseudocount", np.nan)),
                "min_prob_before": float(fs.get("min_prob_before", np.nan)),
                "min_prob_after": float(fs.get("min_prob_after", np.nan)),
                "fraction_zero_bins_before": float(fs.get("fraction_zero_bins_before", np.nan)),
                "fraction_zero_bins_after": float(fs.get("fraction_zero_bins_after", np.nan)),
                "median_sigma_train": float(fs.get("median_sigma_train", fs.get("median_sigma_train_postnorm", np.nan))),
                "median_sigma_val": float(fs.get("median_sigma_val", fs.get("median_sigma_val_postnorm", np.nan))),
                "median_sigma_train_postnorm": float(fs.get("median_sigma_train_postnorm", fs.get("median_sigma_train", np.nan))),
                "median_sigma_val_postnorm": float(fs.get("median_sigma_val_postnorm", fs.get("median_sigma_val", np.nan))),
                "phi_train_checksum": str(fs.get("phi_train_checksum", "")),
                "phi_val_checksum": str(fs.get("phi_val_checksum", "")),
                "phi_train_norm_checksum": str(fs.get("phi_train_norm_checksum", "")),
                "phi_val_norm_checksum": str(fs.get("phi_val_norm_checksum", "")),
                "checksum_train": str(fs.get("phi_train_checksum", "")),
                "checksum_val": str(fs.get("phi_val_checksum", "")),
                "phi_train_mean": float(fs.get("phi_train_mean", np.nan)),
                "phi_train_std": float(fs.get("phi_train_std", np.nan)),
                "phi_val_mean": float(fs.get("phi_val_mean", np.nan)),
                "phi_val_std": float(fs.get("phi_val_std", np.nan)),
                "feat_std_train": json.dumps([float(v) for v in train_std], separators=(",", ":")),
                "feat_std_val": json.dumps([float(v) for v in val_std], separators=(",", ":")),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


def _plot_qrc_top3_vs_persistence(
    top_entries: list[dict[str, Any]],
    persistence_horizon_mape: np.ndarray,
    out_path: Path,
    persistence_label: str = PERSISTENCE_SURFACE_NAME,
) -> None:
    if not top_entries:
        return
    x = np.arange(1, int(np.asarray(persistence_horizon_mape).shape[0]) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, np.asarray(persistence_horizon_mape, dtype=float), label=str(persistence_label), lw=2.6, ls="--", color="#555555")
    for i, ent in enumerate(top_entries[:3]):
        y = np.asarray(ent["horizon_surface_mape"], dtype=float)
        label = f"{ent['model']}[{ent['model_hash'][:6]}]"
        style = _line_style_for_model(str(ent["model"]), color_idx=i + 1)
        ax.plot(x, y, label=label, **style)
    ax.set_title("QRC Top-3 vs Persistence (Validation MAPE by Horizon)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Surface MAPE (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_qrc_collapse_curves(top_entries: list[dict[str, Any]], out_path: Path) -> None:
    if not top_entries:
        return
    x = np.arange(1, int(np.asarray(top_entries[0]["collapse_curve"]).shape[0]) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, np.zeros_like(x, dtype=float), label=PERSISTENCE_SURFACE_NAME, lw=2.6, ls="--", color="#555555")
    for i, ent in enumerate(top_entries[:3]):
        y = np.asarray(ent["collapse_curve"], dtype=float)
        label = f"{ent['model']}[{ent['model_hash'][:6]}]"
        style = _line_style_for_model(str(ent["model"]), color_idx=i + 1)
        ax.plot(x, y, label=label, **style)
    ax.set_title("Collapse Curves (mean_abs_diff vs persistence)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("mean_abs_diff_h")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_qrc_feature_median_std(top_entries: list[dict[str, Any]], out_path: Path) -> None:
    if not top_entries:
        return
    labels = [f"{e['model']}[{e['model_hash'][:6]}]" for e in top_entries[:3]]
    vals = [float(e.get("feature_median_std", 0.0)) for e in top_entries[:3]]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(np.arange(len(labels)), vals, width=0.6, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Median feature std")
    ax.set_title("QRC Feature Median Std by Config")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_qrc_mode_mape_vs_horizon(
    persistence_row: dict[str, Any],
    seasonal_row: dict[str, Any] | None,
    qrc_best_auto: dict[str, Any] | None,
    qrc_best_residual: dict[str, Any] | None,
    out_path: Path,
) -> None:
    x = np.arange(1, int(np.asarray(persistence_row["horizon_surface_mape"]).shape[0]) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        x,
        np.asarray(persistence_row["horizon_surface_mape"], dtype=float),
        label=f"persistence[{str(persistence_row.get('model_hash',''))[:6]}]",
        color="#555555",
        lw=2.6,
        ls="--",
    )
    if seasonal_row is not None:
        ax.plot(
            x,
            np.asarray(seasonal_row["horizon_surface_mape"], dtype=float),
            label=f"seasonal13[{str(seasonal_row.get('model_hash',''))[:6]}]",
            color="#7f7f7f",
            lw=2.4,
            ls="-.",
        )
    if qrc_best_auto is not None:
        ax.plot(
            x,
            np.asarray(qrc_best_auto["horizon_surface_mape"], dtype=float),
            label=f"best_auto[{str(qrc_best_auto.get('model_hash',''))[:6]}]",
            color="#1f77b4",
            lw=2.6,
            marker="o",
            ms=4.0,
        )
    if qrc_best_residual is not None:
        ax.plot(
            x,
            np.asarray(qrc_best_residual["horizon_surface_mape"], dtype=float),
            label=f"best_residual[{str(qrc_best_residual.get('model_hash',''))[:6]}]",
            color="#d62728",
            lw=2.6,
            marker="s",
            ms=4.0,
        )
    ax.set_title("Photonic QRC vs Persistence + Seasonal13 (Validation MAPE by Horizon)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Surface MAPE (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_residual_std_ratio_vs_horizon(diag_df: pd.DataFrame, out_path: Path) -> None:
    if diag_df.empty:
        return
    q = diag_df[diag_df["model"].astype(str).str.contains("qrc|q_bottleneck|quantum", case=False, regex=True)]
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for model_name, grp in q.groupby("model"):
        grp2 = grp.sort_values("horizon")
        x = grp2["horizon"].to_numpy(dtype=int)
        y = grp2["residual_std_ratio"].to_numpy(dtype=float)
        ax.plot(x, y, label=str(model_name), lw=2.2, marker="o", ms=3.8)
    ax.axhline(1.2, color="#d62728", lw=1.8, ls="--", label="ratio=1.2")
    ax.set_title("Residual Std Ratio vs Horizon (Validation)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("residual_std_pred / residual_std_true")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_qrc_training_error_vs_epoch(curve_df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    if curve_df.empty:
        return
    if "epoch" not in curve_df.columns or "train_mae" not in curve_df.columns:
        return
    df = curve_df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["train_mae"] = pd.to_numeric(df["train_mae"], errors="coerce")
    if "val_mae" in df.columns:
        df["val_mae"] = pd.to_numeric(df["val_mae"], errors="coerce")
    df = df[np.isfinite(df["epoch"]) & np.isfinite(df["train_mae"])]
    if df.empty:
        return
    df = df.sort_values("epoch")
    trial = np.arange(1, len(df) + 1, dtype=int)
    train_mae_best = np.minimum.accumulate(df["train_mae"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(trial, train_mae_best, label="train_mae (best-so-far)", lw=2.0, marker="o", ms=3.2)
    if "val_mae" in df.columns:
        val_mae = df["val_mae"].to_numpy(dtype=float)
        if np.isfinite(val_mae).any():
            val_mae_best = np.minimum.accumulate(val_mae)
            ax.plot(trial, val_mae_best, label="val_mae (best-so-far)", lw=1.6, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Error (MAE)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gate_mean_vs_horizon(diag_df: pd.DataFrame, out_path: Path) -> None:
    if diag_df.empty:
        return
    q = diag_df.copy()
    q = q[np.isfinite(q["gate_mean"].to_numpy(dtype=float))]
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for model_name, grp in q.groupby("model"):
        grp2 = grp.sort_values("horizon")
        x = grp2["horizon"].to_numpy(dtype=int)
        y = grp2["gate_mean"].to_numpy(dtype=float)
        ax.plot(x, y, label=str(model_name), lw=2.2, marker="o", ms=3.8)
    ax.set_title("Gate Mean vs Horizon (Validation)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("gate_mean")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_feedback_sanity_csv(
    eval_rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in eval_rows:
        fs = row.get("feedback_sanity")
        if not isinstance(fs, dict):
            continue
        records.append(
            {
                "model": str(row.get("model", "")),
                "model_hash": str(row.get("model_hash", "")),
                "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                "dataset_signature": str(row.get("dataset_signature", "")),
                "feedback_enabled": bool(fs.get("feedback_enabled", False)),
                "feedback_inert": bool(fs.get("feedback_inert", False)),
                "pqrc_gain": float(fs.get("pqrc_gain", 0.0)),
                "checksum_gain0": str(fs.get("checksum_gain0", "")),
                "checksum_gain1": str(fs.get("checksum_gain1", "")),
                "phi_diff_gain0_vs_gain1": float(fs.get("phi_diff_gain0_vs_gain1", np.nan)),
                "phi_gain0_mean": float(fs.get("phi_gain0_mean", np.nan)),
                "phi_gain0_std": float(fs.get("phi_gain0_std", np.nan)),
                "phi_gain1_mean": float(fs.get("phi_gain1_mean", np.nan)),
                "phi_gain1_std": float(fs.get("phi_gain1_std", np.nan)),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


def _write_feature_sanity_csv(
    eval_rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in eval_rows:
        fs = row.get("feature_stats")
        if not isinstance(fs, dict):
            continue
        sanity = fs.get("feature_sanity")
        if not isinstance(sanity, dict):
            continue
        records.append(
            {
                "model": str(row.get("model", "")),
                "model_hash": str(row.get("model_hash", "")),
                "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                "dataset_signature": str(row.get("dataset_signature", "")),
                "modes_ref": int(sanity.get("modes_ref", 0)),
                "modes_alt": int(sanity.get("modes_alt", 0)),
                "checksum_ref": str(sanity.get("checksum_ref", "")),
                "checksum_gain0": str(sanity.get("checksum_gain0", "")),
                "checksum_gain1": str(sanity.get("checksum_gain1", "")),
                "checksum_modes_alt": str(sanity.get("checksum_modes_alt", "")),
                "diff_gain0_vs_gain1": float(sanity.get("diff_gain0_vs_gain1", np.nan)),
                "diff_modes_ref_vs_alt": float(sanity.get("diff_modes_ref_vs_alt", np.nan)),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


def _plot_metric_bars(metrics_df: pd.DataFrame, out_path: Path, metric: str = "mape") -> None:
    labels = metrics_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    factor_col = f"factor_{metric}"
    surface_col = f"surface_{metric}"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    axes[0].bar(x, metrics_df[factor_col], width=0.6)
    axes[0].set_title(f"Factor {metric.upper()} (Validation)")
    axes[0].set_ylabel(metric.upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, metrics_df[surface_col], width=0.6)
    axes[1].set_title(f"Surface {metric.upper()} (Validation)")
    axes[1].set_ylabel(metric.upper())
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _ensure_persistence_anchor(
    rows_use: list[dict[str, Any]],
    rows_all: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Always include the preferred persistence anchor for visual comparison."""
    anchor = _persistence_anchor_name(rows_all if rows_all else rows_use)
    has_persist = any(str(r.get("model", "")).strip().lower() == anchor for r in rows_use)
    if has_persist:
        return rows_use
    persist_row = next(
        (r for r in rows_all if str(r.get("model", "")).strip().lower() == anchor),
        None,
    )
    if persist_row is None:
        return rows_use
    return list(rows_use) + [persist_row]


def _plot_horizon_surface_metric(
    eval_rows: list[dict[str, Any]],
    out_path: Path,
    topk_lines: int,
    metric: str,
) -> None:
    if not eval_rows:
        return
    metric_norm = str(metric).strip().lower()
    rows_sorted = sorted(
        eval_rows,
        key=lambda r: (bool(r.get("collapse_failed", False)), float(r.get(f"surface_{metric_norm}", np.inf))),
    )
    rows_use = rows_sorted[: int(max(1, topk_lines))]
    rows_use = _ensure_persistence_anchor(rows_use, rows_sorted)

    h = int(np.asarray(rows_use[0][f"horizon_surface_{metric_norm}"]).shape[0])
    x = np.arange(1, h + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, row in enumerate(_ordered_rows_for_lines(rows_use)):
        y = np.asarray(row[f"horizon_surface_{metric_norm}"], dtype=float)
        style = _line_style_for_model(str(row["model"]), color_idx=idx)
        ax.plot(x, y, label=row["model"], **style)
    title_metric = "MAPE (%)" if metric_norm in {"mape", "smape", "wape"} else metric_norm.upper()
    ax.set_title(f"Surface {metric_norm.upper()} by Horizon Step (Validation)")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel(title_metric)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_maturity_surface_mape(
    eval_rows: list[dict[str, Any]],
    y_true_test_surface: np.ndarray,
    by_maturity: dict[float, list[tuple[float, int]]],
    out_path: Path,
    mape_eps: float,
    topk_lines: int,
) -> None:
    if not eval_rows:
        return
    rows_sorted = sorted(
        eval_rows,
        key=lambda r: (bool(r.get("collapse_failed", False)), float(r.get("surface_mape", np.inf))),
    )
    rows_use = rows_sorted[: int(max(1, topk_lines))]
    rows_use = _ensure_persistence_anchor(rows_use, rows_sorted)

    maturities = sorted(by_maturity.keys())
    x = np.array(maturities, dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, row in enumerate(_ordered_rows_for_lines(rows_use)):
        pred = np.asarray(row["pred_test_surface"], dtype=float)
        den = np.maximum(np.abs(y_true_test_surface), float(max(1e-12, mape_eps)))
        err_pct = np.abs(pred - y_true_test_surface) / den * 100.0
        vals = []
        for m in maturities:
            idxs = [j for _, j in by_maturity[m]]
            vals.append(float(np.mean(err_pct[:, :, idxs])))
        style = _line_style_for_model(str(row["model"]), color_idx=idx)
        ax.plot(x, vals, label=row["model"], **style)
    ax.set_title("Surface MAPE by Maturity (Validation)")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("MAPE (%)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_horizon_metrics(
    eval_rows: list[dict[str, Any]],
    out_csv: Path,
    out_png_mape: Path,
    out_png_mae: Path,
    out_png_rmse: Path,
    topk_lines: int,
    out_png_all_mape: Path | None = None,
) -> None:
    if not eval_rows:
        return
    rows_sorted = sorted(
        eval_rows,
        key=lambda r: (bool(r.get("collapse_failed", False)), float(r.get("surface_mape", np.inf))),
    )
    rows_use = rows_sorted[: int(max(1, topk_lines))]
    rows_use = _ensure_persistence_anchor(rows_use, rows_sorted)

    records: list[dict[str, Any]] = []
    for row in rows_sorted:
        y_mape = np.asarray(row["horizon_surface_mape"], dtype=float)
        y_mae = np.asarray(row["horizon_surface_mae"], dtype=float)
        y_rmse = np.asarray(row["horizon_surface_rmse"], dtype=float)
        y_smape = np.asarray(row["horizon_surface_smape"], dtype=float)
        y_wape = np.asarray(row["horizon_surface_wape"], dtype=float)
        for h_idx in range(1, y_mape.shape[0] + 1):
            records.append(
                {
                    "horizon": int(h_idx),
                    "model": str(row["model"]),
                    "model_hash": str(row.get("model_hash", "")),
                    "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                    "dataset_signature": str(row.get("dataset_signature", "")),
                    "mape": float(y_mape[h_idx - 1]),
                    "mae": float(y_mae[h_idx - 1]),
                    "rmse": float(y_rmse[h_idx - 1]),
                    "smape": float(y_smape[h_idx - 1]),
                    "wape": float(y_wape[h_idx - 1]),
                }
            )
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    def _plot(metric: str, out_png: Path) -> None:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for idx, row in enumerate(_ordered_rows_for_lines(rows_use)):
            y = np.asarray(row[f"horizon_surface_{metric}"], dtype=float)
            x = np.arange(1, y.shape[0] + 1)
            style = _line_style_for_model(str(row["model"]), color_idx=idx)
            ax.plot(x, y, label=row["model"], **style)
        ylab = "Surface MAPE (%)" if metric == "mape" else ("Surface RMSE" if metric == "rmse" else "Surface MAE")
        ax.set_title(f"{metric.upper()} vs Forecast Horizon (Validation)")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

    _plot("mape", out_png_mape)
    _plot("mae", out_png_mae)
    _plot("rmse", out_png_rmse)

    if out_png_all_mape is not None:
        fig_all, ax_all = plt.subplots(figsize=(10.5, 6.0))
        for idx, row in enumerate(_ordered_rows_for_lines(rows_sorted)):
            y = np.asarray(row["horizon_surface_mape"], dtype=float)
            x = np.arange(1, y.shape[0] + 1)
            style = _line_style_for_model(str(row["model"]), color_idx=idx)
            ax_all.plot(x, y, label=row["model"], **style)
        ax_all.set_title("MAPE vs Forecast Horizon (All Models, Validation)")
        ax_all.set_xlabel("Horizon")
        ax_all.set_ylabel("Surface MAPE (%)")
        ax_all.grid(alpha=0.25)
        ax_all.legend(ncol=2, fontsize=7)
        fig_all.tight_layout()
        fig_all.savefig(out_png_all_mape, dpi=180)
        plt.close(fig_all)


def _plot_best_model_train_val_mape(
    *,
    eval_rows: list[dict[str, Any]],
    metrics_df: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
) -> None:
    if metrics_df.empty:
        return
    best_row: dict[str, Any] | None = None
    for _, mrow in metrics_df.iterrows():
        m = str(mrow.get("model", ""))
        mh = str(mrow.get("model_hash", ""))
        cand = next(
            (
                r
                for r in eval_rows
                if str(r.get("model", "")) == m and str(r.get("model_hash", "")) == mh
            ),
            None,
        )
        if cand is None:
            continue
        h_train = cand.get("horizon_surface_mape_train")
        h_val = cand.get("horizon_surface_mape_val")
        if h_train is None or h_val is None:
            continue
        best_row = cand
        break
    if best_row is None:
        warnings.warn(
            "Best model train/validation horizon MAPE unavailable; skipping best-model train-vs-validation plot.",
            RuntimeWarning,
        )
        return

    h_train_arr = np.asarray(best_row["horizon_surface_mape_train"], dtype=float)
    h_val_arr = np.asarray(best_row["horizon_surface_mape_val"], dtype=float)
    if h_train_arr.size == 0 or h_val_arr.size == 0:
        return
    x = np.arange(1, int(min(h_train_arr.size, h_val_arr.size)) + 1, dtype=int)
    h_train_arr = h_train_arr[: x.size]
    h_val_arr = h_val_arr[: x.size]

    model_name = str(best_row.get("model", "best_model"))
    model_hash = str(best_row.get("model_hash", ""))[:12]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, h_val_arr, label="validation", lw=2.6, marker="s")
    ax.set_title(f"Best Model Validation MAPE vs Horizon ({model_name}[{model_hash}])")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Surface MAPE (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    pd.DataFrame(
        {
            "horizon": x.astype(int),
            "mape_train": h_train_arr.astype(float),
            "mape_validation": h_val_arr.astype(float),
            "model": model_name,
            "model_hash": str(best_row.get("model_hash", "")),
        }
    ).to_csv(out_csv, index=False)


def _missingness_vec(missing_seq_norm: np.ndarray) -> np.ndarray:
    last = missing_seq_norm[:, -1, :]
    mean = missing_seq_norm.mean(axis=1)
    std = missing_seq_norm.std(axis=1)
    return np.concatenate([last, mean, std], axis=1)


def _predict_seasonal_naive(
    x_seq_norm: np.ndarray,
    horizon: int,
    d_factors: int,
    k: int = 13,
) -> np.ndarray:
    n, l, _ = x_seq_norm.shape
    if l <= 0:
        raise ValueError(f"Invalid lookback length for seasonal naive: {l}")
    out = np.zeros((n, int(horizon), int(d_factors)), dtype=float)
    for h_idx in range(int(horizon)):
        step = int(h_idx + 1)
        lag = int(k - step)
        if lag <= 0:
            out[:, h_idx, :] = x_seq_norm[:, -1, : int(d_factors)]
        elif lag < l:
            out[:, h_idx, :] = x_seq_norm[:, -(lag + 1), : int(d_factors)]
        else:
            out[:, h_idx, :] = x_seq_norm[:, -1, : int(d_factors)]
    return out


def _train_eval_photonic_config(
    *,
    kind: str,
    seed: int,
    model_kwargs: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_true_test_norm: np.ndarray,
    bundle: Any,
    y_true_val_surface: np.ndarray,
    y_true_test_surface: np.ndarray | None,
    mape_eps: float,
    residual_penalty_weight: float,
    sweep_objective: str,
) -> dict[str, Any]:
    train_kwargs = {k: v for k, v in model_kwargs.items() if k not in {"seed", "kind"}}
    model = train_classical_forecaster(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        kind=kind,  # type: ignore[arg-type]
        seed=int(seed),
        **train_kwargs,
    )
    y_pred_val = model.predict(x_val)
    y_pred_test = model.predict(x_test)
    y_pred_val_surface = _to_surface_space(bundle, y_pred_val)
    y_pred_test_surface = _to_surface_space(bundle, y_pred_test)
    if hasattr(model, "_baseline_from_x"):
        y_base_val = np.asarray(model._baseline_from_x(x_val), dtype=float)  # type: ignore[attr-defined]
        y_base_test = np.asarray(model._baseline_from_x(x_test), dtype=float)  # type: ignore[attr-defined]
    else:
        y_base_val = np.repeat(x_val[:, -1, : y_val.shape[2]][:, None, :], y_val.shape[1], axis=1)
        y_base_test = np.repeat(x_test[:, -1, : y_true_test_norm.shape[2]][:, None, :], y_true_test_norm.shape[1], axis=1)
    y_base_val_surface = _to_surface_space(bundle, y_base_val)
    y_base_test_surface = _to_surface_space(bundle, y_base_test)
    collapse_curve = np.mean(np.abs(y_pred_test_surface - y_base_test_surface), axis=(0, 2))
    collapse_mean = float(np.mean(collapse_curve))
    val_factor_mape = _compute_mape(y_val, y_pred_val, eps=mape_eps)
    val_factor_mae = float(np.mean(np.abs(y_val - y_pred_val)))
    val_surface_metrics = _compute_metric_block(y_true_val_surface, y_pred_val_surface, eps=mape_eps)
    val_mape = float(val_surface_metrics["mape"])
    val_mae = float(val_surface_metrics["mae"])
    val_delta_persist = float(np.mean(np.abs(y_pred_val_surface - y_base_val_surface)))
    pred_res_val = y_pred_val_surface - y_base_val_surface
    true_res_val = y_true_val_surface - y_base_val_surface
    std_pred = float(np.std(pred_res_val))
    std_true = float(np.std(true_res_val))
    residual_std_ratio = float(std_pred / (std_true + 1e-12))
    residual_penalty = float(max(0.0, residual_std_ratio - 1.2) * float(max(0.0, residual_penalty_weight)))
    objective_name = _sanitize_objective_name(str(sweep_objective))
    if objective_name == "surface_mape":
        objective_raw = float(val_surface_metrics["mape"])
    elif objective_name == "surface_wape":
        objective_raw = float(val_surface_metrics["wape"])
    elif objective_name == "surface_mae":
        objective_raw = float(val_surface_metrics["mae"])
    elif objective_name == "surface_rmse":
        objective_raw = float(val_surface_metrics["rmse"])
    else:
        objective_raw = float(val_surface_metrics["mape"])
    score = float(objective_raw + residual_penalty)
    config_for_hash = {
        **{k: v for k, v in model_kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None},
        "kind": str(kind),
        "seed": int(seed),
    }
    model_hash = stable_hash_config(config_for_hash, name=f"{kind}:config")
    fstats = getattr(model, "feature_stats", None)
    feature_median_std = float(np.median(np.asarray(fstats.get("feat_std_train", []), dtype=float))) if isinstance(fstats, dict) else 0.0
    surface_nonfinite = int(np.size(y_pred_val_surface) - np.isfinite(y_pred_val_surface).sum())
    surface_nonpositive = int(np.sum(y_pred_val_surface <= 0.0))
    if surface_nonfinite > 0 or surface_nonpositive > 0:
        raise RuntimeError(
            f"{kind}: rejected config due to invalid surface outputs "
            f"(nonfinite={surface_nonfinite}, nonpositive={surface_nonpositive})."
        )
    if np.isfinite(feature_median_std) and feature_median_std <= 1e-10:
        raise RuntimeError(
            f"{kind}: rejected config due to feature collapse (feature_std_median={feature_median_std:.3e})."
        )
    qrc_mode_used = str(getattr(model, "qrc_mode", ""))
    qrc_target_used = str(getattr(model, "qrc_target", model_kwargs.get("qrc_target", "")))
    qrc_baseline_used = str(getattr(model, "qrc_baseline", model_kwargs.get("qrc_baseline", "")))
    target_residual_mae_val = np.nan
    target_residual_mae_test = np.nan
    if hasattr(model, "_target_from_level") and hasattr(model, "_baseline_from_x"):
        try:
            base_val_for_target = np.asarray(model._baseline_from_x(x_val), dtype=float)  # type: ignore[attr-defined]
            base_test_for_target = np.asarray(model._baseline_from_x(x_test), dtype=float)  # type: ignore[attr-defined]
            pred_target_val = np.asarray(model._target_from_level(y_pred_val, base_val_for_target, x_val), dtype=float)  # type: ignore[attr-defined]
            true_target_val = np.asarray(model._target_from_level(y_val, base_val_for_target, x_val), dtype=float)  # type: ignore[attr-defined]
            pred_target_test = np.asarray(model._target_from_level(y_pred_test, base_test_for_target, x_test), dtype=float)  # type: ignore[attr-defined]
            true_target_test = np.asarray(model._target_from_level(y_true_test_norm, base_test_for_target, x_test), dtype=float)  # type: ignore[attr-defined]
            target_residual_mae_val = float(np.mean(np.abs(pred_target_val - true_target_val)))
            target_residual_mae_test = float(np.mean(np.abs(pred_target_test - true_target_test)))
        except Exception:
            target_residual_mae_val = np.nan
            target_residual_mae_test = np.nan
    gate_val = getattr(model, "last_gate_", None)
    if gate_val is not None and np.asarray(gate_val).shape == y_pred_test.shape:
        gate_test = np.asarray(gate_val, dtype=float)
    else:
        gate_test = None
    feedback_sanity = getattr(model, "feedback_sanity", None)
    y_true_test_surface_eff = (
        np.asarray(y_true_test_surface, dtype=float)
        if y_true_test_surface is not None
        else _to_surface_space(bundle, y_true_test_norm)
    )
    return {
        "model_obj": model,
        "model_hash": model_hash,
        "model_kwargs": dict(model_kwargs),
        "val_mape": float(val_mape),
        "val_mae": float(val_mae),
        "val_factor_mape": float(val_factor_mape),
        "val_factor_mae": float(val_factor_mae),
        "val_surface_mape": float(val_surface_metrics["mape"]),
        "val_surface_mae": float(val_surface_metrics["mae"]),
        "val_surface_rmse": float(val_surface_metrics["rmse"]),
        "val_surface_wape": float(val_surface_metrics["wape"]),
        "val_delta_persist": float(val_delta_persist),
        "residual_std_ratio": residual_std_ratio,
        "residual_penalty": residual_penalty,
        "objective_name": objective_name,
        "objective_value": float(objective_raw),
        "score": score,
        "collapse_mean": collapse_mean,
        "collapsed": bool(collapse_mean < 1e-6),
        "feature_median_std": feature_median_std,
        "feature_stats": fstats,
        "y_pred_test": y_pred_test,
        "horizon_mape_test": _horizon_metric(y_true_test_surface_eff, y_pred_test_surface, metric="mape", eps=mape_eps),
        "collapse_curve": np.asarray(collapse_curve, dtype=float),
        "qrc_mode_used": qrc_mode_used,
        "qrc_target_used": qrc_target_used,
        "qrc_baseline_used": qrc_baseline_used,
        "target_residual_mae_val": float(target_residual_mae_val) if np.isfinite(target_residual_mae_val) else np.nan,
        "target_residual_mae_test": float(target_residual_mae_test) if np.isfinite(target_residual_mae_test) else np.nan,
        "qrc_gate_test": gate_test,
        "feedback_sanity": feedback_sanity,
    }


def _run_photonic_coordinate_sweep(
    *,
    kind: str,
    seed: int,
    base_kwargs: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_true_test_norm: np.ndarray,
    bundle: Any,
    y_true_val_surface: np.ndarray,
    y_true_test_surface: np.ndarray,
    mape_eps: float,
    sweep_subset: int,
    topk: int,
    residual_penalty_weight: float,
    sweep_objective: str = "surface_mape",
    sweep_modes: bool = False,
    sweep_gain: bool = False,
    sweep_input_scale: bool = False,
    sweep_ridge: bool = False,
) -> dict[str, Any]:
    from .models.photonic_qrc import PhotonicQRCParams, compute_photonic_memory_proxy as compute_photonic_qrc_memory_proxy
    from .models.photonic_memory import (
        PhotonicMemoryParams,
        compute_photonic_memory_proxy as compute_photonic_memory_model_proxy,
    )

    objective = _sanitize_objective_name(str(sweep_objective))
    kind_norm = str(kind).strip().lower()
    is_memory_kind = bool(kind_norm in {"photonic_memory", "photonic_memory_no_feedback"})
    is_no_feedback_kind = bool(kind_norm in {"photonic_qrc_no_feedback", "photonic_memory_no_feedback"})
    n_train_sub = int(max(16, min(int(sweep_subset), x_train.shape[0])))
    n_val_sub = int(max(16, min(max(32, int(sweep_subset // 3)), x_val.shape[0])))
    x_train_sub = x_train[:n_train_sub]
    y_train_sub = y_train[:n_train_sub]
    x_val_sub = x_val[:n_val_sub]
    y_val_sub = y_val[:n_val_sub]

    def _materialize_cfg(cfg_like: dict[str, Any]) -> dict[str, Any]:
        raw = dict(cfg_like)
        cfg_obj = QRCConfig.from_dict(raw).validate()
        cfg = cfg_obj.to_dict()
        for passthrough in [
            "persist_qrc_alpha",
            "persist_qrc_rclip",
            "persist_qrc_drive_scale",
            "qrc_config_hash",
            "dataset_signature",
        ]:
            if passthrough in raw:
                cfg[passthrough] = raw[passthrough]
        cfg.pop("config_hash", None)
        # Keep legacy aliases expected by existing model kwargs.
        cfg["pqrc_M"] = int(cfg["pqrc_modes"])
        cfg["pqrc_Nph"] = int(cfg["pqrc_nph"])
        cfg["tau"] = float(cfg["qrc_gate_tau"])
        cfg["qrc_residual_penalty"] = float(cfg["qrc_residvar_penalty"])
        return cfg

    def _validate_numeric_hparams(cfg: dict[str, Any]) -> None:
        num_keys = [
            "pqrc_modes",
            "pqrc_gain",
            "pqrc_input_scale",
            "pqrc_ridge",
            "pqrc_pseudocount",
            "pqrc_shots",
            "pqrc_budget",
            "qrc_gate_tau",
        ]
        for k in num_keys:
            v = cfg.get(k, np.nan)
            try:
                fv = float(v)
            except Exception as exc:
                raise RuntimeError(f"Invalid sweep hyperparameter {k}={v!r}: {exc}") from exc
            if not np.isfinite(fv):
                raise RuntimeError(f"Invalid sweep hyperparameter {k}={v!r}: non-finite.")

    current = _materialize_cfg(dict(base_kwargs))
    _validate_numeric_hparams(current)
    if is_no_feedback_kind:
        current["pqrc_gain"] = 0.0
    history: list[dict[str, Any]] = []
    memory_proxy_records: list[dict[str, Any]] = []

    ridge_vals = [float(current["pqrc_ridge"])] if not bool(sweep_ridge) else [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    gain_vals = [float(current["pqrc_gain"])] if not bool(sweep_gain) else [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    if is_no_feedback_kind:
        gain_vals = [0.0]
    input_scale_vals = [float(current["pqrc_input_scale"])] if not bool(sweep_input_scale) else [0.5, 1.0, 2.0]
    modes_vals = [int(current["pqrc_modes"])] if not bool(sweep_modes) else [8, 12, 16, 20]

    grid: list[tuple[str, list[Any]]] = [
        ("pqrc_ridge", ridge_vals),
        ("pqrc_gain", gain_vals),
        ("pqrc_shots", [64, 128, 256]),
        ("pqrc_input_scale", input_scale_vals),
        ("qrc_gate_tau", [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]),
        ("pqrc_modes", modes_vals),
        ("pqrc_higher_order", [0, 1, 2]),
    ]
    unique_cfgs: dict[str, dict[str, Any]] = {}

    def _cfg_hash(cfg: dict[str, Any]) -> str:
        return stable_hash_config({"kind": kind, "seed": int(seed), **cfg}, name=f"{kind}:cfg")

    def _params_from_cfg(cfg: dict[str, Any]) -> Any:
        target_mode = str(cfg.get("qrc_target", "delta")).strip().lower()
        tau_default = 0.5 if target_mode == "norm_resid" else 0.05
        tau_val = cfg.get("qrc_gate_tau", cfg.get("tau", tau_default))
        common_kwargs = {
            "modes": int(cfg.get("pqrc_modes", cfg.get("pqrc_M", 12))),
            "n_photons": int(cfg.get("pqrc_Nph", 2)),
            "budget": int(cfg.get("pqrc_budget", 32)),
            "gain": float(cfg.get("pqrc_gain", 0.5)),
            "feature": str(cfg.get("pqrc_feature", "coincidence")),  # type: ignore[dict-item]
            "shots": int(cfg.get("pqrc_shots", 32)),
            "ridge_alpha": float(cfg.get("pqrc_ridge", 1e-2)),
            "qrc_mode": str(cfg.get("qrc_mode", "auto")),
            "qrc_target": str(target_mode),
            "qrc_baseline": str(cfg.get("qrc_baseline", "persistence")),
            "target_transform": str(cfg.get("target_transform", "log")),
            "y_floor_mode": str(cfg.get("y_floor_mode", "train_p001")),
            "y_floor_value": cfg.get("y_floor_value", None),
            "tau": float(tau_val),
            "resid_clip": cfg.get("qrc_resid_clip", None),
            "residvar_penalty": float(cfg.get("qrc_residvar_penalty", residual_penalty_weight)),
            "feat_norm": str(cfg.get("qrc_feat_norm", "standard")),
            "in_pca": int(cfg.get("pqrc_in_pca", 4)),
            "input_scale": float(cfg.get("pqrc_input_scale", 0.75)),
            "min_residual_scale": float(cfg.get("pqrc_min_residual_scale", 0.05)),
            "higher_order": int(cfg.get("pqrc_higher_order", 1)),
            "budget_alloc": str(cfg.get("pqrc_budget_alloc", "uniform")),
            "seed": int(seed),
            "feedback_enabled": bool(not is_no_feedback_kind),
        }
        if is_memory_kind:
            return PhotonicMemoryParams(**common_kwargs)  # type: ignore[arg-type]
        common_kwargs["pseudocount"] = float(cfg.get("pqrc_pseudocount", 0.5))
        return PhotonicQRCParams(**common_kwargs)  # type: ignore[arg-type]

    def _memory_proxy_eval(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
        p = _params_from_cfg(cfg)
        if is_no_feedback_kind:
            p.gain = 0.0
            p.feedback_enabled = False
        if is_memory_kind:
            mp = compute_photonic_memory_model_proxy(
                x_train_sub,
                params=p,
                max_k=10,
                ridge_alpha=float(max(1e-8, cfg.get("pqrc_ridge", 1.0))),
                max_samples=min(256, n_train_sub),
            )
        else:
            mp = compute_photonic_qrc_memory_proxy(
                x_train_sub,
                params=p,
                max_k=10,
                ridge_alpha=float(max(1e-8, cfg.get("pqrc_ridge", 1.0))),
                max_samples=min(256, n_train_sub),
            )
        rec = {
            "phase": str(phase),
            "model_hash": _cfg_hash(cfg),
            "model_kwargs": dict(cfg),
            "mc_score": float(mp.get("mc_score", 0.0)),
            "stability_penalty": float(mp.get("stability_penalty", 0.0)),
            "mc_score_adj": float(mp.get("mc_score_adj", 0.0)),
            "state_autocorr_lag1": float(mp.get("state_autocorr_lag1", np.nan)),
            "feature_std_median": float(mp.get("feature_std_median", np.nan)),
            "r2_by_k": list(mp.get("r2_by_k", [])),
            "failed": False,
        }
        for i_k, v_k in enumerate(rec["r2_by_k"], start=1):
            rec[f"r2_k{i_k}"] = float(v_k)
        return rec

    for pname, vals in grid:
        if is_no_feedback_kind and pname == "pqrc_gain":
            current["pqrc_gain"] = 0.0
            continue
        local_best_mc: dict[str, Any] | None = None
        local_best_map: dict[str, Any] | None = None
        for v in vals:
            cfg = dict(current)
            cfg[pname] = v
            if is_no_feedback_kind:
                cfg["pqrc_gain"] = 0.0
            cfg = _materialize_cfg(cfg)
            _validate_numeric_hparams(cfg)
            h_cfg = _cfg_hash(cfg)
            unique_cfgs[h_cfg] = cfg
            try:
                if objective == "mc_then_mape":
                    rec_mc = _memory_proxy_eval(cfg, phase=f"sweep_{pname}")
                    rec_mc["objective"] = str(objective)
                    rec_mc["objective_components"] = json.dumps(
                        {
                            "mc_score_adj": float(rec_mc.get("mc_score_adj", np.nan)),
                            "mc_score": float(rec_mc.get("mc_score", np.nan)),
                            "stability_penalty": float(rec_mc.get("stability_penalty", np.nan)),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    memory_proxy_records.append(rec_mc)
                    if (local_best_mc is None) or (
                        (float(rec_mc["mc_score_adj"]), float(rec_mc["mc_score"]))
                        > (float(local_best_mc["mc_score_adj"]), float(local_best_mc["mc_score"]))
                    ):
                        local_best_mc = rec_mc
                else:
                    rec = _train_eval_photonic_config(
                        kind=kind,
                        seed=int(seed),
                        model_kwargs=cfg,
                        x_train=x_train_sub,
                        y_train=y_train_sub,
                        x_val=x_val_sub,
                        y_val=y_val_sub,
                        x_test=x_test,
                        y_true_test_norm=y_true_test_norm,
                        bundle=bundle,
                        y_true_val_surface=y_true_val_surface[: x_val_sub.shape[0]],
                        y_true_test_surface=y_true_test_surface,
                        mape_eps=mape_eps,
                        residual_penalty_weight=float(residual_penalty_weight),
                        sweep_objective=objective,
                    )
                    rec["phase"] = f"sweep_{pname}"
                    rec["objective"] = str(objective)
                    rec["objective_components"] = json.dumps(
                        {
                            "objective_name": str(rec.get("objective_name", objective)),
                            "objective_value": float(rec.get("objective_value", np.nan)),
                            "val_surface_mape": float(rec.get("val_surface_mape", np.nan)),
                            "val_surface_wape": float(rec.get("val_surface_wape", np.nan)),
                            "val_surface_mae": float(rec.get("val_surface_mae", np.nan)),
                            "val_surface_rmse": float(rec.get("val_surface_rmse", np.nan)),
                            "residual_penalty": float(rec.get("residual_penalty", np.nan)),
                            "score": float(rec.get("score", np.nan)),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    history.append(rec)
                    if (local_best_map is None) or (
                        (bool(rec["collapsed"]), float(rec.get("score", np.inf)), float(rec.get("objective_value", np.inf)), float(rec.get("val_mape", np.inf)))
                        < (
                            bool(local_best_map["collapsed"]),
                            float(local_best_map.get("score", np.inf)),
                            float(local_best_map.get("objective_value", np.inf)),
                            float(local_best_map.get("val_mape", np.inf)),
                        )
                    ):
                        local_best_map = rec
            except Exception as exc:
                rec_fail = {
                    "phase": f"sweep_{pname}",
                    "model_hash": h_cfg,
                    "model_kwargs": cfg,
                    "failed": True,
                    "error": str(exc),
                    "objective": str(objective),
                }
                if objective == "mc_then_mape":
                    memory_proxy_records.append(
                        {
                            **rec_fail,
                            "mc_score": float("-inf"),
                            "stability_penalty": float("inf"),
                            "mc_score_adj": float("-inf"),
                            "r2_by_k": [],
                            "objective_components": json.dumps({"error": str(exc)}, sort_keys=True, separators=(",", ":")),
                        }
                    )
                else:
                    history.append(
                        {
                            **rec_fail,
                            "collapsed": True,
                            "val_mape": float("inf"),
                            "val_mae": float("inf"),
                            "objective_name": str(objective),
                            "objective_value": float("inf"),
                            "score": float("inf"),
                            "objective_components": json.dumps({"error": str(exc)}, sort_keys=True, separators=(",", ":")),
                        }
                    )
        if objective == "mc_then_mape":
            if local_best_mc is not None:
                current = dict(local_best_mc["model_kwargs"])
        elif local_best_map is not None:
            current = dict(local_best_map["model_kwargs"])

    top_cfgs: list[dict[str, Any]] = []
    if objective == "mc_then_mape":
        memory_success = [r for r in memory_proxy_records if not bool(r.get("failed", False))]
        memory_sorted = sorted(memory_success, key=lambda r: (-float(r.get("mc_score_adj", -np.inf)), -float(r.get("mc_score", -np.inf))))
        seen: set[str] = set()
        for rec in memory_sorted:
            h_cfg = str(rec.get("model_hash", ""))
            if h_cfg in seen:
                continue
            seen.add(h_cfg)
            top_cfgs.append(rec)
            if len(top_cfgs) >= 10:
                break
    else:
        success = [h for h in history if not bool(h.get("failed", False))]
        success_sorted = sorted(
            success,
            key=lambda r: (
                bool(r.get("collapsed", False)),
                float(r.get("score", np.inf)),
                float(r.get("objective_value", np.inf)),
                float(r.get("val_mape", np.inf)),
            ),
        )
        seen_hashes: set[str] = set()
        for rec in success_sorted:
            h = str(rec["model_hash"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            top_cfgs.append(rec)
            if len(top_cfgs) >= int(max(1, topk)):
                break
    if not top_cfgs:
        raise RuntimeError(f"{kind}: photonic sweep produced no successful configuration.")

    top_full: list[dict[str, Any]] = []
    for rank, rec in enumerate(top_cfgs, start=1):
        cfg = dict(rec.get("model_kwargs", {}))
        full = _train_eval_photonic_config(
            kind=kind,
            seed=int(seed),
            model_kwargs=cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_true_test_norm=y_true_test_norm,
            bundle=bundle,
            y_true_val_surface=y_true_val_surface,
            y_true_test_surface=y_true_test_surface,
            mape_eps=mape_eps,
            residual_penalty_weight=float(residual_penalty_weight),
            sweep_objective=objective,
        )
        full["sweep_rank"] = int(rank)
        if objective == "mc_then_mape":
            full["mc_score"] = float(rec.get("mc_score", np.nan))
            full["mc_score_adj"] = float(rec.get("mc_score_adj", np.nan))
            full["r2_by_k"] = list(rec.get("r2_by_k", []))
        top_full.append(full)
    top_full = sorted(
        top_full,
        key=lambda r: (
            bool(r["collapsed"]),
            float(r.get("score", np.inf)),
            float(r.get("objective_value", np.inf)),
            float(r.get("val_mape", np.inf)),
        ),
    )
    if objective == "mc_then_mape" and not history:
        history = [
            {
                "phase": "mc_then_mape_eval",
                "model_hash": str(rec.get("model_hash", "")),
                "model_kwargs": dict(rec.get("model_kwargs", {})),
                "val_mape": float(rec.get("val_mape", np.inf)),
                "val_mae": float(rec.get("val_mae", np.inf)),
                "score": float(rec.get("score", np.inf)),
                "collapsed": bool(rec.get("collapsed", False)),
                "objective_name": str(rec.get("objective_name", objective)),
                "objective_value": float(rec.get("objective_value", np.inf)),
                "val_surface_mape": float(rec.get("val_surface_mape", np.nan)),
                "val_surface_wape": float(rec.get("val_surface_wape", np.nan)),
                "val_surface_mae": float(rec.get("val_surface_mae", np.nan)),
                "val_surface_rmse": float(rec.get("val_surface_rmse", np.nan)),
                "residual_std_ratio": float(rec.get("residual_std_ratio", np.nan)),
                "residual_penalty": float(rec.get("residual_penalty", np.nan)),
                "qrc_mode_used": str(rec.get("qrc_mode_used", "")),
                "objective": str(objective),
                "objective_components": json.dumps(
                    {
                        "objective_name": str(rec.get("objective_name", objective)),
                        "objective_value": float(rec.get("objective_value", np.nan)),
                        "val_surface_mape": float(rec.get("val_surface_mape", np.nan)),
                        "val_surface_wape": float(rec.get("val_surface_wape", np.nan)),
                        "val_surface_mae": float(rec.get("val_surface_mae", np.nan)),
                        "val_surface_rmse": float(rec.get("val_surface_rmse", np.nan)),
                        "residual_penalty": float(rec.get("residual_penalty", np.nan)),
                        "score": float(rec.get("score", np.nan)),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            for rec in top_full
        ]
    best = top_full[0]
    if bool(best["collapsed"]):
        warnings.warn(
            f"{kind}: all swept configs collapsed to persistence; returning best-available collapsed config.",
            RuntimeWarning,
        )
    return {
        "best": best,
        "top_full": top_full,
        "history": history,
        "memory_proxy_records": memory_proxy_records,
        "config_hash": str(best.get("model_hash", "")),
        "best_config": dict(best.get("model_kwargs", {})),
    }


def _build_quantum_comparison(
    bundle: Any,
    classical_name: str,
    y_base_train: np.ndarray,
    y_base_val: np.ndarray,
    y_base_test: np.ndarray,
    backend: str,
    quantum_mode: str,
    shots: int,
    qrc_stride: int,
    qrc_feature_dim: int,
    train_qeval_budget: int,
    batch_size: int,
    seed: int,
    quantum_feature_mode: str,
    q_dim: int,
    q_angle_scale: float,
    q_seed: int,
    min_qfeat_std: float,
    w_min: float,
    mape_eps: float,
    train_fit_slice: slice | None = None,
    train_tune_slice: slice | None = None,
) -> dict[str, Any]:
    if not _HAS_QUANTUM:
        raise RuntimeError(_QUANTUM_IMPORT_ERROR)
    assert QuantumAnnotatorResidual is not None

    cand_train_full = np.concatenate([bundle.x_train_norm, bundle.missing_train_norm], axis=2)
    cand_val_eval = np.concatenate([bundle.x_val_norm, bundle.missing_val_norm], axis=2)
    cand_test = np.concatenate([bundle.x_test_norm, bundle.missing_test_norm], axis=2)
    if train_fit_slice is None or train_tune_slice is None:
        cand_train = cand_train_full
        cand_val_tune = cand_val_eval
        y_true_train_fit = np.asarray(bundle.y_train_norm, dtype=float)
        y_true_tune = np.asarray(bundle.y_val_norm, dtype=float)
        y_base_train_fit = np.asarray(y_base_train, dtype=float)
        y_base_tune = np.asarray(y_base_val, dtype=float)
    else:
        cand_train = cand_train_full[train_fit_slice]
        cand_val_tune = cand_train_full[train_tune_slice]
        y_true_train_fit = np.asarray(bundle.y_train_norm[train_fit_slice], dtype=float)
        y_true_tune = np.asarray(bundle.y_train_norm[train_tune_slice], dtype=float)
        y_base_train_fit = np.asarray(y_base_train[train_fit_slice], dtype=float)
        y_base_tune = np.asarray(y_base_train[train_tune_slice], dtype=float)

    backend_cap = 20 if str(backend).lower() == "sim" else 24
    merlin_cap = 20
    mode_cap = min(backend_cap, merlin_cap)
    total_dim = int(cand_train.shape[-1])
    if total_dim > mode_cap:
        residual_target_train = (y_true_train_fit - y_base_train_fit).reshape(y_base_train_fit.shape[0], -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            selected_idx = wrapper_forward_selection(
                features=cand_train[:, -1, :],
                target=residual_target_train,
                max_k=mode_cap,
                score_fn=lambda f, t: ridge_validation_mae_score(f, t, val_fraction=0.2, seed=int(seed)),
            )
    else:
        selected_idx = list(range(total_dim))

    if not selected_idx:
        raise RuntimeError("Quantum feature selection returned empty set.")
    validate_constraints(input_dim=len(selected_idx), backend=backend)

    q_train = cand_train[:, :, selected_idx]
    q_val_tune = cand_val_tune[:, :, selected_idx]
    q_val_eval = cand_val_eval[:, :, selected_idx]
    q_test = cand_test[:, :, selected_idx]

    q_model = QuantumAnnotatorResidual(
        input_dim=len(selected_idx),
        mode=quantum_mode,
        shots=shots,
        qrc_stride=qrc_stride,
        qrc_feature_dim=qrc_feature_dim,
        train_qeval_budget=train_qeval_budget,
        batch_size=batch_size,
        seed=seed,
        quantum_feature_mode=quantum_feature_mode,
        q_dim=q_dim,
        q_angle_scale=q_angle_scale,
        q_seed=q_seed,
        min_qfeat_std=min_qfeat_std,
    )
    q_model.fit(
        x_train=q_train,
        y_true_train=y_true_train_fit,
        y_classical_pred_train=y_base_train_fit,
        x_val=q_val_tune,
        y_true_val=y_true_tune,
        y_classical_pred_val=y_base_tune,
    )
    q_res_train = q_model.predict_residual(cand_train_full[:, :, selected_idx])
    q_res_val = q_model.predict_residual(q_val_eval)
    q_res_test = q_model.predict_residual(q_test)
    y_pred_train = y_base_train + q_res_train
    y_pred_val = y_base_val + q_res_val
    y_pred_test = y_base_test + q_res_test
    floor = float(getattr(bundle, "surface_floor", 0.0))
    y_true_train_surface = np.clip(bundle.y_train_surface_scaled * bundle.price_scale, floor, bundle.clip_cap)
    y_true_val_surface = np.clip(bundle.y_val_surface_scaled * bundle.price_scale, floor, bundle.clip_cap)
    split_extra = _split_surface_mape_extra(
        y_true_train_surface=y_true_train_surface,
        y_pred_train_surface=_to_surface_space(bundle, y_pred_train),
        y_true_val_surface=y_true_val_surface,
        y_pred_val_surface=_to_surface_space(bundle, y_pred_val),
        eps=float(mape_eps),
    )
    diag = q_model.diagnostics() if hasattr(q_model, "diagnostics") else {}
    q_features = diag.get("q_features_infer") if isinstance(diag, dict) else None
    if q_features is None:
        raise RuntimeError(
            "Quantum model produced degenerate q_features; refusing to continue. "
            "Check encoding scale / shots / featuremap."
        )
    q_features_arr = np.asarray(q_features, dtype=float)
    q_std = float(np.std(np.nan_to_num(q_features_arr, nan=0.0, posinf=0.0, neginf=0.0)))
    if q_std <= float(max(1e-12, min_qfeat_std)):
        raise RuntimeError(
            "Quantum model produced degenerate q_features; refusing to continue. "
            f"Check encoding scale / shots / featuremap. std={q_std:.3e}"
        )

    return {
        "name": f"{classical_name}+{quantum_mode}",
        "pred_test_norm": y_pred_test,
        "pred_val_norm": y_pred_val,
        "w_star": 1.0,
        "w_min": float(w_min),
        "selected_dim": int(len(selected_idx)),
        "quantum_mode": quantum_mode,
        "q_features": q_features_arr,
        "kernel_gram": diag.get("kernel_gram") if isinstance(diag, dict) else None,
        "q_feature_summary": diag.get("q_feature_summary") if isinstance(diag, dict) else None,
        "config_fingerprint": {
            "q_dim": int(q_dim),
            "shots": int(shots),
            "angle_scale": float(q_angle_scale),
            "featuremap_seed": int(q_seed),
        },
        **split_extra,
        **q_model.budget_dict(),
    }


def _build_ported_comparisons(
    bundle: Any,
    backend: str,
    shots: int,
    qrc_stride: int,
    qrc_feature_dim: int,
    train_qeval_budget: int,
    batch_size: int,
    seed: int,
    train_fit_slice: slice | None = None,
    train_tune_slice: slice | None = None,
) -> list[dict[str, Any]]:
    if not _HAS_PORTED:
        raise RuntimeError(_PORTED_IMPORT_ERROR)
    if run_ported_model is None:
        raise RuntimeError("Ported model functions are unavailable.")

    h = int(bundle.y_train_norm.shape[1])
    x_train_factor_full = bundle.x_train_norm[:, :, : bundle.factor_dim]
    x_eval_factor = bundle.x_val_norm[:, :, : bundle.factor_dim]
    if train_fit_slice is None or train_tune_slice is None:
        x_train_factor = x_train_factor_full
        y_train_factor = bundle.y_train_norm
        x_tune_factor = x_eval_factor
        y_tune_factor = bundle.y_val_norm
        m_train = _missingness_vec(bundle.missing_train_norm)
        m_tune = _missingness_vec(bundle.missing_val_norm)
    else:
        x_train_factor = x_train_factor_full[train_fit_slice]
        y_train_factor = bundle.y_train_norm[train_fit_slice]
        x_tune_factor = x_train_factor_full[train_tune_slice]
        y_tune_factor = bundle.y_train_norm[train_tune_slice]
        m_train_full = _missingness_vec(bundle.missing_train_norm)
        m_train = m_train_full[train_fit_slice]
        m_tune = m_train_full[train_tune_slice]
    m_eval = _missingness_vec(bundle.missing_val_norm)

    d_q = min(8, int(x_train_factor.shape[-1]))
    d_phi = int(max(8, qrc_feature_dim))

    out: list[dict[str, Any]] = []

    def _payload(name: str, mf: Any) -> dict[str, Any]:
        q_features = mf.q_features if hasattr(mf, "q_features") else None
        if str(name).startswith("ported_q"):
            if q_features is None:
                raise RuntimeError(
                    "Quantum model produced degenerate q_features; refusing to continue. "
                    "Check encoding scale / shots / featuremap."
                )
            q_std = float(np.std(np.asarray(q_features, dtype=float)))
            if q_std <= 1e-8:
                raise RuntimeError(
                    "Quantum model produced degenerate q_features; refusing to continue. "
                    f"Check encoding scale / shots / featuremap. model={name} std={q_std:.3e}"
                )
        return {
            "name": str(name),
            "pred_eval_norm": mf.z_pred_test,
            "w_star": float(mf.best_blend_w),
            "algo_id": str(getattr(mf, "algo_id", "")),
            "q_features": q_features,
            "q_feature_summary": getattr(mf, "q_feature_summary", None),
            "config_fingerprint": dict(getattr(mf, "config", {}) or {}),
            "extra": dict(getattr(mf, "extra", {}) or {}),
            **mf.budget.as_dict(),
        }

    ctrl_rff = run_ported_model(
        "ported_ctrl_rff",
        x_train=x_train_factor,
        y_train=y_train_factor,
        x_val=x_tune_factor,
        y_val=y_tune_factor,
        x_test=x_eval_factor,
        horizon=h,
        seed=seed + 11,
        d_q=d_q,
        d_phi=d_phi,
        use_mask_features=True,
        mask_train_vec=m_train,
        mask_val_vec=m_tune,
        mask_test_vec=m_eval,
    )
    out.append(_payload("ported_ctrl_rff", ctrl_rff))

    ctrl_lfm = run_ported_model(
        "ported_ctrl_learned_featuremap",
        x_train=x_train_factor,
        y_train=y_train_factor,
        x_val=x_tune_factor,
        y_val=y_tune_factor,
        x_test=x_eval_factor,
        horizon=h,
        seed=seed + 13,
        d_q=d_q,
        d_phi=d_phi,
        use_mask_features=True,
        mask_train_vec=m_train,
        mask_val_vec=m_tune,
        mask_test_vec=m_eval,
        batch_size=batch_size,
    )
    out.append(_payload("ported_ctrl_learned_featuremap", ctrl_lfm))

    ctrl_res = run_ported_model(
        "ported_ctrl_classical_reservoir",
        x_train=x_train_factor,
        y_train=y_train_factor,
        x_val=x_tune_factor,
        y_val=y_tune_factor,
        x_test=x_eval_factor,
        horizon=h,
        seed=seed + 17,
        d_phi=d_phi,
        qrc_stride=qrc_stride,
        qrc_pool="none",
        reservoir_alpha=0.7,
        reservoir_rho=0.9,
        reservoir_ridge_lambda=1.0,
    )
    out.append(_payload("ported_ctrl_classical_reservoir", ctrl_res))

    validate_constraints(input_dim=int(x_train_factor.shape[-1]), backend=backend)
    q_b = run_ported_model(
        "ported_q_bottleneck",
        x_train=x_train_factor,
        y_train=y_train_factor,
        x_val=x_tune_factor,
        y_val=y_tune_factor,
        x_test=x_eval_factor,
        horizon=h,
        shots=shots,
        train_qeval_budget=train_qeval_budget,
        batch_size=batch_size,
        seed=seed + 19,
        d_q=d_q,
        d_phi=d_phi,
        train_quantum=False,
        grad_qeval_multiplier=1.0,
        use_mask_features=True,
        mask_train_vec=m_train,
        mask_val_vec=m_tune,
        mask_test_vec=m_eval,
    )
    out.append(_payload("ported_q_bottleneck", q_b))

    q_r = run_ported_model(
        "ported_qrc",
        x_train=x_train_factor,
        y_train=y_train_factor,
        x_val=x_tune_factor,
        y_val=y_tune_factor,
        x_test=x_eval_factor,
        horizon=h,
        shots=shots,
        train_qeval_budget=train_qeval_budget,
        batch_size=batch_size,
        seed=seed + 23,
        d_phi=d_phi,
        train_quantum=False,
        grad_qeval_multiplier=1.0,
        qrc_stride=qrc_stride,
        qrc_pool="none",
        reservoir_alpha=0.7,
        reservoir_rho=0.9,
        reservoir_ridge_lambda=1.0,
    )
    out.append(_payload("ported_qrc", q_r))

    return out


def _write_hash_logs(
    eval_rows: list[dict[str, Any]],
    out_path: Path,
    seed: int,
    level: int,
    backend: str,
    mape_eps: float,
) -> dict[str, list[str]]:
    if out_path.exists():
        out_path.unlink()

    hash_to_models: dict[str, list[str]] = {}
    for row in eval_rows:
        model = str(row["model"])
        pred_surface = np.asarray(row["pred_test_surface"], dtype=float)
        pred_factor = np.asarray(row["pred_test_norm"], dtype=float)

        # Use stable shared hash names so duplicate predictions across models are detectable.
        h_surface = stable_hash_array(pred_surface, name="pred_surface")
        h_factor = stable_hash_array(pred_factor, name="pred_factor")
        hash_to_models.setdefault(h_surface, []).append(model)

        q_features = row.get("q_features")
        kernel_gram = row.get("kernel_gram")
        features_input = row.get("features_input")
        if _is_quantum_named(model):
            if q_features is None:
                raise RuntimeError(
                    f"Quantum model {model} is missing q_features; refusing to continue."
                )
            q_std = float(np.std(np.asarray(q_features, dtype=float)))
            if q_std <= 1e-8:
                raise RuntimeError(
                    "Quantum model produced degenerate q_features; refusing to continue. "
                    f"Check encoding scale / shots / featuremap. model={model} std={q_std:.3e}"
                )

        total_qevals = int(row.get("total_qevals", 0))
        total_shots = int(row.get("total_shots", 0))
        shots_per_eval = int(row.get("shots_per_eval", 0))
        record = {
            "model": model,
            "model_name": model,
            "model_hash": str(row.get("model_hash", "")),
            "config": {
                "seed": int(seed),
                "level": int(level),
                "backend": str(backend),
                "mape_eps": float(mape_eps),
                "qrc_config_hash": str(row.get("qrc_config_hash", "")),
                "dataset_signature": str(row.get("dataset_signature", "")),
                "algo_id": row.get("algo_id"),
                "w_min": row.get("w_min"),
                "config_fingerprint": row.get("config_fingerprint"),
            },
            "hashes": {
                "hash_pred_surface": h_surface,
                "hash_pred_factors": h_factor,
                "hash_features_input": None if features_input is None else stable_hash_array(np.asarray(features_input), name=f"{model}:features_input"),
                "hash_q_features": None if q_features is None else stable_hash_array(np.asarray(q_features), name=f"{model}:q_features"),
                "hash_kernel_gram": None if kernel_gram is None else stable_hash_array(np.asarray(kernel_gram), name=f"{model}:kernel_gram"),
            },
            "summaries": {
                "pred_surface": summarize_matrix(pred_surface),
                "pred_factors": summarize_matrix(pred_factor),
                "q_features": None if q_features is None else summarize_matrix(np.asarray(q_features)),
                "kernel_gram": None if kernel_gram is None else _kernel_gram_summary(np.asarray(kernel_gram)),
            },
            "metrics": {
                "surface_mape": float(row.get("surface_mape", np.nan)),
                "factor_mape": float(row.get("factor_mape", np.nan)),
                "w_star": None if row.get("w_star") is None else float(row.get("w_star")),
                "shots_per_eval": shots_per_eval,
                "total_qevals": total_qevals,
                "total_shots": total_shots,
            },
            "budgets": {
                "shots_per_eval": shots_per_eval,
                "total_qevals": total_qevals,
                "total_shots": total_shots,
            },
        }
        append_jsonl(out_path, record)

    duplicates = {h: ms for h, ms in hash_to_models.items() if len(ms) > 1}
    if duplicates:
        flat = [sorted(v) for v in duplicates.values()]
        print(f"[model_compare] hash-identical surface predictions detected: {flat}")
    return duplicates


def run_comparison(
    data_dir: str | Path,
    level: int,
    backend: str,
    seed: int,
    out_dir: str | Path,
    include_quantum: bool = True,
    include_ported: bool = True,
    fast: bool = False,
    quantum_bases: list[str] | None = None,
    quantum_modes: list[str] | None = None,
    use_cycle_phase: bool = True,
    forecast_horizons: int = 6,
    metric: str = "mape",
    metrics: list[str] | None = None,
    topk_lines: int = 8,
    mape_eps: float = 1e-8,
    quantum_feature_mode: str = "concrete_features",
    q_dim: int = 64,
    q_shots: int = 256,
    q_angle_scale: float = 1.0,
    q_seed: int = 0,
    w_min: float = 0.05,
    min_qfeat_std: float = 1e-8,
    allow_duplicate_hashes: bool = False,
    include_classical: bool = True,
    lookback: int | None = None,
    sanity_checks: bool = True,
    model: str | None = None,
    classical_models: list[str] | None = None,
    pqrc_M: int = 8,
    pqrc_modes: int | None = None,
    pqrc_Nph: int = 2,
    pqrc_nphotons: int | None = None,
    pqrc_budget: int = 32,
    pqrc_gain: float | None = None,
    pqrc_feature: str = "clickprob",
    pqrc_pseudocount: float = 0.25,
    pqrc_higher_order: int = 0,
    pqrc_budget_alloc: str = "uniform",
    pqrc_shots: int = 32,
    pqrc_ridge: float | None = None,
    pqrc_input_scale: float | None = None,
    pqrc_in_pca: int = 4,
    pqrc_factor_cap: int = 0,
    qrc_mode: str = "residual",
    qrc_target: str = "delta",
    qrc_baseline: str = "persistence",
    target_transform: str = "log",
    y_floor_mode: str = "train_p001",
    y_floor_value: float | None = None,
    persist_qrc_alpha: float = 0.25,
    persist_qrc_rclip: float = 0.05,
    persist_qrc_drive_scale: float = 0.25,
    tau: float | None = None,
    qrc_gate_tau: float = 0.06184302083987248,
    qrc_resid_clip: float | None = None,
    qrc_residvar_penalty: float = 0.1,
    qrc_feat_norm: str = "standard",
    qrc_residual_penalty: float = 0.1,
    qrc_load_from: str | None = None,
    qrc_model_hash: str | None = None,
    qrc_optimal_guard: bool = True,
    allow_nonoptimal_qrc_hash: bool = False,
    qrc_sweep_objective: str = "surface_mape",
    sweep_objective: str | None = None,
    pqrc_sweep: bool = False,
    pqrc_sweep_subset: int = 256,
    pqrc_sweep_topk: int = 3,
    sweep_modes: bool = False,
    sweep_gain: bool = False,
    sweep_input_scale: bool = False,
    sweep_ridge: bool = False,
    performance_preset: bool = False,
    debug_positive_checks: bool = False,
) -> tuple[pd.DataFrame, Path]:
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

    metric_norm = str(metric).strip().lower()
    if metric_norm not in {"mape", "mae", "rmse", "smape", "wape"}:
        raise ValueError("--metric must be one of: mape, mae, rmse, smape, wape")
    metrics_use = _parse_metrics_arg(",".join(metrics) if metrics is not None else None, default_metric=metric_norm)
    print(f"[model_compare] metrics enabled: {metrics_use}")

    cfg = _load_submission_config(Path(__file__).with_name("config.yaml"))
    lcfg = _level_cfg(cfg, int(level))
    qcfg = lcfg.get("quantum", {}) if isinstance(lcfg.get("quantum", {}), dict) else {}

    L_cfg = int(lcfg.get("L", 20))
    L = int(lookback) if (lookback is not None and int(lookback) > 0) else L_cfg
    H_cfg = int(lcfg.get("H", 10))
    H = int(max(1, forecast_horizons))
    if L != L_cfg:
        print(f"[model_compare] overriding config lookback L={L_cfg} with lookback={L}.")
    if H != H_cfg:
        print(f"[model_compare] overriding config horizon H={H_cfg} with forecast_horizons={H}.")
    qrc_target_norm = str(qrc_target).strip().lower()
    tau_default = 0.5 if qrc_target_norm == "norm_resid" else (0.02 if qrc_target_norm == "log_return" else 0.05)
    gate_tau_raw = float((qrc_gate_tau if tau is None else tau) if (qrc_gate_tau is not None or tau is not None) else tau_default)
    if qrc_target_norm == "norm_resid" and tau is None and float(qrc_gate_tau) == 0.05:
        gate_tau_raw = 0.5
    if qrc_target_norm == "log_return" and tau is None and float(qrc_gate_tau) == 0.05:
        gate_tau_raw = 0.02

    pqrc_modes_input = int(pqrc_M if pqrc_modes is None else pqrc_modes)
    pqrc_nph_input = int(pqrc_nphotons if pqrc_nphotons is not None else pqrc_Nph)
    if bool(performance_preset):
        # Keep local defaults safe, but let users opt into high-capacity settings explicitly.
        if pqrc_modes is None and int(pqrc_M) == 12:
            pqrc_modes_input = 20
        if pqrc_nphotons is None and int(pqrc_Nph) == 2:
            pqrc_nph_input = 10
        print(
            "[model_compare] performance preset enabled: "
            f"pqrc_modes={int(pqrc_modes_input)} pqrc_nphotons={int(pqrc_nph_input)}"
        )
    objective_cli = str(sweep_objective).strip() if sweep_objective is not None else ""
    qrc_sweep_objective_eff = objective_cli or str(qrc_sweep_objective)

    qrc_cfg = QRCConfig.from_dict(
        {
            "qrc_mode": str(qrc_mode),
            "qrc_target": str(qrc_target),
            "qrc_baseline": str(qrc_baseline),
            "target_transform": str(target_transform),
            "y_floor_mode": str(y_floor_mode),
            "y_floor_value": (None if y_floor_value is None else float(y_floor_value)),
            "qrc_feat_norm": str(qrc_feat_norm),
            "qrc_gate_tau": float(gate_tau_raw),
            "qrc_resid_clip": None if qrc_resid_clip is None else float(qrc_resid_clip),
            "qrc_residvar_penalty": float(qrc_residvar_penalty),
            "qrc_sweep_objective": _sanitize_objective_name(str(qrc_sweep_objective_eff)),
            "pqrc_modes": int(np.clip(int(pqrc_modes_input), 1, 20)),
            "pqrc_gain": float(0.5 if pqrc_gain is None else pqrc_gain),
            "pqrc_input_scale": float(0.75 if pqrc_input_scale is None else pqrc_input_scale),
            "pqrc_ridge": float(1e-2 if pqrc_ridge is None else pqrc_ridge),
            "pqrc_shots": int(pqrc_shots),
            "pqrc_budget": int(pqrc_budget),
            "pqrc_nph": int(pqrc_nph_input),
            "pqrc_in_pca": int(pqrc_in_pca),
            "pqrc_feature": str(pqrc_feature),
            "pqrc_pseudocount": float(pqrc_pseudocount),
            "pqrc_higher_order": int(pqrc_higher_order),
            "pqrc_budget_alloc": str(pqrc_budget_alloc),
            "seed": int(seed),
            "level": int(level),
            "lookback": int(L),
            "horizons": int(H),
        }
    ).validate()

    loaded_qrc_hash: str | None = None
    if qrc_load_from:
        preferred_qrc_models: tuple[str, ...] = ("photonic_qrc_feedback",)
        model_cli = str(model or "").strip().lower()
        if model_cli in {"photonic_qrc_feedback", "photonic_qrc_no_feedback"}:
            preferred_qrc_models = (model_cli, "photonic_qrc_feedback")
        elif model_cli in {"photonic_memory", "photonic_memory_no_feedback"}:
            preferred_qrc_models = (model_cli, "photonic_memory")
        elif classical_models:
            req = [str(m).strip().lower() for m in classical_models]
            if "photonic_qrc_feedback" in req:
                preferred_qrc_models = ("photonic_qrc_feedback", "photonic_qrc_no_feedback")
            elif "photonic_qrc_no_feedback" in req:
                preferred_qrc_models = ("photonic_qrc_no_feedback", "photonic_qrc_feedback")
            elif "photonic_memory" in req:
                preferred_qrc_models = ("photonic_memory", "photonic_memory_no_feedback")
            elif "photonic_memory_no_feedback" in req:
                preferred_qrc_models = ("photonic_memory_no_feedback", "photonic_memory")
        loaded_cfg_dict, loaded_qrc_hash = _load_qrc_config_from_source(
            qrc_load_from,
            level=int(level),
            model_hash=qrc_model_hash,
            objective_name=_sanitize_objective_name(str(qrc_cfg.qrc_sweep_objective)),
            enforce_optimal=bool(qrc_optimal_guard),
            allow_nonoptimal_hash=bool(allow_nonoptimal_qrc_hash),
            preferred_models=preferred_qrc_models,
        )
        qrc_cfg = QRCConfig.from_dict({**qrc_cfg.to_dict(), **loaded_cfg_dict, "level": int(level), "lookback": int(L), "horizons": int(H)}).validate()
        if bool(pqrc_sweep):
            print("[model_compare] qrc_load_from provided; disabling internal pqrc_sweep and replaying loaded config exactly.")
        pqrc_sweep = False

    qrc_mode = str(qrc_cfg.qrc_mode)
    qrc_target = str(qrc_cfg.qrc_target)
    qrc_baseline = str(qrc_cfg.qrc_baseline)
    target_transform = str(qrc_cfg.target_transform)
    y_floor_mode = str(qrc_cfg.y_floor_mode)
    y_floor_value = qrc_cfg.y_floor_value
    qrc_feat_norm = str(qrc_cfg.qrc_feat_norm)
    gate_tau = float(qrc_cfg.qrc_gate_tau)
    qrc_resid_clip = qrc_cfg.qrc_resid_clip
    qrc_residvar_penalty = float(qrc_cfg.qrc_residvar_penalty)
    qrc_sweep_objective = _sanitize_objective_name(str(qrc_cfg.qrc_sweep_objective))
    pqrc_modes_eff = int(np.clip(int(qrc_cfg.pqrc_modes), 1, 20))
    pqrc_gain = float(qrc_cfg.pqrc_gain)
    pqrc_ridge = float(qrc_cfg.pqrc_ridge)
    pqrc_shots = int(qrc_cfg.pqrc_shots)
    pqrc_budget = int(qrc_cfg.pqrc_budget)
    pqrc_Nph = int(qrc_cfg.pqrc_nph)
    pqrc_in_pca = int(qrc_cfg.pqrc_in_pca)
    pqrc_feature = str(qrc_cfg.pqrc_feature)
    pqrc_higher_order = int(qrc_cfg.pqrc_higher_order)
    pqrc_budget_alloc = str(qrc_cfg.pqrc_budget_alloc)
    pqrc_input_scale = float(qrc_cfg.pqrc_input_scale)
    qrc_config_hash = str(loaded_qrc_hash or config_to_hash(qrc_cfg))
    qrc_seed_eff = int(getattr(qrc_cfg, "seed", seed))

    imputer = str(lcfg.get("imputer", "ffill_interp" if int(level) == 1 else "svd_iterative"))
    d_factors = int(lcfg.get("d_factors", 8))

    quantum_mode = str(qcfg.get("mode", "q_bottleneck"))
    shots = int(max(1, int(qcfg.get("shots", q_shots))))
    qrc_stride = int(qcfg.get("qrc_stride", 1))
    qrc_feature_dim = int(qcfg.get("qrc_feature_dim", 24))
    train_qeval_budget = int(qcfg.get("train_qeval_budget", 5000))
    batch_size = int(qcfg.get("batch_size", 32))

    if fast or os.environ.get("SUBMISSION_SMOKE_FAST", "0") == "1":
        imputer = "ffill_interp"
        d_factors = min(d_factors, 6)
        shots = max(1, min(shots, 16))
        q_dim = min(int(q_dim), 32)
        qrc_feature_dim = min(qrc_feature_dim, 16)
        qrc_stride = max(qrc_stride, 2)
        train_qeval_budget = min(train_qeval_budget, 1200)
        batch_size = max(batch_size, 64)
        print("[model_compare] fast mode enabled.")

    bundle = load_and_preprocess(
        data_dir=data_dir,
        level=int(level),
        L=L,
        H=H,
        imputer=imputer,
        d_factors=d_factors,
        seed=seed,
        use_cycle_phase=bool(use_cycle_phase),
    )
    if str(target_transform).strip().lower() == "log":
        floor_fit = PositiveLogTransform(
            floor_mode=str(y_floor_mode),
            floor_value=y_floor_value,
            eps_min=1e-8,
        ).fit(bundle.y_train_surface_scaled * bundle.price_scale)
        setattr(bundle, "surface_floor", float(floor_fit.clamp(np.array([0.0]))[0]))
    else:
        setattr(bundle, "surface_floor", 0.0)
    y_surface_floor = float(getattr(bundle, "surface_floor", 0.0))
    y_true_train_surface = np.clip(bundle.y_train_surface_scaled * bundle.price_scale, y_surface_floor, bundle.clip_cap)
    y_true_val_surface = np.clip(bundle.y_val_surface_scaled * bundle.price_scale, y_surface_floor, bundle.clip_cap)
    y_true_surface = np.clip(bundle.y_test_surface_scaled * bundle.price_scale, y_surface_floor, bundle.clip_cap)
    y_true_eval_surface = y_true_val_surface
    y_true_eval_norm = bundle.y_val_norm
    train_fit_slice, train_tune_slice = _train_fit_tune_slices(int(bundle.y_train_norm.shape[0]))
    print(
        f"[model_compare] internal_train_split fit={int(train_fit_slice.stop - train_fit_slice.start)} "
        f"tune={int(train_tune_slice.stop - train_tune_slice.start)} "
        "(external validation kept strictly for evaluation)."
    )
    dataset_signature = _dataset_signature(bundle, level=int(level), lookback=int(L), horizon=int(H))
    print(f"[model_compare] dataset_signature={dataset_signature}")

    pqrc_shots_eff = int(pqrc_shots)
    if pqrc_shots_eff <= 0:
        pqrc_shots_eff = int(max(1, int(qcfg.get("shots", 32))))
        print(
            f"[model_compare] non-positive pqrc_shots={int(pqrc_shots)} received; "
            f"using pqrc_shots={pqrc_shots_eff}."
        )
    pqrc_modes_eff = int(np.clip(int(qrc_cfg.pqrc_modes), 1, 20))
    run_config_hash = stable_hash_config(
        {
            "data_dir": str(data_dir),
            "level": int(level),
            "backend": str(backend),
            "seed": int(seed),
            "lookback": int(L),
            "horizon": int(H),
            "fast": bool(fast),
            "classical_models": [str(m) for m in (classical_models or [])],
            "pqrc_M": int(pqrc_modes_eff),
            "pqrc_modes": int(pqrc_modes_eff),
            "pqrc_Nph": int(pqrc_Nph),
            "pqrc_nphotons": int(pqrc_Nph),
            "pqrc_budget": int(pqrc_budget),
            "pqrc_gain": float(pqrc_gain),
            "pqrc_input_scale": float(pqrc_input_scale),
            "pqrc_feature": str(pqrc_feature),
            "pqrc_pseudocount": float(qrc_cfg.pqrc_pseudocount),
            "pqrc_higher_order": int(pqrc_higher_order),
            "pqrc_budget_alloc": str(pqrc_budget_alloc),
            "pqrc_shots": int(pqrc_shots_eff),
            "pqrc_ridge": float(pqrc_ridge),
            "pqrc_in_pca": int(pqrc_in_pca),
            "pqrc_factor_cap": int(pqrc_factor_cap),
            "qrc_mode": str(qrc_mode),
            "qrc_target": str(qrc_target),
            "qrc_baseline": str(qrc_baseline),
            "target_transform": str(target_transform),
            "y_floor_mode": str(y_floor_mode),
            "y_floor_value": None if y_floor_value is None else float(y_floor_value),
            "tau": float(gate_tau),
            "qrc_gate_tau": float(gate_tau),
            "qrc_resid_clip": None if qrc_resid_clip is None else float(qrc_resid_clip),
            "qrc_residvar_penalty": float(qrc_residvar_penalty),
            "qrc_feat_norm": str(qrc_feat_norm),
            "qrc_residual_penalty": float(qrc_residual_penalty),
            "qrc_sweep_objective": str(qrc_sweep_objective),
            "qrc_seed": int(qrc_seed_eff),
            "persist_qrc_alpha": float(persist_qrc_alpha),
            "persist_qrc_rclip": float(persist_qrc_rclip),
            "persist_qrc_drive_scale": float(persist_qrc_drive_scale),
            "pqrc_sweep": bool(pqrc_sweep),
            "sweep_modes": bool(sweep_modes),
            "sweep_gain": bool(sweep_gain),
            "sweep_input_scale": bool(sweep_input_scale),
            "sweep_ridge": bool(sweep_ridge),
            "pqrc_sweep_subset": int(pqrc_sweep_subset),
            "performance_preset": bool(performance_preset),
            "qrc_load_from": None if qrc_load_from is None else str(qrc_load_from),
            "qrc_model_hash": None if qrc_model_hash is None else str(qrc_model_hash),
            "qrc_optimal_guard": bool(qrc_optimal_guard),
            "allow_nonoptimal_qrc_hash": bool(allow_nonoptimal_qrc_hash),
            "qrc_config_hash": str(qrc_config_hash),
            "dataset_signature": str(dataset_signature),
        },
        name="run_config",
    )

    eval_rows: list[dict[str, Any]] = []

    classical_models_all: list[tuple[str, str]] = [
        (PERSISTENCE_SURFACE_NAME, PERSISTENCE_SURFACE_NAME),
        ("persistence_naive", "persistence"),
        ("factor_ar", "factor_ar"),
        ("mlp", "mlp"),
        ("gru", "gru"),
        ("lstm", "lstm"),
        ("reservoir", "reservoir"),
        ("photonic_qrc_feedback", "photonic_qrc_feedback"),
        ("photonic_qrc_no_feedback", "photonic_qrc_no_feedback"),
        ("photonic_memory", "photonic_memory"),
        ("photonic_memory_no_feedback", "photonic_memory_no_feedback"),
        ("persist_qrc_weak", "persist_qrc_weak"),
    ]
    model_kind_map = {name: kind for name, kind in classical_models_all}
    if model is not None and str(model).strip():
        classical_models = [str(model).strip().lower()]
    requested_classical_models = (
        [str(m).strip().lower() for m in classical_models]
        if classical_models is not None
        else list(model_kind_map.keys())
    )
    requested_classical_models = [m for m in requested_classical_models if m in model_kind_map]
    if (include_classical or include_quantum) and PERSISTENCE_SURFACE_NAME not in requested_classical_models:
        requested_classical_models = [PERSISTENCE_SURFACE_NAME] + requested_classical_models
    base_name_cfg = str(lcfg.get("classical_kind", "mlp")).strip().lower()
    if base_name_cfg == "persistence":
        base_name_cfg = PERSISTENCE_SURFACE_NAME
    requested_q_bases = (
        quantum_bases
        if quantum_bases is not None
        else [base_name_cfg, "mlp", "gru", "lstm", "factor_ar", "reservoir", PERSISTENCE_SURFACE_NAME]
    )
    requested_q_bases = list(dict.fromkeys([b for b in requested_q_bases if b in model_kind_map]))
    if include_quantum and not requested_q_bases:
        requested_q_bases = ["mlp"]

    if include_classical:
        classical_models = [(name, model_kind_map[name]) for name in requested_classical_models]
    elif include_quantum:
        classical_models = [(name, model_kind_map[name]) for name in requested_q_bases]
    else:
        classical_models = []

    stored_classical: dict[str, dict[str, np.ndarray]] = {}
    qrc_top_plot_entries: list[dict[str, Any]] = []
    qrc_sweep_history_records: list[dict[str, Any]] = []
    qrc_memory_proxy_records: list[dict[str, Any]] = []
    best_qrc_config_to_save: dict[str, Any] | None = None
    persistence_eval_surface = _surface_persistence_prediction(bundle, bundle.ds.val_slice, H)
    for model_name, kind in classical_models:
        kind_norm = str(kind).strip().lower()
        if kind_norm in {
            "reservoir",
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "photonic_memory",
            "photonic_memory_no_feedback",
            "persist_qrc_weak",
        }:
            x_train_in = np.concatenate([bundle.x_train_norm, bundle.missing_train_norm], axis=2)
            x_val_in = np.concatenate([bundle.x_val_norm, bundle.missing_val_norm], axis=2)
            x_test_in = np.concatenate([bundle.x_test_norm, bundle.missing_test_norm], axis=2)
        else:
            x_train_in = bundle.x_train_norm
            x_val_in = bundle.x_val_norm
            x_test_in = bundle.x_test_norm
        model_kwargs: dict[str, Any] = {}
        model_seed = int(qrc_seed_eff) if kind_norm in {
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "photonic_memory",
            "photonic_memory_no_feedback",
            "persist_qrc_weak",
        } else int(seed)
        if kind_norm in {
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "photonic_memory",
            "photonic_memory_no_feedback",
            "persist_qrc_weak",
        }:
            model_kwargs = {
                "pqrc_M": int(pqrc_modes_eff),
                "pqrc_modes": int(pqrc_modes_eff),
                "pqrc_Nph": int(pqrc_Nph),
                "pqrc_nph": int(pqrc_Nph),
                "pqrc_nphotons": int(pqrc_Nph),
                "pqrc_budget": int(pqrc_budget),
                "pqrc_gain": float(pqrc_gain),
                "pqrc_input_scale": float(pqrc_input_scale),
                "pqrc_feature": str(pqrc_feature),
                "pqrc_pseudocount": float(qrc_cfg.pqrc_pseudocount),
                "pqrc_higher_order": int(pqrc_higher_order),
                "pqrc_budget_alloc": str(pqrc_budget_alloc),
                "pqrc_shots": int(pqrc_shots_eff),
                "pqrc_ridge": float(pqrc_ridge),
                "pqrc_in_pca": int(pqrc_in_pca),
                "pqrc_factor_cap": int(pqrc_factor_cap),
                "qrc_mode": str(qrc_mode),
                "qrc_target": str(qrc_target),
                "qrc_baseline": str(qrc_baseline),
                "target_transform": str(target_transform),
                "y_floor_mode": str(y_floor_mode),
                "y_floor_value": (None if y_floor_value is None else float(y_floor_value)),
                "tau": float(gate_tau),
                "qrc_gate_tau": float(gate_tau),
                "qrc_resid_clip": None if qrc_resid_clip is None else float(qrc_resid_clip),
                "qrc_residvar_penalty": float(qrc_residvar_penalty),
                "qrc_feat_norm": str(qrc_feat_norm),
                "persist_qrc_alpha": float(persist_qrc_alpha),
                "persist_qrc_rclip": float(persist_qrc_rclip),
                "persist_qrc_drive_scale": float(persist_qrc_drive_scale),
                "seed": int(qrc_seed_eff),
                "level": int(level),
                "lookback": int(L),
                "horizons": int(H),
                "qrc_config_hash": str(qrc_config_hash),
                "dataset_signature": str(dataset_signature),
            }
        y_train_surface_pred: np.ndarray | None = None
        y_val_surface_pred: np.ndarray | None = None
        y_test_surface_pred: np.ndarray | None = None
        if kind_norm in {
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "photonic_memory",
            "photonic_memory_no_feedback",
            "persist_qrc_weak",
        } and bool(pqrc_sweep):
            sweep_res = _run_photonic_coordinate_sweep(
                kind=kind_norm,
                seed=int(qrc_seed_eff),
                base_kwargs=model_kwargs,
                x_train=x_train_in[train_fit_slice],
                y_train=bundle.y_train_norm[train_fit_slice],
                x_val=x_train_in[train_tune_slice],
                y_val=bundle.y_train_norm[train_tune_slice],
                x_test=x_val_in,
                y_true_test_norm=bundle.y_val_norm,
                bundle=bundle,
                y_true_val_surface=y_true_train_surface[train_tune_slice],
                y_true_test_surface=y_true_eval_surface,
                mape_eps=float(mape_eps),
                sweep_subset=int(pqrc_sweep_subset),
                topk=int(max(1, pqrc_sweep_topk)),
                residual_penalty_weight=float(qrc_residvar_penalty if qrc_residvar_penalty is not None else qrc_residual_penalty),
                sweep_objective=str(qrc_sweep_objective),
                sweep_modes=bool(sweep_modes),
                sweep_gain=bool(sweep_gain),
                sweep_input_scale=bool(sweep_input_scale),
                sweep_ridge=bool(sweep_ridge),
            )
            best = sweep_res["best"]
            model = best["model_obj"]
            model_kwargs = dict(best["model_kwargs"])
            if str(model_name) == "photonic_qrc_feedback":
                best_qrc_config_to_save = dict(model_kwargs)
            y_train = model.predict(x_train_in)
            y_val = model.predict(x_val_in)
            y_test = model.predict(x_test_in)
            for ent in sweep_res["top_full"]:
                qrc_top_plot_entries.append(
                    {
                        "model": str(model_name),
                        "model_hash": str(ent["model_hash"]),
                        "horizon_surface_mape": np.asarray(ent["horizon_mape_test"], dtype=float),
                        "collapse_curve": np.asarray(ent["collapse_curve"], dtype=float),
                        "feature_median_std": float(ent.get("feature_median_std", 0.0)),
                        "val_mape": float(ent.get("val_mape", np.inf)),
                        "qrc_mode_used": str(ent.get("qrc_mode_used", "")),
                    }
                )
            for rec in sweep_res["history"]:
                cfg_row = dict(rec.get("model_kwargs", {}) or {})
                row_obj = {
                    "model": str(model_name),
                    "model_hash": str(rec.get("model_hash", "")),
                    "config_hash": str(rec.get("model_hash", "")),
                    "dataset_signature": str(dataset_signature),
                    "phase": str(rec.get("phase", "")),
                    "objective": str(rec.get("objective", qrc_sweep_objective)),
                    "objective_name": str(rec.get("objective_name", qrc_sweep_objective)),
                    "objective_value": float(rec.get("objective_value", np.nan)),
                    "score": float(rec.get("score", np.inf)),
                    "val_mape": float(rec.get("val_mape", np.inf)),
                    "val_mae": float(rec.get("val_mae", np.inf)),
                    "val_surface_mape": float(rec.get("val_surface_mape", np.nan)),
                    "val_surface_wape": float(rec.get("val_surface_wape", np.nan)),
                    "val_surface_mae": float(rec.get("val_surface_mae", np.nan)),
                    "val_surface_rmse": float(rec.get("val_surface_rmse", np.nan)),
                    "residual_std_ratio": float(rec.get("residual_std_ratio", np.nan)),
                    "residual_penalty": float(rec.get("residual_penalty", np.nan)),
                    "qrc_mode_used": str(rec.get("qrc_mode_used", "")),
                    "collapsed": bool(rec.get("collapsed", False)),
                    "failed": bool(rec.get("failed", False)),
                    "error": str(rec.get("error", "")),
                    "sweep_objective": str(qrc_sweep_objective),
                    "objective_components": str(rec.get("objective_components", "")),
                    "config": json.dumps(cfg_row, sort_keys=True, default=str, separators=(",", ":")),
                }
                for hp in [
                    "qrc_mode",
                    "qrc_target",
                    "qrc_baseline",
                    "target_transform",
                    "y_floor_mode",
                    "y_floor_value",
                    "qrc_feat_norm",
                    "qrc_gate_tau",
                    "qrc_resid_clip",
                    "qrc_residvar_penalty",
                    "pqrc_modes",
                    "pqrc_gain",
                    "pqrc_input_scale",
                    "pqrc_ridge",
                    "pqrc_shots",
                    "pqrc_budget",
                    "pqrc_nph",
                    "pqrc_in_pca",
                    "pqrc_feature",
                    "pqrc_pseudocount",
                    "pqrc_higher_order",
                    "pqrc_budget_alloc",
                    "persist_qrc_alpha",
                    "persist_qrc_rclip",
                    "persist_qrc_drive_scale",
                    "seed",
                    "level",
                    "lookback",
                    "horizons",
                ]:
                    row_obj[hp] = cfg_row.get(hp, np.nan)
                qrc_sweep_history_records.append(
                    row_obj
                )
            for rec in sweep_res.get("memory_proxy_records", []):
                cfg_row = dict(rec.get("model_kwargs", {}) or {})
                mp_row = {
                    "model": str(model_name),
                    "model_hash": str(rec.get("model_hash", "")),
                    "config_hash": str(rec.get("model_hash", "")),
                    "dataset_signature": str(dataset_signature),
                    "phase": str(rec.get("phase", "")),
                    "objective": str(rec.get("objective", qrc_sweep_objective)),
                    "objective_components": str(rec.get("objective_components", "")),
                    "mc_score": float(rec.get("mc_score", np.nan)),
                    "stability_penalty": float(rec.get("stability_penalty", np.nan)),
                    "mc_score_adj": float(rec.get("mc_score_adj", np.nan)),
                    "state_autocorr_lag1": float(rec.get("state_autocorr_lag1", np.nan)),
                    "feature_std_median": float(rec.get("feature_std_median", np.nan)),
                    "failed": bool(rec.get("failed", False)),
                    "error": str(rec.get("error", "")),
                    "config": json.dumps(cfg_row, sort_keys=True, default=str, separators=(",", ":")),
                }
                for hp in [
                    "pqrc_modes",
                    "pqrc_gain",
                    "pqrc_input_scale",
                    "pqrc_ridge",
                    "pqrc_shots",
                    "pqrc_budget",
                    "qrc_gate_tau",
                    "pqrc_feature",
                    "pqrc_pseudocount",
                    "qrc_mode",
                    "qrc_target",
                    "qrc_baseline",
                    "target_transform",
                    "y_floor_mode",
                    "y_floor_value",
                ]:
                    mp_row[hp] = cfg_row.get(hp, np.nan)
                r2_vals = list(rec.get("r2_by_k", []))
                for i_k, v_k in enumerate(r2_vals, start=1):
                    mp_row[f"r2_k{i_k}"] = float(v_k)
                qrc_memory_proxy_records.append(mp_row)
        elif kind_norm in {"persistence_surface", PERSISTENCE_SURFACE_NAME}:
            y_train_surface_pred = _surface_persistence_prediction(bundle, bundle.ds.train_slice, H)
            y_val_surface_pred = _surface_persistence_prediction(bundle, bundle.ds.val_slice, H)
            y_test_surface_pred = _surface_persistence_prediction(bundle, bundle.ds.test_slice, H)
            y_train = _surface_to_factor_norm(bundle, y_train_surface_pred)
            y_val = _surface_to_factor_norm(bundle, y_val_surface_pred)
            y_test = _surface_to_factor_norm(bundle, y_test_surface_pred)
            max_abs_surface_diff = float(np.max(np.abs(y_val_surface_pred - persistence_eval_surface)))
            model_kwargs["surface_persistence_max_abs_diff_eval"] = max_abs_surface_diff
        else:
            model_train_kwargs = {k: v for k, v in model_kwargs.items() if k not in {"seed", "kind"}}
            model = train_classical_forecaster(
                x_train=x_train_in,
                y_train=bundle.y_train_norm,
                x_val=x_val_in,
                y_val=bundle.y_val_norm,
                kind=kind,  # type: ignore[arg-type]
                seed=model_seed,
                **model_train_kwargs,
            )
            y_train = model.predict(x_train_in)
            y_val = model.predict(x_val_in)
            y_test = model.predict(x_test_in)
        if y_train_surface_pred is None:
            y_train_surface_pred = _to_surface_space(bundle, y_train)
        if y_val_surface_pred is None:
            y_val_surface_pred = _to_surface_space(bundle, y_val)
        if y_test_surface_pred is None:
            y_test_surface_pred = _to_surface_space(bundle, y_test)
        if kind_norm in {
            "photonic_qrc_feedback",
            "photonic_qrc_no_feedback",
            "photonic_memory",
            "photonic_memory_no_feedback",
            "persist_qrc_weak",
        }:
            delta_vs_persist = float(np.mean(np.abs(y_val_surface_pred - persistence_eval_surface)))
            print(f"[model_compare] {model_name} mean_abs_delta_vs_persistence={delta_vs_persist:.3e}")
            if delta_vs_persist <= 1e-8 and kind_norm in {
                "photonic_qrc_feedback",
                "photonic_qrc_no_feedback",
                "photonic_memory",
                "photonic_memory_no_feedback",
            }:
                retry_kwargs = {
                    **model_kwargs,
                    "pqrc_input_scale": 1.5,
                    "pqrc_min_residual_scale": 0.20,
                    "pqrc_gain": max(float(model_kwargs.get("pqrc_gain", 0.0)), 0.90)
                    if kind_norm in {"photonic_qrc_feedback", "photonic_memory"}
                    else 0.0,
                }
                print(
                    f"[model_compare] {model_name} collapsed to persistence; retrying with stronger settings: "
                    f"{retry_kwargs}"
                )
                retry_train_kwargs = {k: v for k, v in retry_kwargs.items() if k not in {"seed", "kind"}}
                model = train_classical_forecaster(
                    x_train=x_train_in,
                    y_train=bundle.y_train_norm,
                    x_val=x_val_in,
                    y_val=bundle.y_val_norm,
                    kind=kind,  # type: ignore[arg-type]
                    seed=model_seed + 101,
                    **retry_train_kwargs,
                )
                model_kwargs = retry_kwargs
                y_train = model.predict(x_train_in)
                y_val = model.predict(x_val_in)
                y_test = model.predict(x_test_in)
                y_train_surface_pred = _to_surface_space(bundle, y_train)
                y_val_surface_pred = _to_surface_space(bundle, y_val)
                y_test_surface_pred = _to_surface_space(bundle, y_test)
                delta_vs_persist = float(np.mean(np.abs(y_val_surface_pred - persistence_eval_surface)))
                print(f"[model_compare] {model_name} retry mean_abs_delta_vs_persistence={delta_vs_persist:.3e}")
                if delta_vs_persist <= 1e-8:
                    warnings.warn(
                        f"{model_name} remains collapsed to persistence after retry; marking as collapse_failed.",
                        RuntimeWarning,
                    )
                    model_kwargs["collapse_detected_runtime"] = True
            mode_used = getattr(model, "qrc_mode", None)
            if mode_used is not None:
                model_kwargs["qrc_mode_used"] = str(mode_used)
            target_used = getattr(model, "qrc_target", None)
            if target_used is not None:
                model_kwargs["qrc_target_used"] = str(target_used)
            baseline_used = getattr(model, "qrc_baseline", None)
            if baseline_used is not None:
                model_kwargs["qrc_baseline_used"] = str(baseline_used)
            if hasattr(model, "tau"):
                model_kwargs["tau_used"] = float(getattr(model, "tau"))
                model_kwargs["qrc_gate_tau"] = float(getattr(model, "tau"))
            if hasattr(model, "residual_scale"):
                model_kwargs["pqrc_residual_scale"] = float(getattr(model, "residual_scale"))
            if hasattr(model, "val_target_residual_mae"):
                model_kwargs["target_residual_mae_val"] = float(getattr(model, "val_target_residual_mae"))
            if hasattr(model, "qrc_config_hash"):
                model_kwargs["qrc_config_hash"] = str(getattr(model, "qrc_config_hash"))
            model_kwargs["qrc_mode_requested"] = str(qrc_mode)
            fstats = getattr(model, "feature_stats", None)
            if isinstance(fstats, dict):
                model_kwargs["feature_stats"] = fstats
                for k in (
                    "pqrc_pseudocount",
                    "min_prob_before",
                    "min_prob_after",
                    "fraction_zero_bins_before",
                    "fraction_zero_bins_after",
                ):
                    if k in fstats:
                        model_kwargs[k] = float(fstats.get(k, np.nan))
            q_features_arr = getattr(model, "last_q_features_", None)
            if q_features_arr is not None:
                model_kwargs["q_features"] = np.asarray(q_features_arr, dtype=float)
            q_gate = getattr(model, "last_gate_", None)
            if q_gate is not None and np.asarray(q_gate).shape == y_test.shape:
                model_kwargs["qrc_gate"] = np.asarray(q_gate, dtype=float)
            if hasattr(model, "_target_from_level") and hasattr(model, "_baseline_from_x"):
                try:
                    base_test_t = np.asarray(model._baseline_from_x(x_test_in), dtype=float)  # type: ignore[attr-defined]
                    pred_target_t = np.asarray(model._target_from_level(y_test, base_test_t, x_test_in), dtype=float)  # type: ignore[attr-defined]
                    true_target_t = np.asarray(model._target_from_level(bundle.y_test_norm, base_test_t, x_test_in), dtype=float)  # type: ignore[attr-defined]
                    model_kwargs["target_residual_mae_test"] = float(np.mean(np.abs(pred_target_t - true_target_t)))
                except Exception:
                    pass
            fb_sanity = getattr(model, "feedback_sanity", None)
            if isinstance(fb_sanity, dict):
                fb_diff = float(fb_sanity.get("phi_diff_gain0_vs_gain1", np.nan))
                fb_sanity["feedback_inert"] = bool(np.isfinite(fb_diff) and fb_diff < 1e-9)
                model_kwargs["feedback_sanity"] = fb_sanity
            train_qevals = int(getattr(model, "train_qevals", 0))
            infer_qevals = int(getattr(model, "infer_qevals", 0))
            train_total_shots = int(getattr(model, "train_total_shots", 0))
            infer_total_shots = int(getattr(model, "infer_total_shots", 0))
            shots_per_eval = int(getattr(model, "shots_per_eval", int(model_kwargs.get("pqrc_shots", 0))))
            model_kwargs["shots_per_eval"] = shots_per_eval
            model_kwargs["total_qevals"] = int(train_qevals + infer_qevals)
            model_kwargs["total_shots"] = int(train_total_shots + infer_total_shots)
            training_curve = getattr(model, "training_curve_", None)
            if isinstance(training_curve, list) and len(training_curve) > 0:
                try:
                    curve_df = pd.DataFrame(training_curve)
                    out_dir_cur = Path(out_dir)
                    out_dir_cur.mkdir(parents=True, exist_ok=True)
                    curve_csv = out_dir_cur / f"level{int(level)}_{str(model_name)}_train_error_vs_epoch.csv"
                    curve_png = out_dir_cur / f"level{int(level)}_{str(model_name)}_train_error_vs_epoch.png"
                    curve_df.to_csv(curve_csv, index=False)
                    _plot_qrc_training_error_vs_epoch(
                        curve_df,
                        curve_png,
                        title=f"Photonic QRC Best-So-Far Error vs Trial ({str(model_name)})",
                    )
                    model_kwargs["qrc_training_curve_csv"] = str(curve_csv)
                    model_kwargs["qrc_training_curve_png"] = str(curve_png)
                except Exception as exc:
                    warnings.warn(
                        f"failed to write qrc training-curve artifacts for {model_name}: {exc}",
                        RuntimeWarning,
                    )
        stored_classical[model_name] = {
            "train": y_train,
            "val": y_val,
            "test": y_test,
            "val_surface": y_val_surface_pred,
        }
        if include_classical:
            split_extra = _split_surface_mape_extra(
                y_true_train_surface=y_true_train_surface,
                y_pred_train_surface=y_train_surface_pred,
                y_true_val_surface=y_true_val_surface,
                y_pred_val_surface=y_val_surface_pred,
                eps=float(mape_eps),
            )
            eval_rows.append(
                _evaluate_model(
                    name=model_name,
                    y_pred_test_norm=y_val,
                    y_true_test_norm=y_true_eval_norm,
                    y_true_test_surface=y_true_eval_surface,
                    bundle=bundle,
                    mape_eps=mape_eps,
                    y_pred_test_surface=y_val_surface_pred,
                    extra={"features_input": x_val_in, **split_extra, **model_kwargs},
                )
            )

    if "photonic_qrc_feedback" in stored_classical and "photonic_qrc_no_feedback" in stored_classical:
        fb = stored_classical["photonic_qrc_feedback"]["val_surface"]
        nf = stored_classical["photonic_qrc_no_feedback"]["val_surface"]
        delta_fb_nf = float(np.mean(np.abs(fb - nf)))
        print(f"[model_compare] photonic feedback-vs-no_feedback mean_abs_delta={delta_fb_nf:.3e}")
        if delta_fb_nf <= 1e-8 and not bool(allow_duplicate_hashes):
            warnings.warn(
                "photonic_qrc_feedback and photonic_qrc_no_feedback are numerically identical on validation-surface predictions; "
                "both runs will be marked collapse_failed.",
                RuntimeWarning,
            )
    if "photonic_memory" in stored_classical and "photonic_memory_no_feedback" in stored_classical:
        fb = stored_classical["photonic_memory"]["val_surface"]
        nf = stored_classical["photonic_memory_no_feedback"]["val_surface"]
        delta_fb_nf = float(np.mean(np.abs(fb - nf)))
        print(f"[model_compare] photonic_memory feedback-vs-no_feedback mean_abs_delta={delta_fb_nf:.3e}")
        if delta_fb_nf <= 1e-8 and not bool(allow_duplicate_hashes):
            warnings.warn(
                "photonic_memory and photonic_memory_no_feedback are numerically identical on validation-surface predictions; "
                "both runs will be marked collapse_failed.",
                RuntimeWarning,
            )

    if include_classical or (not include_quantum and not include_ported):
        seasonal_pred_train = _predict_seasonal_naive(
            bundle.x_train_norm,
            horizon=H,
            d_factors=bundle.factor_dim,
            k=13,
        )
        seasonal_pred_val = _predict_seasonal_naive(
            bundle.x_val_norm,
            horizon=H,
            d_factors=bundle.factor_dim,
            k=13,
        )
        seasonal_pred = _predict_seasonal_naive(
            bundle.x_test_norm,
            horizon=H,
            d_factors=bundle.factor_dim,
            k=13,
        )
        seasonal_split_extra = _split_surface_mape_extra(
            y_true_train_surface=y_true_train_surface,
            y_pred_train_surface=_to_surface_space(bundle, seasonal_pred_train),
            y_true_val_surface=y_true_val_surface,
            y_pred_val_surface=_to_surface_space(bundle, seasonal_pred_val),
            eps=float(mape_eps),
        )
        eval_rows.append(
            _evaluate_model(
                name="seasonal_naive_13",
                y_pred_test_norm=seasonal_pred_val,
                y_true_test_norm=y_true_eval_norm,
                y_true_test_surface=y_true_eval_surface,
                bundle=bundle,
                mape_eps=mape_eps,
                extra={"features_input": bundle.x_val_norm, **seasonal_split_extra},
            )
        )

    mlp_test = stored_classical.get("mlp", {}).get("test")
    if include_classical and mlp_test is not None and all(k in stored_classical for k in ["persistence_naive", "factor_ar", "gru", "lstm", "reservoir"]):
        for other in ["persistence_naive", "factor_ar", "gru", "lstm", "reservoir"]:
            delta = float(np.mean(np.abs(mlp_test - stored_classical[other]["test"])))
            print(f"[model_compare] MLP distinctness vs {other}: mean_abs_delta={delta:.6e}")
            if delta < 1e-8:
                raise RuntimeError(f"MLP predictions are numerically identical to {other}; check model implementation.")

    if include_quantum and _HAS_QUANTUM:
        requested_bases = requested_q_bases
        requested_modes = quantum_modes if quantum_modes is not None else [quantum_mode, "q_bottleneck", "qrc"]
        q_bases = list(dict.fromkeys([b for b in requested_bases if b in stored_classical]))
        q_modes = list(dict.fromkeys([m for m in requested_modes if m in {"q_bottleneck", "qrc"}]))
        if not q_bases:
            q_bases = [base_name_cfg] if base_name_cfg in stored_classical else ["mlp"]
        if not q_modes:
            q_modes = [quantum_mode if quantum_mode in {"q_bottleneck", "qrc"} else "q_bottleneck"]

        n_q_variants = max(1, len(q_bases) * len(q_modes))
        per_variant_qeval_budget = max(200, int(train_qeval_budget // n_q_variants))
        print(
            f"[model_compare] quantum variants bases={q_bases} modes={q_modes} "
            f"per_variant_train_qeval_budget={per_variant_qeval_budget}"
        )

        for base_name in q_bases:
            for q_mode in q_modes:
                try:
                    q_payload = _build_quantum_comparison(
                        bundle=bundle,
                        classical_name=base_name,
                        y_base_train=stored_classical[base_name]["train"],
                        y_base_val=stored_classical[base_name]["val"],
                        y_base_test=stored_classical[base_name]["test"],
                        backend=backend,
                        quantum_mode=q_mode,
                        shots=shots,
                        qrc_stride=qrc_stride,
                        qrc_feature_dim=qrc_feature_dim,
                        train_qeval_budget=per_variant_qeval_budget,
                        batch_size=batch_size,
                        seed=seed,
                        quantum_feature_mode=quantum_feature_mode,
                        q_dim=int(q_dim),
                        q_angle_scale=float(q_angle_scale),
                        q_seed=int(q_seed),
                        min_qfeat_std=float(min_qfeat_std),
                        w_min=float(w_min),
                        mape_eps=float(mape_eps),
                        train_fit_slice=train_fit_slice,
                        train_tune_slice=train_tune_slice,
                    )
                    row = _evaluate_model(
                        name=q_payload["name"],
                        y_pred_test_norm=q_payload["pred_val_norm"],
                        y_true_test_norm=y_true_eval_norm,
                        y_true_test_surface=y_true_eval_surface,
                        bundle=bundle,
                        mape_eps=mape_eps,
                        extra={
                            "quantum_mode": q_payload.get("quantum_mode"),
                            "selected_dim": q_payload.get("selected_dim"),
                            "w_star": q_payload.get("w_star"),
                            "w_min": q_payload.get("w_min"),
                            "config_fingerprint": q_payload.get("config_fingerprint"),
                            "total_qevals": q_payload.get("total_qevals"),
                            "total_shots": q_payload.get("total_shots"),
                            "shots_per_eval": q_payload.get("shots_per_eval"),
                            "q_features": q_payload.get("q_features"),
                            "kernel_gram": q_payload.get("kernel_gram"),
                            "features_input": bundle.x_val_norm,
                            "surface_mape_train": q_payload.get("surface_mape_train"),
                            "surface_mape_val": q_payload.get("surface_mape_val"),
                            "horizon_surface_mape_train": q_payload.get("horizon_surface_mape_train"),
                            "horizon_surface_mape_val": q_payload.get("horizon_surface_mape_val"),
                        },
                    )
                    eval_rows.append(row)
                except Exception as exc:
                    print(f"[model_compare] quantum model skipped for base={base_name}, mode={q_mode}: {exc}")
    elif include_quantum and not _HAS_QUANTUM:
        print(f"[model_compare] quantum model skipped: {_QUANTUM_IMPORT_ERROR}")

    if include_ported and _HAS_PORTED:
        try:
            ported_rows = _build_ported_comparisons(
                bundle=bundle,
                backend=backend,
                shots=shots,
                qrc_stride=qrc_stride,
                qrc_feature_dim=qrc_feature_dim,
                train_qeval_budget=max(300, int(train_qeval_budget)),
                batch_size=batch_size,
                seed=seed,
                train_fit_slice=train_fit_slice,
                train_tune_slice=train_tune_slice,
            )
            for p in ported_rows:
                row = _evaluate_model(
                    name=p["name"],
                    y_pred_test_norm=p["pred_eval_norm"],
                    y_true_test_norm=y_true_eval_norm,
                    y_true_test_surface=y_true_eval_surface,
                    bundle=bundle,
                    mape_eps=mape_eps,
                    extra={
                        **dict(p.get("extra", {}) or {}),
                        "w_star": p.get("w_star"),
                        "w_min": float(w_min),
                        "algo_id": p.get("algo_id"),
                        "config_fingerprint": p.get("config_fingerprint"),
                        "q_features": p.get("q_features"),
                        "total_qevals": int(p.get("total_qevals", int(p.get("train_qevals", 0)) + int(p.get("infer_qevals", 0)))),
                        "total_shots": int(p.get("total_shots", int(p.get("train_total_shots", 0)) + int(p.get("infer_total_shots", 0)))),
                        "shots_per_eval": int(p.get("shots_per_eval", 0)),
                        "features_input": bundle.x_val_norm,
                    },
                )
                eval_rows.append(row)
        except Exception as exc:
            print(f"[model_compare] ported models skipped: {exc}")
    elif include_ported and not _HAS_PORTED:
        print(f"[model_compare] ported models skipped: {_PORTED_IMPORT_ERROR}")

    for row in eval_rows:
        row["model_hash"] = _build_model_hash(row, seed=int(seed), level=int(level), backend=str(backend))
        row.setdefault("qrc_config_hash", str(qrc_config_hash))
        row.setdefault("dataset_signature", str(dataset_signature))
    if bool(debug_positive_checks):
        bad = [
            (
                str(r.get("model")),
                int(r.get("surface_nonfinite_count", 0)),
                int(r.get("surface_nonpositive_count", 0)),
            )
            for r in eval_rows
            if int(r.get("surface_nonfinite_count", 0)) > 0 or int(r.get("surface_nonpositive_count", 0)) > 0
        ]
        if bad:
            raise RuntimeError(f"Positive-surface debug check failed: {bad}")

    collapse_diag_records = _compute_collapse_diagnostics(
        eval_rows,
        y_true_test_surface=y_true_eval_surface,
        eps=float(mape_eps),
    )
    if collapse_diag_records:
        by_model: dict[str, list[dict[str, Any]]] = {}
        for rec in collapse_diag_records:
            by_model.setdefault(str(rec["model"]), []).append(rec)
        for m, recs in sorted(by_model.items()):
            mean_abs = float(np.mean([float(r["mean_abs_diff_h"]) for r in recs]))
            mean_corr = float(np.mean([float(r["corr_h"]) for r in recs]))
            print(
                f"[collapse_diag] model={m} mean_abs_diff={mean_abs:.3e} mean_corr={mean_corr:.4f} "
                f"model_hash={recs[0].get('model_hash','')}"
            )

    metrics_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        metrics_rows.append(
            {
                "model": row["model"],
                "model_hash": row.get("model_hash"),
                "qrc_config_hash": row.get("qrc_config_hash"),
                "dataset_signature": row.get("dataset_signature"),
                "factor_mae": row["factor_mae"],
                "factor_rmse": row["factor_rmse"],
                "factor_mape": row["factor_mape"],
                "factor_smape": row["factor_smape"],
                "factor_wape": row["factor_wape"],
                "surface_mae": row["surface_mae"],
                "surface_rmse": row["surface_rmse"],
                "surface_mape": row["surface_mape"],
                "surface_smape": row["surface_smape"],
                "surface_wape": row["surface_wape"],
                "horizon_surface_mae": json.dumps([float(v) for v in np.asarray(row["horizon_surface_mae"])], separators=(",", ":")),
                "horizon_surface_rmse": json.dumps([float(v) for v in np.asarray(row["horizon_surface_rmse"])], separators=(",", ":")),
                "horizon_surface_mape": json.dumps([float(v) for v in np.asarray(row["horizon_surface_mape"])], separators=(",", ":")),
                "horizon_surface_smape": json.dumps([float(v) for v in np.asarray(row["horizon_surface_smape"])], separators=(",", ":")),
                "horizon_surface_wape": json.dumps([float(v) for v in np.asarray(row["horizon_surface_wape"])], separators=(",", ":")),
                "horizon_factor_mae": json.dumps([float(v) for v in np.asarray(row["horizon_factor_mae"])], separators=(",", ":")),
                "horizon_factor_rmse": json.dumps([float(v) for v in np.asarray(row["horizon_factor_rmse"])], separators=(",", ":")),
                "horizon_factor_mape": json.dumps([float(v) for v in np.asarray(row["horizon_factor_mape"])], separators=(",", ":")),
                "horizon_factor_smape": json.dumps([float(v) for v in np.asarray(row["horizon_factor_smape"])], separators=(",", ":")),
                "horizon_factor_wape": json.dumps([float(v) for v in np.asarray(row["horizon_factor_wape"])], separators=(",", ":")),
                "quantum_mode": row.get("quantum_mode"),
                "chosen_mode": row.get("qrc_mode_used", row.get("qrc_mode", row.get("quantum_mode"))),
                "qrc_target": row.get("qrc_target_used", row.get("qrc_target")),
                "qrc_baseline": row.get("qrc_baseline_used", row.get("qrc_baseline")),
                "target_transform": row.get("target_transform", target_transform),
                "y_floor_mode": row.get("y_floor_mode", y_floor_mode),
                "y_floor_value": row.get("y_floor_value", y_floor_value),
                "pqrc_pseudocount": row.get("pqrc_pseudocount", np.nan),
                "min_prob_before": row.get("min_prob_before", np.nan),
                "min_prob_after": row.get("min_prob_after", np.nan),
                "fraction_zero_bins_before": row.get("fraction_zero_bins_before", np.nan),
                "fraction_zero_bins_after": row.get("fraction_zero_bins_after", np.nan),
                "surface_floor": row.get("surface_floor", np.nan),
                "qrc_gate_tau": row.get("qrc_gate_tau", row.get("tau", np.nan)),
                "target_residual_mae_val": row.get("target_residual_mae_val", np.nan),
                "target_residual_mae_test": row.get("target_residual_mae_test", np.nan),
                "selected_dim": row.get("selected_dim"),
                "w_star": row.get("w_star"),
                "w_min": row.get("w_min"),
                "total_qevals": row.get("total_qevals"),
                "total_shots": row.get("total_shots"),
                "shots_per_eval": row.get("shots_per_eval"),
                "collapse_failed": bool(row.get("collapse_failed", False)),
                "collapse_mean_abs_diff": float(row.get("collapse_mean_abs_diff", np.nan)),
                "no_gain": bool(row.get("no_gain", False)),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["surface_metric"] = metrics_df[f"surface_{metric_norm}"]
    metrics_df["factor_metric"] = metrics_df[f"factor_{metric_norm}"]
    metrics_df = metrics_df.sort_values(["collapse_failed", "no_gain", "surface_metric"], ascending=[True, True, True]).reset_index(drop=True)

    mlp_row = next((r for r in eval_rows if str(r["model"]).lower() == "mlp"), None)
    if mlp_row is not None:
        mlp_surface = np.asarray(mlp_row["pred_test_surface"], dtype=float)
        model_to_delta_pred_mape: dict[str, float] = {}
        for r in eval_rows:
            pred = np.asarray(r["pred_test_surface"], dtype=float)
            den = np.maximum(np.abs(mlp_surface), float(max(1e-12, mape_eps)))
            delta = float(np.mean(np.abs(pred - mlp_surface) / den) * 100.0)
            model_to_delta_pred_mape[str(r["model"])] = delta
        metrics_df["delta_vs_mlp_mape_pred"] = metrics_df["model"].map(model_to_delta_pred_mape).astype(float)
        overlapping = [m for m, d in model_to_delta_pred_mape.items() if m != "mlp" and d < 1e-12]
        if overlapping:
            print(f"[model_compare] models overlapping MLP curve: {overlapping}")

    seasonal_row = next((r for r in eval_rows if str(r.get("model", "")).strip().lower() == "seasonal_naive_13"), None)
    if seasonal_row is not None:
        s_surface_mape = float(seasonal_row.get("surface_mape", np.nan))
        s_factor_mae = float(seasonal_row.get("factor_mae", np.nan))
        metrics_df["surface_mape_vs_seasonal13_delta"] = metrics_df["surface_mape"] - s_surface_mape
        metrics_df["factor_mae_vs_seasonal13_delta"] = metrics_df["factor_mae"] - s_factor_mae
    else:
        metrics_df["surface_mape_vs_seasonal13_delta"] = np.nan
        metrics_df["factor_mae_vs_seasonal13_delta"] = np.nan

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    # Always persist the fully materialized QRC config used for this run.
    save_config(qrc_cfg, out_dir_p / f"level{int(level)}_qrc_config_{qrc_config_hash}.json")
    if qrc_load_from:
        save_config(qrc_cfg, out_dir_p / "loaded_config.json")
    if best_qrc_config_to_save is not None:
        best_cfg = QRCConfig.from_dict(best_qrc_config_to_save).validate()
        save_config(best_cfg, out_dir_p / "best_config.json")

    metrics_path = out_dir_p / f"level{int(level)}_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(out_dir_p / f"level{int(level)}_metrics_summary_{run_config_hash}.csv", index=False)

    collapse_df = pd.DataFrame(collapse_diag_records)
    if not collapse_df.empty:
        collapse_path = out_dir_p / f"level{int(level)}_qrc_collapse_diag.csv"
        collapse_path_h = out_dir_p / f"level{int(level)}_qrc_collapse_diag_{run_config_hash}.csv"
        collapse_df.to_csv(collapse_path, index=False)
        collapse_df.to_csv(collapse_path_h, index=False)
        collapse_df.to_csv(out_dir_p / f"level{int(level)}_qrc_collapse_diag_{qrc_config_hash}.csv", index=False)

    feature_stats_df = _write_feature_stats_csv(
        eval_rows,
        out_path=out_dir_p / f"level{int(level)}_qrc_feature_stats_{run_config_hash}.csv",
    )
    if not feature_stats_df.empty:
        feature_stats_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feature_stats.csv", index=False)
        feature_stats_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feature_stats_{qrc_config_hash}.csv", index=False)

    fb_sanity_df = _write_feedback_sanity_csv(
        eval_rows,
        out_path=out_dir_p / f"level{int(level)}_qrc_feedback_sanity_{run_config_hash}.csv",
    )
    if not fb_sanity_df.empty:
        fb_sanity_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feedback_sanity.csv", index=False)
        fb_sanity_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feedback_sanity_{qrc_config_hash}.csv", index=False)

    feat_sanity_df = _write_feature_sanity_csv(
        eval_rows,
        out_path=out_dir_p / f"level{int(level)}_qrc_feature_sanity_{run_config_hash}.csv",
    )
    if not feat_sanity_df.empty:
        feat_sanity_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feature_sanity.csv", index=False)
        feat_sanity_df.to_csv(out_dir_p / f"level{int(level)}_qrc_feature_sanity_{qrc_config_hash}.csv", index=False)

    if qrc_sweep_history_records:
        hist_df = pd.DataFrame(qrc_sweep_history_records)
        numeric_hp = [
            "pqrc_modes",
            "pqrc_gain",
            "pqrc_input_scale",
            "pqrc_ridge",
            "pqrc_shots",
            "pqrc_budget",
            "qrc_gate_tau",
        ]
        for col in numeric_hp:
            if col in hist_df.columns:
                bad = ~np.isfinite(pd.to_numeric(hist_df[col], errors="coerce").to_numpy(dtype=float))
                if bool(np.any(bad)):
                    bad_rows = hist_df.loc[bad, ["model", "model_hash", col]].head(3).to_dict(orient="records")
                    raise RuntimeError(f"Sweep history has non-finite hyperparameter {col}. examples={bad_rows}")
        hist_df.to_csv(
            out_dir_p / f"level{int(level)}_qrc_sweep_history_{run_config_hash}.csv",
            index=False,
        )
        hist_df.to_csv(out_dir_p / f"level{int(level)}_qrc_sweep_history.csv", index=False)
    if qrc_memory_proxy_records:
        mp_df = pd.DataFrame(qrc_memory_proxy_records)
        mp_df.to_csv(out_dir_p / f"level{int(level)}_qrc_memory_proxy_{run_config_hash}.csv", index=False)
        mp_df.to_csv(out_dir_p / f"level{int(level)}_qrc_memory_proxy.csv", index=False)

    if not qrc_top_plot_entries:
        collapse_by_model_h: dict[tuple[str, int], float] = {}
        for rec in collapse_diag_records:
            collapse_by_model_h[(str(rec["model"]), int(rec["horizon"]))] = float(rec["mean_abs_diff_h"])
        for row in eval_rows:
            model_name = str(row.get("model", ""))
            if "qrc" not in model_name:
                continue
            h_mape = np.asarray(row.get("horizon_surface_mape"), dtype=float)
            collapse_curve = np.array(
                [collapse_by_model_h.get((model_name, i + 1), 0.0) for i in range(h_mape.shape[0])],
                dtype=float,
            )
            fs = row.get("feature_stats")
            fmed = 0.0
            if isinstance(fs, dict):
                std_arr = np.asarray(fs.get("feat_std_train", []), dtype=float)
                if std_arr.size:
                    fmed = float(np.median(std_arr))
            qrc_top_plot_entries.append(
                {
                    "model": model_name,
                    "model_hash": str(row.get("model_hash", "")),
                    "horizon_surface_mape": h_mape,
                    "collapse_curve": collapse_curve,
                    "feature_median_std": fmed,
                    "val_mape": float(row.get("surface_mape", np.inf)),
                    "qrc_mode_used": str(row.get("qrc_mode_used", row.get("qrc_mode", ""))),
                }
            )

    persistence_anchor = _persistence_anchor_name(eval_rows)
    persistence_row = next((r for r in eval_rows if str(r.get("model", "")).strip().lower() == persistence_anchor), None)
    if persistence_row is not None and qrc_top_plot_entries:
        qrc_top_sorted = sorted(
            qrc_top_plot_entries,
            key=lambda r: float(r.get("val_mape", np.inf)),
        )[: int(max(1, pqrc_sweep_topk))]
        _plot_qrc_top3_vs_persistence(
            qrc_top_sorted,
            persistence_horizon_mape=np.asarray(persistence_row["horizon_surface_mape"], dtype=float),
            out_path=out_dir_p / f"level{int(level)}_qrc_top3_mape_vs_horizon_{run_config_hash}.png",
            persistence_label=str(persistence_row.get("model", persistence_anchor)),
        )
        _plot_qrc_collapse_curves(
            qrc_top_sorted,
            out_path=out_dir_p / f"level{int(level)}_qrc_collapse_curves_{run_config_hash}.png",
        )
        _plot_qrc_feature_median_std(
            qrc_top_sorted,
            out_path=out_dir_p / f"level{int(level)}_qrc_feature_median_std_{run_config_hash}.png",
        )
        qrc_rows_eval = [r for r in eval_rows if str(r.get("model", "")).startswith("photonic_qrc")]
        best_auto = None
        if qrc_rows_eval:
            best_auto = sorted(qrc_rows_eval, key=lambda r: float(r.get("surface_mape", np.inf)))[0]
        residual_rows = [
            r
            for r in qrc_rows_eval
            if str(r.get("qrc_mode_used", r.get("qrc_mode", ""))).strip().lower() == "residual"
        ]
        best_residual = None
        if residual_rows:
            best_residual = sorted(residual_rows, key=lambda r: float(r.get("surface_mape", np.inf)))[0]
        _plot_qrc_mode_mape_vs_horizon(
            persistence_row=persistence_row,
            seasonal_row=next((r for r in eval_rows if str(r.get("model", "")).strip().lower() == "seasonal_naive_13"), None),
            qrc_best_auto=best_auto,
            qrc_best_residual=best_residual,
            out_path=out_dir_p / f"level{int(level)}_qrc_mode_mape_vs_horizon_{run_config_hash}.png",
        )
        if not collapse_df.empty:
            _plot_residual_std_ratio_vs_horizon(
                collapse_df,
                out_path=out_dir_p / f"level{int(level)}_qrc_residual_std_ratio_{run_config_hash}.png",
            )
            _plot_gate_mean_vs_horizon(
                collapse_df,
                out_path=out_dir_p / f"level{int(level)}_qrc_gate_mean_{run_config_hash}.png",
            )

    _plot_metric_bars(metrics_df, out_dir_p / f"level{int(level)}_metrics_bar_mape.png", metric="mape")
    _plot_metric_bars(metrics_df, out_dir_p / f"level{int(level)}_metrics_bar_mae.png", metric="mae")
    _plot_metric_bars(metrics_df, out_dir_p / f"level{int(level)}_metrics_bar_rmse.png", metric="rmse")
    # Keep legacy name for compatibility.
    _plot_metric_bars(metrics_df, out_dir_p / f"level{int(level)}_metrics_bar.png", metric=metric_norm)

    _plot_horizon_surface_metric(
        eval_rows,
        out_dir_p / f"level{int(level)}_horizon_surface_mape.png",
        topk_lines=int(max(1, topk_lines)),
        metric="mape",
    )
    _plot_horizon_surface_metric(
        eval_rows,
        out_dir_p / f"level{int(level)}_horizon_surface_mae.png",
        topk_lines=int(max(1, topk_lines)),
        metric="mae",
    )
    _plot_horizon_surface_metric(
        eval_rows,
        out_dir_p / f"level{int(level)}_horizon_surface_rmse.png",
        topk_lines=int(max(1, topk_lines)),
        metric="rmse",
    )
    classical_only = [r for r in eval_rows if _model_family(str(r["model"])) in {"persistence", "classical"}]
    _plot_horizon_surface_metric(
        classical_only,
        out_dir_p / f"level{int(level)}_horizon_surface_mape_classical.png",
        topk_lines=int(max(1, topk_lines)),
        metric="mape",
    )
    _plot_horizon_surface_metric(
        classical_only,
        out_dir_p / f"level{int(level)}_horizon_surface_mae_classical.png",
        topk_lines=int(max(1, topk_lines)),
        metric="mae",
    )
    _plot_horizon_surface_metric(
        classical_only,
        out_dir_p / f"level{int(level)}_horizon_surface_rmse_classical.png",
        topk_lines=int(max(1, topk_lines)),
        metric="rmse",
    )

    _plot_maturity_surface_mape(
        eval_rows,
        y_true_eval_surface,
        by_maturity=bundle.ds.by_maturity,
        out_path=out_dir_p / f"level{int(level)}_maturity_surface_mape.png",
        mape_eps=mape_eps,
        topk_lines=int(max(1, topk_lines)),
    )
    _plot_maturity_surface_mape(
        classical_only,
        y_true_eval_surface,
        by_maturity=bundle.ds.by_maturity,
        out_path=out_dir_p / f"level{int(level)}_maturity_surface_mape_classical.png",
        mape_eps=mape_eps,
        topk_lines=int(max(1, topk_lines)),
    )

    horizon_csv_default = out_dir_p / f"level{int(level)}_mape_vs_horizon.csv"
    _write_horizon_metrics(
        eval_rows,
        out_csv=horizon_csv_default,
        out_png_mape=out_dir_p / f"level{int(level)}_mape_vs_horizon.png",
        out_png_mae=out_dir_p / f"level{int(level)}_mae_vs_horizon.png",
        out_png_rmse=out_dir_p / f"level{int(level)}_rmse_vs_horizon.png",
        topk_lines=int(max(1, topk_lines)),
        out_png_all_mape=out_dir_p / f"level{int(level)}_mape_vs_horizon_all_models.png",
    )
    if horizon_csv_default.exists():
        shutil.copy2(
            horizon_csv_default,
            out_dir_p / f"level{int(level)}_mape_vs_horizon_{run_config_hash}.csv",
        )
    _plot_best_model_train_val_mape(
        eval_rows=eval_rows,
        metrics_df=metrics_df,
        out_png=out_dir_p / f"level{int(level)}_best_model_train_val_mape_vs_horizon.png",
        out_csv=out_dir_p / f"level{int(level)}_best_model_train_val_mape_vs_horizon.csv",
    )

    if bool(sanity_checks):
        run_fast_sanity_checks(
            bundle=bundle,
            eval_rows=eval_rows,
            y_true_test_surface=y_true_eval_surface,
            lookback=int(L),
            out_dir=out_dir_p,
            level=int(level),
            eps=float(mape_eps),
            split="val",
        )

    duplicate_hashes = _write_hash_logs(
        eval_rows,
        out_path=out_dir_p / "model_hashes.jsonl",
        seed=seed,
        level=level,
        backend=backend,
        mape_eps=mape_eps,
    )

    if duplicate_hashes:
        by_model_family: list[list[str]] = []
        for models in duplicate_hashes.values():
            fams = {_model_family(m) for m in models}
            if len(fams) > 1:
                by_model_family.append(sorted(models))
        if by_model_family:
            print(f"[model_compare][DUPLICATE WARNING] cross-family duplicate predictions: {by_model_family}")
        photonic_dups = []
        for models in duplicate_hashes.values():
            s = set(models)
            has_fb = "photonic_qrc_feedback" in s
            has_nf = "photonic_qrc_no_feedback" in s
            has_mem_fb = "photonic_memory" in s
            has_mem_nf = "photonic_memory_no_feedback" in s
            has_persist = ("persistence_naive" in s) or (PERSISTENCE_SURFACE_NAME in s)
            if (
                (has_fb and has_nf)
                or ((has_fb or has_nf) and has_persist)
                or (has_mem_fb and has_mem_nf)
                or ((has_mem_fb or has_mem_nf) and has_persist)
            ):
                photonic_dups.append(sorted(models))
        if photonic_dups and not bool(allow_duplicate_hashes):
            warnings.warn(
                "Photonic duplicate predictions detected with persistence/no-feedback variant. "
                "Runs are retained but marked collapse_failed; adjust pqrc settings (shots/input_scale/min_residual_scale). "
                f"duplicates={photonic_dups}",
                RuntimeWarning,
            )
        ported_dups = [sorted(models) for models in duplicate_hashes.values() if sum(m.startswith("ported_") for m in models) > 1]
        if ported_dups and not bool(allow_duplicate_hashes):
            raise RuntimeError(
                "Duplicate prediction hashes detected across distinct ported models. "
                f"Set --allow_duplicate_hashes to bypass. duplicates={ported_dups}"
            )

    return metrics_df, out_dir_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model performance and generate evaluation plots.")
    parser.add_argument("--data_dir", type=str, default="Quandela/Challenge_Swaptions")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--backend", type=str, choices=["sim", "qpu"], default="sim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--model", type=str, default="", help="Run one classical model name only.")
    parser.add_argument(
        "--classical_models",
        type=str,
        default=(
            "persistence_surface_naive,persistence_naive,factor_ar,mlp,gru,lstm,reservoir,"
            "photonic_qrc_feedback,photonic_qrc_no_feedback,photonic_memory,photonic_memory_no_feedback,persist_qrc_weak"
        ),
        help=(
            "Comma-separated classical models. Available: persistence_surface_naive,persistence_naive,factor_ar,mlp,gru,lstm,reservoir,"
            "photonic_qrc_feedback,photonic_qrc_no_feedback,photonic_memory,photonic_memory_no_feedback,persist_qrc_weak"
        ),
    )
    parser.add_argument("--skip_quantum", action="store_true")
    parser.add_argument("--skip_ported", action="store_true")
    parser.add_argument("--skip_classical", action="store_true")
    parser.add_argument("--run_quantum", action="store_true", help="Alias to force quantum runs (overrides --skip_quantum).")
    parser.add_argument("--run_ported", action="store_true", help="Alias to force ported runs (overrides --skip_ported).")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--metric", type=str, choices=["mape", "mae", "rmse", "smape", "wape"], default="mape")
    parser.add_argument(
        "--metrics",
        type=str,
        default="mape,mae,rmse,smape,wape",
        help="Comma-separated metrics to compute/report from {mape,mae,rmse,smape,wape}.",
    )
    parser.add_argument("--mape_eps", type=float, default=1e-8)
    parser.add_argument("--debug_positive_checks", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lookback", type=int, default=0, help="Lookback window in observed samples (0 uses config).")
    parser.add_argument("--sanity_checks", type=int, choices=[0, 1], default=1)
    parser.add_argument("--topk_lines", type=int, default=8)
    parser.add_argument("--forecast_horizons", type=int, default=6)
    parser.add_argument("--use_cycle_phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quantum_feature_mode",
        type=str,
        choices=["concrete_features", "q_featuremap_features", "q_kernel_gram"],
        default="concrete_features",
    )
    parser.add_argument("--q_dim", type=int, default=64)
    parser.add_argument("--q_shots", type=int, default=256)
    parser.add_argument("--q_angle_scale", type=float, default=1.0)
    parser.add_argument("--q_seed", type=int, default=0)
    parser.add_argument("--w_min", type=float, default=0.05)
    parser.add_argument("--min_qfeat_std", type=float, default=1e-8)
    parser.add_argument("--allow_duplicate_hashes", action="store_true")
    parser.add_argument("--pqrc_M", type=int, default=8)
    parser.add_argument("--pqrc_modes", type=int, default=0, help="Alias for photonic mode count (0 uses --pqrc_M).")
    parser.add_argument("--pqrc_Nph", type=int, default=2)
    parser.add_argument("--pqrc_nphotons", type=int, default=None, help="Alias for --pqrc_Nph.")
    parser.add_argument("--pqrc_budget", type=int, default=32)
    parser.add_argument("--pqrc_gain", type=float, default=None)
    parser.add_argument("--pqrc_input_scale", type=float, default=None)
    parser.add_argument("--pqrc_feature", type=str, choices=["coincidence", "clickprob", "pseudo_coincidence"], default="clickprob")
    parser.add_argument("--pqrc_pseudocount", type=float, default=0.25, help="Additive pseudocount smoothing for shot-derived feature histograms (0 disables).")
    parser.add_argument("--pqrc_higher_order", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--pqrc_budget_alloc", type=str, choices=["uniform", "weighted"], default="uniform")
    parser.add_argument("--pqrc_shots", type=int, default=32)
    parser.add_argument("--pqrc_ridge", type=float, default=None)
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
    parser.add_argument("--qrc_residual_penalty", type=float, default=0.1)
    parser.add_argument(
        "--qrc_sweep_objective",
        type=str,
        choices=["surface_mape", "surface_wape", "surface_mae", "surface_rmse", "mape", "mc_then_mape"],
        default="surface_mape",
    )
    parser.add_argument(
        "--sweep_objective",
        type=str,
        choices=["surface_mape", "surface_wape", "surface_mae", "surface_rmse", "mape", "mc_then_mape"],
        default="",
        help="Alias for --qrc_sweep_objective (surface-based objective for sweep selection).",
    )
    parser.add_argument(
        "--performance_preset",
        action="store_true",
        help="Use higher-capacity photonic defaults when knobs are not explicitly set (modes=20, nphotons=10).",
    )
    parser.add_argument("--pqrc_sweep", action="store_true", help="Run deterministic photonic-QRC hyperparameter coordinate sweep.")
    parser.add_argument("--qrc_load_from", type=str, default="", help="Replay exact QRC config from best_config.json or sweep directory.")
    parser.add_argument("--qrc_model_hash", type=str, default="", help="Optional model hash selector when qrc_load_from points to sweep history.")
    parser.add_argument(
        "--qrc_optimal_guard",
        type=int,
        choices=[0, 1],
        default=1,
        help="When replaying qrc_load_from, enforce selection of the best sweep config for the chosen objective.",
    )
    parser.add_argument(
        "--allow_nonoptimal_qrc_hash",
        action="store_true",
        help="Allow --qrc_model_hash replay even when it is not the best sweep row (disables strict guard for hash override).",
    )
    parser.add_argument("--sweep_modes", action="store_true", help="Allow modes grid search during pqrc_sweep (otherwise locked).")
    parser.add_argument("--sweep_gain", action="store_true", help="Allow gain grid search during pqrc_sweep (otherwise locked).")
    parser.add_argument("--sweep_input_scale", action="store_true", help="Allow input_scale grid search during pqrc_sweep (otherwise locked).")
    parser.add_argument("--sweep_ridge", action="store_true", help="Allow ridge grid search during pqrc_sweep (otherwise locked).")
    parser.add_argument("--pqrc_sweep_subset", type=int, default=256, help="Sweep subset size (number of train windows).")
    parser.add_argument("--pqrc_sweep_topk", type=int, default=3, help="Number of top swept photonic configs for reporting/plots.")
    parser.add_argument(
        "--quantum_bases",
        type=str,
        default="mlp,gru,lstm,factor_ar,reservoir,persistence_surface_naive",
        help="Comma-separated classical backbones for quantum residual annotators.",
    )
    parser.add_argument(
        "--quantum_modes",
        type=str,
        default="q_bottleneck,qrc",
        help="Comma-separated quantum modes from {q_bottleneck,qrc}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.strip() if args.out_dir else ""
    if not out_dir:
        out_dir = str(Path("results") / f"model_compare_level{int(args.level)}")

    quantum_bases = [t.strip().lower() for t in str(args.quantum_bases).split(",") if t.strip()]
    quantum_modes = [t.strip().lower() for t in str(args.quantum_modes).split(",") if t.strip()]
    classical_models = [t.strip().lower() for t in str(args.classical_models).split(",") if t.strip()]
    include_quantum = (not bool(args.skip_quantum)) or bool(args.run_quantum)
    include_ported = (not bool(args.skip_ported)) or bool(args.run_ported)
    sweep_modes_eff = bool(args.sweep_modes) or (int(args.pqrc_modes) <= 0)
    sweep_gain_eff = bool(args.sweep_gain) or (args.pqrc_gain is None)
    sweep_input_scale_eff = bool(args.sweep_input_scale) or (args.pqrc_input_scale is None)
    sweep_ridge_eff = bool(args.sweep_ridge) or (args.pqrc_ridge is None)
    sweep_objective_eff = str(args.sweep_objective).strip() or str(args.qrc_sweep_objective)
    pqrc_nphotons_eff = None if args.pqrc_nphotons is None else int(args.pqrc_nphotons)
    metrics_df, out_dir_p = run_comparison(
        data_dir=args.data_dir,
        level=int(args.level),
        backend=str(args.backend),
        seed=int(args.seed),
        out_dir=out_dir,
        include_quantum=bool(include_quantum),
        include_ported=bool(include_ported),
        include_classical=not bool(args.skip_classical),
        model=(str(args.model).strip().lower() or None),
        classical_models=classical_models,
        fast=bool(args.fast),
        lookback=(int(args.lookback) if int(args.lookback) > 0 else None),
        sanity_checks=bool(int(args.sanity_checks)),
        quantum_bases=quantum_bases,
        quantum_modes=quantum_modes,
        use_cycle_phase=bool(args.use_cycle_phase),
        forecast_horizons=int(args.forecast_horizons),
        metric=str(args.metric),
        metrics=_parse_metrics_arg(str(args.metrics), default_metric=str(args.metric)),
        topk_lines=int(args.topk_lines),
        mape_eps=float(args.mape_eps),
        quantum_feature_mode=str(args.quantum_feature_mode),
        q_dim=int(args.q_dim),
        q_shots=int(args.q_shots),
        q_angle_scale=float(args.q_angle_scale),
        q_seed=int(args.q_seed),
        w_min=float(args.w_min),
        min_qfeat_std=float(args.min_qfeat_std),
        allow_duplicate_hashes=bool(args.allow_duplicate_hashes),
        pqrc_M=int(args.pqrc_M),
        pqrc_modes=(int(args.pqrc_modes) if int(args.pqrc_modes) > 0 else None),
        pqrc_Nph=int(args.pqrc_Nph),
        pqrc_nphotons=pqrc_nphotons_eff,
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
        qrc_load_from=(str(args.qrc_load_from).strip() or None),
        qrc_model_hash=(str(args.qrc_model_hash).strip() or None),
        qrc_optimal_guard=bool(int(args.qrc_optimal_guard)),
        allow_nonoptimal_qrc_hash=bool(args.allow_nonoptimal_qrc_hash),
        qrc_sweep_objective=str(sweep_objective_eff),
        sweep_objective=str(sweep_objective_eff),
        pqrc_sweep=bool(args.pqrc_sweep),
        sweep_modes=bool(sweep_modes_eff),
        sweep_gain=bool(sweep_gain_eff),
        sweep_input_scale=bool(sweep_input_scale_eff),
        sweep_ridge=bool(sweep_ridge_eff),
        pqrc_sweep_subset=int(args.pqrc_sweep_subset),
        pqrc_sweep_topk=int(args.pqrc_sweep_topk),
        performance_preset=bool(args.performance_preset),
        debug_positive_checks=bool(int(args.debug_positive_checks)),
    )

    print("Saved comparison outputs:")
    print(out_dir_p / f"level{int(args.level)}_metrics_summary.csv")
    print(out_dir_p / f"level{int(args.level)}_metrics_bar_mape.png")
    print(out_dir_p / f"level{int(args.level)}_metrics_bar_mae.png")
    print(out_dir_p / f"level{int(args.level)}_metrics_bar_rmse.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_mape.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_mae.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_rmse.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_mape_classical.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_mae_classical.png")
    print(out_dir_p / f"level{int(args.level)}_horizon_surface_rmse_classical.png")
    print(out_dir_p / f"level{int(args.level)}_maturity_surface_mape.png")
    print(out_dir_p / f"level{int(args.level)}_maturity_surface_mape_classical.png")
    print(out_dir_p / f"level{int(args.level)}_mape_vs_horizon.csv")
    print(out_dir_p / f"level{int(args.level)}_mape_vs_horizon.png")
    print(out_dir_p / f"level{int(args.level)}_best_model_train_val_mape_vs_horizon.csv")
    print(out_dir_p / f"level{int(args.level)}_best_model_train_val_mape_vs_horizon.png")
    print(out_dir_p / f"level{int(args.level)}_mae_vs_horizon.png")
    print(out_dir_p / f"level{int(args.level)}_rmse_vs_horizon.png")
    print(out_dir_p / f"level{int(args.level)}_mape_vs_horizon_all_models.png")
    print(out_dir_p / f"level{int(args.level)}_qrc_collapse_diag.csv")
    print(out_dir_p / f"level{int(args.level)}_qrc_feature_stats.csv")
    print(out_dir_p / f"level{int(args.level)}_qrc_feature_sanity.csv")
    print(out_dir_p / f"level{int(args.level)}_qrc_memory_proxy.csv")
    print(out_dir_p / f"level{int(args.level)}_qrc_feedback_sanity.csv")
    print(out_dir_p / f"level{int(args.level)}_sanity_report.json")
    print(out_dir_p / "model_hashes.jsonl")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
