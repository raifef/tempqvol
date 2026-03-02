from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .constraints import validate_constraints
from .data_swaptions import FactorZScoreNormalizer, prepare_windowed_dataset
from .make_submission import infer_submission_horizon, make_submission_from_predictions


def _resolve_surface_pca() -> Any:
    try:
        from aaquandela.qml.utils_surface import fit_surface_pca
    except Exception:  # pragma: no cover - fallback for alternate PYTHONPATH layouts
        from qml.utils_surface import fit_surface_pca  # type: ignore
    return fit_surface_pca


@dataclass
class PreprocessedBundle:
    ds: Any
    pca: Any
    normalizer: Any
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    x_train_norm: np.ndarray
    x_val_norm: np.ndarray
    x_test_norm: np.ndarray
    y_train_norm: np.ndarray
    y_val_norm: np.ndarray
    y_test_norm: np.ndarray
    missing_train_norm: np.ndarray
    missing_val_norm: np.ndarray
    missing_test_norm: np.ndarray
    missing_future_norm: np.ndarray
    y_train_surface_scaled: np.ndarray
    y_val_surface_scaled: np.ndarray
    y_test_surface_scaled: np.ndarray
    last_window: np.ndarray
    last_window_norm: np.ndarray
    price_scale: float
    clip_cap: float
    factor_dim: int
    use_cycle_phase: bool


def sanitize_surface_matrix(arr: np.ndarray, cap_hint: float | None = None) -> tuple[np.ndarray, float]:
    clean = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0)
    if cap_hint is None:
        p999 = float(np.nanpercentile(clean, 99.9))
        cap = max(500.0, min(1e6, p999 * 10.0))
    else:
        cap = float(cap_hint)
    clean = np.clip(clean, 0.0, cap)
    return clean, cap


def _load_submission_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")

    def _coerce_scalar(token: str) -> Any:
        t = token.strip()
        if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
            t = t[1:-1]
        low = t.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in {"none", "null"}:
            return None
        try:
            if t.startswith("0") and len(t) > 1 and t[1].isdigit():
                raise ValueError
            return int(t)
        except Exception:
            pass
        try:
            return float(t)
        except Exception:
            return t

    def _parse_simple_yaml(yaml_text: str) -> dict[str, Any]:
        root: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
        for raw_line in yaml_text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            if not stack:
                stack = [(-1, root)]
            parent = stack[-1][1]

            if value == "":
                node: dict[str, Any] = {}
                parent[key] = node
                stack.append((indent, node))
            else:
                parent[key] = _coerce_scalar(value)
        return root

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return _parse_simple_yaml(text)


def _level_cfg(config: dict[str, Any], level: int) -> dict[str, Any]:
    k = f"level{int(level)}"
    out = config.get(k, {})
    return out if isinstance(out, dict) else {}


def _future_dates(data_dir: str | Path, ds: Any, H: int) -> list[str]:
    data_dir_p = Path(data_dir)
    sample_path = data_dir_p / "sample_submission.csv"
    if sample_path.exists():
        try:
            sample_df = pd.read_csv(sample_path)
            if "Date" in sample_df.columns and sample_df.shape[0] >= int(H):
                return sample_df["Date"].astype(str).iloc[: int(H)].tolist()
        except Exception:
            pass

    last_date = pd.to_datetime(str(ds.dates[-1]), errors="coerce")
    if pd.isna(last_date):
        return [f"T+{i + 1}" for i in range(int(H))]
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=int(H))
    return [d.strftime("%Y-%m-%d") for d in dates]


def _compute_regime_score(x_seq_norm: np.ndarray, missing_seq_norm: np.ndarray, source: str = "volatility") -> np.ndarray:
    source_norm = str(source).strip().lower()
    if source_norm == "missingness":
        # Use last-day normalized missingness intensity.
        return np.mean(np.abs(missing_seq_norm[:, -1, :]), axis=1)
    # Default: rolling volatility over recent factor window.
    recent = x_seq_norm[:, -min(5, x_seq_norm.shape[1]) :, :]
    return np.std(recent, axis=(1, 2))


def _best_blend_weight(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_residual: np.ndarray,
    grid: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> tuple[float, float]:
    best_w = 0.0
    best_mae = float("inf")
    for w in grid:
        pred = y_base + float(w) * y_residual
        mae = float(np.mean(np.abs(pred - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return float(np.clip(best_w, 0.05, 0.95)), best_mae


def _compute_metric_block(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    err = y_pred - y_true
    den = np.maximum(np.abs(y_true), float(max(1e-12, eps)))
    smape_den = np.abs(y_true) + np.abs(y_pred) + float(max(1e-12, eps))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mape": float(np.mean(np.abs(err) / den) * 100.0),
        "smape": float(np.mean(2.0 * np.abs(err) / smape_den) * 100.0),
        "wape": float((np.sum(np.abs(err)) / max(np.sum(np.abs(y_true) + float(max(1e-12, eps))), float(max(1e-12, eps)))) * 100.0),
    }


def _horizon_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, eps: float = 1e-8) -> list[float]:
    h = int(y_true.shape[1])
    out: list[float] = []
    m = str(metric).strip().lower()
    for k in range(h):
        yt = y_true[:, k, :]
        yp = y_pred[:, k, :]
        err = yp - yt
        if m == "mae":
            val = float(np.mean(np.abs(err)))
        elif m == "rmse":
            val = float(np.sqrt(np.mean(err * err)))
        elif m == "mape":
            den = np.maximum(np.abs(yt), float(max(1e-12, eps)))
            val = float(np.mean(np.abs(err) / den) * 100.0)
        elif m == "smape":
            den = np.abs(yt) + np.abs(yp) + float(max(1e-12, eps))
            val = float(np.mean(2.0 * np.abs(err) / den) * 100.0)
        elif m == "wape":
            num = float(np.sum(np.abs(err)))
            den = float(np.sum(np.abs(yt) + float(max(1e-12, eps))))
            val = float((num / max(den, float(max(1e-12, eps)))) * 100.0)
        else:
            raise ValueError(f"Unsupported horizon metric: {metric}")
        out.append(val)
    return out


def _to_surface_space(bundle: PreprocessedBundle, z_pred_norm: np.ndarray) -> np.ndarray:
    z_pred = bundle.normalizer.inverse_transform_seq(z_pred_norm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        surf_scaled = bundle.pca.inverse_transform_seq(z_pred)
    surf = np.clip(surf_scaled * bundle.price_scale, 0.0, bundle.clip_cap)
    return np.nan_to_num(surf, nan=0.0, posinf=bundle.clip_cap, neginf=0.0)


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


def load_and_preprocess(
    data_dir: str | Path,
    level: int,
    L: int,
    H: int,
    imputer: str,
    d_factors: int = 8,
    seed: int = 0,
    use_cycle_phase: bool = True,
) -> PreprocessedBundle:
    """Load challenge data and return factor-space train/val/test tensors."""
    from sklearn.preprocessing import StandardScaler

    fit_surface_pca = _resolve_surface_pca()
    with warnings.catch_warnings():
        # Known benign warnings on some BLAS/sklearn builds during randomized SVD.
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*encountered in matmul.*",
        )
        ds = prepare_windowed_dataset(
            data_dir=data_dir,
            level=int(level),
            lookback=int(L),
            horizon=int(H),
            imputer=str(imputer),
            seed=int(seed),
            use_cycle_phase=bool(use_cycle_phase),
        )

    filled_clean, cap = sanitize_surface_matrix(ds.filled)
    target_clean, _ = sanitize_surface_matrix(ds.target, cap_hint=cap)
    price_scale = max(1.0, float(np.nanpercentile(filled_clean, 95)))
    filled_scaled = filled_clean / price_scale
    target_scaled = target_clean / price_scale
    # Stabilize PCA numerics on occasional heavy tails.
    filled_scaled = np.clip(np.nan_to_num(filled_scaled, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
    target_scaled = np.clip(np.nan_to_num(target_scaled, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)

    train_last_day = min(filled_scaled.shape[0] - 1, ds.train_slice.stop + int(L) + int(H) - 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        pca = fit_surface_pca(filled_scaled[: train_last_day + 1], d_factors=int(d_factors), seed=int(seed))
        z_inputs_all = pca.transform(filled_scaled)
        z_target_all = pca.transform(target_scaled)
    z_inputs_all = np.clip(np.nan_to_num(z_inputs_all, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    z_target_all = np.clip(np.nan_to_num(z_target_all, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    d = z_inputs_all.shape[1]
    n_windows = ds.starts.shape[0]
    n_surface = filled_scaled.shape[1]

    x_z = np.zeros((n_windows, int(L), d), dtype=float)
    y_z = np.zeros((n_windows, int(H), d), dtype=float)
    y_surface_scaled = np.zeros((n_windows, int(H), n_surface), dtype=float)
    for i, start in enumerate(ds.starts):
        x_z[i] = z_inputs_all[start : start + int(L)]
        y_z[i] = z_target_all[start + int(L) : start + int(L) + int(H)]
        y_surface_scaled[i] = target_scaled[start + int(L) : start + int(L) + int(H)]

    x_train_factor = x_z[ds.train_slice]
    x_val_factor = x_z[ds.val_slice]
    x_test_factor = x_z[ds.test_slice]
    y_train = y_z[ds.train_slice]
    y_val = y_z[ds.val_slice]
    y_test = y_z[ds.test_slice]
    y_train_surface_scaled = y_surface_scaled[ds.train_slice]
    y_val_surface_scaled = y_surface_scaled[ds.val_slice]
    y_test_surface_scaled = y_surface_scaled[ds.test_slice]

    normalizer = FactorZScoreNormalizer().fit(x_train_factor.reshape(-1, d))
    x_train_n = normalizer.transform_seq(x_train_factor)
    x_val_n = normalizer.transform_seq(x_val_factor)
    x_test_n = normalizer.transform_seq(x_test_factor)
    y_train_n = normalizer.transform_seq(y_train)
    y_val_n = normalizer.transform_seq(y_val)
    y_test_n = normalizer.transform_seq(y_test)

    m_train = ds.missing_summary_seq[ds.train_slice]
    m_val = ds.missing_summary_seq[ds.val_slice]
    m_test = ds.missing_summary_seq[ds.test_slice]
    m_future = ds.missing_summary_daily[-int(L) :][None, :, :]
    m_dim = m_train.shape[-1]
    m_scaler = StandardScaler().fit(m_train.reshape(-1, m_dim))
    m_train_n = m_scaler.transform(m_train.reshape(-1, m_dim)).reshape(m_train.shape)
    m_val_n = m_scaler.transform(m_val.reshape(-1, m_dim)).reshape(m_val.shape)
    m_test_n = m_scaler.transform(m_test.reshape(-1, m_dim)).reshape(m_test.shape)
    m_future_n = m_scaler.transform(m_future.reshape(-1, m_dim)).reshape(m_future.shape)[0]

    # Temporal features include dt_days and optional deterministic cycle-phase encodings.
    t_train = ds.temporal_seq[ds.train_slice]
    t_val = ds.temporal_seq[ds.val_slice]
    t_test = ds.temporal_seq[ds.test_slice]
    t_future = ds.temporal_daily[-int(L) :][None, :, :]
    t_dim = int(t_train.shape[-1])
    t_scaler = StandardScaler().fit(t_train.reshape(-1, t_dim))
    t_train_n = t_scaler.transform(t_train.reshape(-1, t_dim)).reshape(t_train.shape)
    t_val_n = t_scaler.transform(t_val.reshape(-1, t_dim)).reshape(t_val.shape)
    t_test_n = t_scaler.transform(t_test.reshape(-1, t_dim)).reshape(t_test.shape)
    t_future_n = t_scaler.transform(t_future.reshape(-1, t_dim)).reshape(t_future.shape)[0]

    # Final model inputs: factor history + temporal phase/dt features.
    x_train_model_n = np.concatenate([x_train_n, t_train_n], axis=2)
    x_val_model_n = np.concatenate([x_val_n, t_val_n], axis=2)
    x_test_model_n = np.concatenate([x_test_n, t_test_n], axis=2)

    last_window = z_inputs_all[-int(L) :]
    last_window_factor_n = normalizer.transform_seq(last_window[None, :, :])[0]
    last_window_n = np.concatenate([last_window_factor_n, t_future_n], axis=1)

    return PreprocessedBundle(
        ds=ds,
        pca=pca,
        normalizer=normalizer,
        x_train=np.concatenate([x_train_factor, t_train], axis=2),
        x_val=np.concatenate([x_val_factor, t_val], axis=2),
        x_test=np.concatenate([x_test_factor, t_test], axis=2),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        x_train_norm=x_train_model_n,
        x_val_norm=x_val_model_n,
        x_test_norm=x_test_model_n,
        y_train_norm=y_train_n,
        y_val_norm=y_val_n,
        y_test_norm=y_test_n,
        missing_train_norm=m_train_n,
        missing_val_norm=m_val_n,
        missing_test_norm=m_test_n,
        missing_future_norm=m_future_n,
        y_train_surface_scaled=y_train_surface_scaled,
        y_val_surface_scaled=y_val_surface_scaled,
        y_test_surface_scaled=y_test_surface_scaled,
        last_window=last_window,
        last_window_norm=last_window_n,
        price_scale=price_scale,
        clip_cap=cap,
        factor_dim=int(d),
        use_cycle_phase=bool(use_cycle_phase),
    )


def run_end_to_end(
    data_dir: str | Path,
    level: int,
    out_csv: str | Path,
    backend: str = "sim",
    seed: int = 0,
    use_cycle_phase: bool | None = None,
    classical_kind: str | None = None,
    lookback: int | None = None,
    pqrc_M: int = 12,
    pqrc_modes: int | None = None,
    pqrc_Nph: int = 2,
    pqrc_nphotons: int | None = None,
    pqrc_budget: int = 32,
    pqrc_gain: float = 0.5,
    pqrc_feature: str = "coincidence",
    pqrc_pseudocount: float = 0.5,
    pqrc_higher_order: int = 1,
    pqrc_budget_alloc: str = "uniform",
    pqrc_shots: int = 32,
    pqrc_ridge: float = 1e-2,
    pqrc_in_pca: int = 1,
    pqrc_factor_cap: int = 0,
    qrc_mode: str = "auto",
    qrc_target: str = "delta",
    qrc_baseline: str = "persistence",
    target_transform: str = "log",
    y_floor_mode: str = "train_p001",
    y_floor_value: float | None = None,
    persist_qrc_alpha: float = 0.25,
    persist_qrc_rclip: float = 0.05,
    persist_qrc_drive_scale: float = 0.25,
    tau: float | None = None,
    qrc_gate_tau: float = 0.05,
    qrc_resid_clip: float | None = None,
    qrc_residvar_penalty: float = 0.1,
    qrc_feat_norm: str = "standard",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .feature_selection import ridge_validation_mae_score, wrapper_forward_selection
    from .models_classical import train_classical_forecaster
    from .models_quantum import QuantumAnnotatorResidual

    cfg = _load_submission_config(Path(__file__).with_name("config.yaml"))
    lcfg = _level_cfg(cfg, int(level))
    qcfg = lcfg.get("quantum", {}) if isinstance(lcfg.get("quantum", {}), dict) else {}
    gcfg = lcfg.get("gating", {}) if isinstance(lcfg.get("gating", {}), dict) else {}

    L_cfg = int(lcfg.get("L", 20))
    L = int(lookback) if (lookback is not None and int(lookback) > 0) else L_cfg
    H_cfg = int(lcfg.get("H", 10))
    H = infer_submission_horizon(data_dir=data_dir, default_h=H_cfg)
    if L != L_cfg:
        print(f"[submission] overriding lookback from config L={L_cfg} to lookback={L}.")
    if H != H_cfg:
        print(f"[submission] overriding horizon from config H={H_cfg} to sample_submission rows H={H}.")
    gate_tau = float(qrc_gate_tau if tau is None else tau)
    if str(qrc_target).strip().lower() == "norm_resid" and tau is None and float(qrc_gate_tau) == 0.05:
        gate_tau = 0.5
    imputer = str(lcfg.get("imputer", "ffill_interp" if int(level) == 1 else "svd_iterative"))
    d_factors = int(lcfg.get("d_factors", 8))
    use_cycle_phase_cfg = bool(lcfg.get("use_cycle_phase", True))
    use_cycle_phase_final = use_cycle_phase_cfg if use_cycle_phase is None else bool(use_cycle_phase)
    classical_kind_cfg = str(lcfg.get("classical_kind", "mlp"))
    classical_kind = str(classical_kind_cfg if classical_kind is None else classical_kind).strip().lower()
    quantum_mode = str(qcfg.get("mode", "q_bottleneck"))
    shots = int(max(1, int(qcfg.get("shots", 32))))
    qrc_stride = int(qcfg.get("qrc_stride", 1))
    qrc_feature_dim = int(qcfg.get("qrc_feature_dim", 24))
    train_qeval_budget = int(qcfg.get("train_qeval_budget", 5000))
    batch_size = int(qcfg.get("batch_size", 32))
    gate_enabled = bool(gcfg.get("enabled", True))
    gate_source = str(gcfg.get("source", "volatility"))
    gate_quantile = float(gcfg.get("quantile", 0.5))

    # Optional fast path for smoke tests (keeps default runs unchanged).
    if os.environ.get("SUBMISSION_SMOKE_FAST", "0") == "1":
        imputer = "ffill_interp"
        d_factors = min(d_factors, 6)
        classical_kind = "mlp"
        if int(level) == 2:
            quantum_mode = "q_bottleneck"
        shots = max(1, min(shots, 16))
        train_qeval_budget = min(train_qeval_budget, 800)
        qrc_stride = max(qrc_stride, 4)
        qrc_feature_dim = min(qrc_feature_dim, 12)
        batch_size = max(batch_size, 64)
        gate_enabled = False
        print("[submission] SUBMISSION_SMOKE_FAST=1: using faster smoke-test settings.")

    pqrc_shots_eff = int(pqrc_shots)
    if pqrc_shots_eff <= 0:
        pqrc_shots_eff = int(max(1, int(qcfg.get("shots", 32))))
        print(
            f"[submission] non-positive pqrc_shots={int(pqrc_shots)} received; "
            f"using pqrc_shots={pqrc_shots_eff}."
        )

    bundle = load_and_preprocess(
        data_dir=data_dir,
        level=int(level),
        L=L,
        H=H,
        imputer=imputer,
        d_factors=d_factors,
        seed=int(seed),
        use_cycle_phase=bool(use_cycle_phase_final),
    )
    train_fit_slice, train_tune_slice = _train_fit_tune_slices(int(bundle.y_train_norm.shape[0]))
    print(
        f"[submission] internal_train_split fit={int(train_fit_slice.stop - train_fit_slice.start)} "
        f"tune={int(train_tune_slice.stop - train_tune_slice.start)} "
        "(external validation kept for evaluation only)."
    )

    # Cadence-aware augmented input uses factors + missingness summaries.
    use_augmented_input = str(classical_kind).strip().lower() in {
        "reservoir",
        "photonic_qrc_feedback",
        "photonic_qrc_no_feedback",
        "persist_qrc_weak",
        "photonic_memory",
        "photonic_memory_no_feedback",
    }
    if use_augmented_input:
        x_train_classical = np.concatenate([bundle.x_train_norm, bundle.missing_train_norm], axis=2)
        x_val_classical = np.concatenate([bundle.x_val_norm, bundle.missing_val_norm], axis=2)
        x_test_classical = np.concatenate([bundle.x_test_norm, bundle.missing_test_norm], axis=2)
        future_in = np.concatenate([bundle.last_window_norm[None, :, :], bundle.missing_future_norm[None, :, :]], axis=2)
    else:
        x_train_classical = bundle.x_train_norm
        x_val_classical = bundle.x_val_norm
        x_test_classical = bundle.x_test_norm
        future_in = bundle.last_window_norm[None, :, :]

    pqrc_nphotons_eff = int(pqrc_nphotons if pqrc_nphotons is not None else pqrc_Nph)
    classical = train_classical_forecaster(
        x_train=x_train_classical,
        y_train=bundle.y_train_norm,
        x_val=x_val_classical,
        y_val=bundle.y_val_norm,
        kind=classical_kind,  # type: ignore[arg-type]
        seed=int(seed),
        pqrc_M=int(np.clip(int(pqrc_M if pqrc_modes is None else pqrc_modes), 4, 20)),
        pqrc_modes=int(np.clip(int(pqrc_M if pqrc_modes is None else pqrc_modes), 4, 20)),
        pqrc_Nph=int(pqrc_nphotons_eff),
        pqrc_nphotons=int(pqrc_nphotons_eff),
        pqrc_budget=int(pqrc_budget),
        pqrc_gain=float(pqrc_gain),
        pqrc_feature=str(pqrc_feature),
        pqrc_pseudocount=float(pqrc_pseudocount),
        pqrc_higher_order=int(pqrc_higher_order),
        pqrc_budget_alloc=str(pqrc_budget_alloc),
        pqrc_shots=int(pqrc_shots_eff),
        pqrc_ridge=float(pqrc_ridge),
        pqrc_in_pca=int(pqrc_in_pca),
        pqrc_factor_cap=int(pqrc_factor_cap),
        qrc_mode=str(qrc_mode),
        qrc_target=str(qrc_target),
        qrc_baseline=str(qrc_baseline),
        target_transform=str(target_transform),
        y_floor_mode=str(y_floor_mode),
        y_floor_value=(None if y_floor_value is None else float(y_floor_value)),
        persist_qrc_alpha=float(persist_qrc_alpha),
        persist_qrc_rclip=float(persist_qrc_rclip),
        persist_qrc_drive_scale=float(persist_qrc_drive_scale),
        tau=float(gate_tau),
        qrc_gate_tau=float(gate_tau),
        qrc_resid_clip=(None if qrc_resid_clip is None else float(qrc_resid_clip)),
        qrc_residvar_penalty=float(qrc_residvar_penalty),
        qrc_feat_norm=str(qrc_feat_norm),
    )

    base_train = classical.predict(x_train_classical)
    base_val = classical.predict(x_val_classical)
    base_test = classical.predict(x_test_classical)
    base_future = classical.predict(future_in)

    # Candidate quantum inputs: PCA factors + missingness summaries.
    cand_train = np.concatenate([bundle.x_train_norm, bundle.missing_train_norm], axis=2)
    cand_val = np.concatenate([bundle.x_val_norm, bundle.missing_val_norm], axis=2)
    cand_test = np.concatenate([bundle.x_test_norm, bundle.missing_test_norm], axis=2)
    cand_future = np.concatenate([bundle.last_window_norm[None, :, :], bundle.missing_future_norm[None, :, :]], axis=2)

    backend_cap = 20 if str(backend).lower() == "sim" else 24
    merlin_cap = 20
    mode_cap = min(backend_cap, merlin_cap)
    total_dim = int(cand_train.shape[-1])
    selected_idx: list[int]
    if total_dim > mode_cap:
        residual_target_train = (bundle.y_train_norm - base_train).reshape(base_train.shape[0], -1)
        selected_idx = wrapper_forward_selection(
            features=cand_train[:, -1, :],
            target=residual_target_train,
            max_k=mode_cap,
            score_fn=lambda f, t: ridge_validation_mae_score(f, t, val_fraction=0.2, seed=int(seed)),
        )
    else:
        selected_idx = list(range(total_dim))
        print(f"[feature_selection] skipped, using all features. final_dim={len(selected_idx)} <= cap={mode_cap}")

    if len(selected_idx) == 0:
        raise RuntimeError("Feature selection produced an empty feature set.")
    final_dim = len(selected_idx)
    print(f"[feature_selection] selected_feature_indices={selected_idx}")
    print(f"[feature_selection] final_dim={final_dim} mode_cap={mode_cap}")
    validate_constraints(input_dim=final_dim, backend=backend)

    q_train = cand_train[:, :, selected_idx]
    q_train_fit = q_train[train_fit_slice]
    q_tune = q_train[train_tune_slice]
    q_val = cand_val[:, :, selected_idx]
    q_test = cand_test[:, :, selected_idx]
    q_future = cand_future[:, :, selected_idx]

    q_annotator = QuantumAnnotatorResidual(
        input_dim=final_dim,
        mode=quantum_mode,
        shots=shots,
        qrc_stride=qrc_stride,
        qrc_feature_dim=qrc_feature_dim,
        train_qeval_budget=train_qeval_budget,
        batch_size=batch_size,
        seed=int(seed),
    )
    q_annotator.fit(
        x_train=q_train_fit,
        y_true_train=bundle.y_train_norm[train_fit_slice],
        y_classical_pred_train=base_train[train_fit_slice],
        x_val=q_tune,
        y_true_val=bundle.y_train_norm[train_tune_slice],
        y_classical_pred_val=base_train[train_tune_slice],
    )

    q_res_val = q_annotator.predict_residual(q_val)
    q_res_test = q_annotator.predict_residual(q_test)
    q_res_future = q_annotator.predict_residual(q_future)

    q_feature_mode = str(getattr(q_annotator, "quantum_feature_mode", "")).strip().lower()
    w_star = 1.0
    pred_val_used = base_val + q_res_val
    val_mae_used = float(np.mean(np.abs(pred_val_used - bundle.y_val_norm)))
    use_gate = False
    gate_threshold = None

    pred_test_norm = base_test + q_res_test
    pred_future_norm = base_future + q_res_future

    pred_test_surface = _to_surface_space(bundle, pred_test_norm)
    pred_future_surface = _to_surface_space(bundle, pred_future_norm)
    y_test_surface_true = np.clip(bundle.y_test_surface_scaled * bundle.price_scale, 0.0, bundle.clip_cap)

    factor_base_metrics = _compute_metric_block(bundle.y_test_norm, base_test, eps=1e-8)
    factor_blend_metrics = _compute_metric_block(bundle.y_test_norm, pred_test_norm, eps=1e-8)
    surface_blend_metrics = _compute_metric_block(y_test_surface_true, pred_test_surface, eps=1e-8)

    metrics = {
        "level": int(level),
        "backend": str(backend),
        "L": int(L),
        "L_config": int(L_cfg),
        "H_config": int(H_cfg),
        "H": int(H),
        "imputer": imputer,
        "use_cycle_phase": bool(use_cycle_phase_final),
        "cycle_mismatch_count": int(getattr(bundle.ds, "cycle_mismatch_count", 0)),
        "classical_kind": classical_kind,
        "pqrc_M": int(np.clip(int(pqrc_M if pqrc_modes is None else pqrc_modes), 4, 20)),
        "pqrc_modes": int(np.clip(int(pqrc_M if pqrc_modes is None else pqrc_modes), 4, 20)),
        "pqrc_Nph": int(pqrc_nphotons_eff),
        "pqrc_nphotons": int(pqrc_nphotons_eff),
        "pqrc_budget": int(pqrc_budget),
        "pqrc_gain": float(pqrc_gain),
        "pqrc_feature": str(pqrc_feature),
        "pqrc_pseudocount": float(pqrc_pseudocount),
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
        "persist_qrc_alpha": float(persist_qrc_alpha),
        "persist_qrc_rclip": float(persist_qrc_rclip),
        "persist_qrc_drive_scale": float(persist_qrc_drive_scale),
        "tau": float(gate_tau),
        "qrc_gate_tau": float(gate_tau),
        "qrc_resid_clip": None if qrc_resid_clip is None else float(qrc_resid_clip),
        "qrc_residvar_penalty": float(qrc_residvar_penalty),
        "qrc_feat_norm": str(qrc_feat_norm),
        "quantum_mode": quantum_mode,
        "quantum_feature_mode": q_feature_mode if q_feature_mode else "unknown",
        "selected_feature_indices": selected_idx,
        "selected_dim": int(final_dim),
        "mode_cap": int(mode_cap),
        "w_star": float(w_star),
        "gating_enabled_requested": bool(gate_enabled),
        "gating_used": bool(use_gate),
        "gate_source": gate_source,
        "gate_threshold": None if gate_threshold is None else float(gate_threshold),
        "val_mae_no_gate": float(val_mae_used),
        "val_mae_used": float(val_mae_used),
        "test_factor_mae_base": float(factor_base_metrics["mae"]),
        "test_factor_rmse_base": float(factor_base_metrics["rmse"]),
        "test_factor_mape_base": float(factor_base_metrics["mape"]),
        "test_factor_smape_base": float(factor_base_metrics["smape"]),
        "test_factor_wape_base": float(factor_base_metrics["wape"]),
        "test_factor_mae_blend": float(factor_blend_metrics["mae"]),
        "test_factor_rmse_blend": float(factor_blend_metrics["rmse"]),
        "test_factor_mape_blend": float(factor_blend_metrics["mape"]),
        "test_factor_smape_blend": float(factor_blend_metrics["smape"]),
        "test_factor_wape_blend": float(factor_blend_metrics["wape"]),
        "test_surface_mae_blend": float(surface_blend_metrics["mae"]),
        "test_surface_rmse_blend": float(surface_blend_metrics["rmse"]),
        "test_surface_mape_blend": float(surface_blend_metrics["mape"]),
        "test_surface_smape_blend": float(surface_blend_metrics["smape"]),
        "test_surface_wape_blend": float(surface_blend_metrics["wape"]),
        "test_horizon_factor_mae_blend": _horizon_metric(bundle.y_test_norm, pred_test_norm, metric="mae", eps=1e-8),
        "test_horizon_factor_rmse_blend": _horizon_metric(bundle.y_test_norm, pred_test_norm, metric="rmse", eps=1e-8),
        "test_horizon_factor_mape_blend": _horizon_metric(bundle.y_test_norm, pred_test_norm, metric="mape", eps=1e-8),
        "test_horizon_factor_smape_blend": _horizon_metric(bundle.y_test_norm, pred_test_norm, metric="smape", eps=1e-8),
        "test_horizon_factor_wape_blend": _horizon_metric(bundle.y_test_norm, pred_test_norm, metric="wape", eps=1e-8),
        "test_horizon_surface_mae_blend": _horizon_metric(y_test_surface_true, pred_test_surface, metric="mae", eps=1e-8),
        "test_horizon_surface_rmse_blend": _horizon_metric(y_test_surface_true, pred_test_surface, metric="rmse", eps=1e-8),
        "test_horizon_surface_mape_blend": _horizon_metric(y_test_surface_true, pred_test_surface, metric="mape", eps=1e-8),
        "test_horizon_surface_smape_blend": _horizon_metric(y_test_surface_true, pred_test_surface, metric="smape", eps=1e-8),
        "test_horizon_surface_wape_blend": _horizon_metric(y_test_surface_true, pred_test_surface, metric="wape", eps=1e-8),
    }
    metrics.update(q_annotator.budget_dict())

    out_df = make_submission_from_predictions(
        data_dir=data_dir,
        predicted_surfaces=pred_future_surface[0],
        out_csv=out_csv,
        expected_surface_cols=bundle.ds.surface_cols,
    )
    print(f"Saved submission CSV: {out_csv}")
    print(f"Submission shape: {out_df.shape}")
    return out_df, {"metrics": metrics}


def _run_self_test(args: argparse.Namespace) -> None:
    bundle = load_and_preprocess(
        data_dir=args.data_dir,
        level=args.level,
        L=args.L,
        H=args.H,
        imputer=args.imputer,
        d_factors=args.d_factors,
        seed=args.seed,
        use_cycle_phase=bool(args.use_cycle_phase),
    )

    shape_items = {
        "x_train": bundle.x_train.shape,
        "x_val": bundle.x_val.shape,
        "x_test": bundle.x_test.shape,
        "y_train": bundle.y_train.shape,
        "y_val": bundle.y_val.shape,
        "y_test": bundle.y_test.shape,
        "last_window": bundle.last_window.shape,
        "missing_train_norm": bundle.missing_train_norm.shape,
        "factor_dim": (bundle.factor_dim,),
    }
    for name, shape in shape_items.items():
        print(f"{name} shape: {shape}")

    finite_checks = {
        "x_train": np.isfinite(bundle.x_train).all(),
        "x_val": np.isfinite(bundle.x_val).all(),
        "x_test": np.isfinite(bundle.x_test).all(),
        "y_train": np.isfinite(bundle.y_train).all(),
        "y_val": np.isfinite(bundle.y_val).all(),
        "y_test": np.isfinite(bundle.y_test).all(),
        "last_window": np.isfinite(bundle.last_window).all(),
        "x_train_norm": np.isfinite(bundle.x_train_norm).all(),
        "missing_train_norm": np.isfinite(bundle.missing_train_norm).all(),
    }
    for name, ok in finite_checks.items():
        print(f"{name} finite: {bool(ok)}")
    if not all(finite_checks.values()):
        raise RuntimeError("Self-test failed: detected NaN/Inf values.")
    print("Self-test passed: shapes printed and no NaNs/Infs found.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission preprocessing self-test.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--imputer", type=str, choices=["ffill_interp", "svd_iterative"], default="ffill_interp")
    parser.add_argument("--d_factors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cycle_phase", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    _run_self_test(_parse_args())
