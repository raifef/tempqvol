from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warnings

from ..positive_transform import PositiveLogTransform

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

FeatureType = Literal["coincidence", "clickprob", "pseudo_coincidence"]
QRCMode = Literal["residual", "gated_residual", "absolute", "auto"]
FeatNormMode = Literal["none", "standard"]
QRCTargetMode = Literal["level", "delta", "log_return", "norm_resid"]
QRCBaselineMode = Literal["persistence", "persistence_naive", "seasonal13"]
BudgetAllocMode = Literal["uniform", "weighted"]
TargetTransformMode = Literal["none", "log"]
DEFAULT_PQRC_SHOTS = 32


def _sanitize_shots(shots: int, *, field_name: str) -> int:
    s = int(shots)
    if s >= 1:
        return s
    warnings.warn(
        f"{field_name}={s} is non-positive; forcing shots={DEFAULT_PQRC_SHOTS} "
        "so quantum models run with finite shot sampling.",
        RuntimeWarning,
    )
    return int(DEFAULT_PQRC_SHOTS)


def _wrap_phase(phi: np.ndarray) -> np.ndarray:
    out = np.asarray(phi, dtype=np.float64)
    return (out + np.pi) % (2.0 * np.pi) - np.pi


def _haar_like_unitary(m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    a = rng.normal(size=(int(m), int(m))) + 1j * rng.normal(size=(int(m), int(m)))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    d = np.where(np.abs(d) > 1e-12, d / np.abs(d), 1.0 + 0.0j)
    q = q * d[None, :]
    return np.asarray(q, dtype=np.complex128)


def _fit_linear_projector(x: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    x2 = np.asarray(x, dtype=np.float64)
    x2 = np.clip(np.nan_to_num(x2, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    mean = np.mean(x2, axis=0, keepdims=True)
    xc = x2 - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[: int(n_comp)].T.copy()
    return mean.reshape(-1), comp


def _project_linear(x: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    x2 = np.asarray(x, dtype=np.float64)
    x2 = np.clip(np.nan_to_num(x2, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = (x2 - mean[None, :]) @ comp
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)
    return out


def _pair_indices(m: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(m), k=1)


def _analytic_click_prob(p: np.ndarray, nph: int) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    n = max(1, int(nph))
    return np.clip(1.0 - np.power(1.0 - p_clip, n), 0.0, 1.0)


def _sample_click_features(p: np.ndarray, nph: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    m = int(p.shape[0])
    s = int(max(1, shots))
    n = int(max(1, nph))
    clicks = np.zeros((s, m), dtype=np.float64)
    for i in range(s):
        occ = rng.multinomial(n=n, pvals=p)
        clicks[i] = (occ > 0).astype(np.float64)
    click_prob = np.mean(clicks, axis=0)
    iu, ju = _pair_indices(m)
    pair = np.mean(clicks[:, iu] * clicks[:, ju], axis=0)
    return np.concatenate([click_prob, pair], axis=0)


def _sample_click_features_deterministic(
    p: np.ndarray,
    nph: int,
    shot_uniforms: np.ndarray,
) -> np.ndarray:
    p2 = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    s = int(shot_uniforms.shape[0])
    m = int(p2.shape[0])
    if s <= 0 or m <= 0:
        return np.zeros((m + (m * (m - 1)) // 2,), dtype=np.float64)
    cdf = np.cumsum(p2)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    u = np.clip(np.asarray(shot_uniforms, dtype=np.float64), 0.0, 1.0 - 1e-12)
    n = int(max(1, min(int(nph), int(u.shape[1]))))
    mode_idx = np.searchsorted(cdf, u[:, :n], side="right")
    mode_idx = np.clip(mode_idx, 0, m - 1)
    clicks = np.zeros((s, m), dtype=np.float64)
    rows = np.arange(s, dtype=int)
    for j in range(n):
        clicks[rows, mode_idx[:, j]] = 1.0
    click_prob = np.mean(clicks, axis=0)
    iu, ju = _pair_indices(m)
    pair = np.mean(clicks[:, iu] * clicks[:, ju], axis=0)
    return np.concatenate([click_prob, pair], axis=0)


def _smooth_histogram_probs(
    p: np.ndarray,
    *,
    pseudocount: float,
    shots: int,
) -> np.ndarray:
    v = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if v.size == 0:
        return v
    alpha = float(max(0.0, pseudocount))
    if alpha <= 0.0:
        return v
    n_eff = float(max(1, int(shots))) if int(shots) > 0 else float(max(1.0, np.sum(v)))
    counts = np.clip(v, 0.0, 1.0) * n_eff
    # Treat each bin as a Bernoulli event with Beta(alpha, alpha) prior.
    # This preserves per-bin probabilities instead of forcing a multinomial sum=1.
    denom = float(n_eff + 2.0 * alpha)
    smoothed = (counts + alpha) / max(1e-12, denom)
    return np.clip(np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    den = np.maximum(np.abs(np.asarray(y_true, dtype=np.float64)), float(max(1e-12, eps)))
    num = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    return float(np.mean(num / den) * 100.0)


def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))))


def _checksum_matrix(arr: np.ndarray, max_rows: int = 64) -> str:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 0:
        x2 = x.reshape(1, 1)
    elif x.ndim == 1:
        x2 = x.reshape(1, -1)
    else:
        x2 = x.reshape(x.shape[0], -1)
    x2 = np.nan_to_num(x2, nan=0.0, posinf=1e6, neginf=-1e6)
    head = x2[: int(max(1, min(max_rows, x2.shape[0])))]
    return hashlib.sha1(np.ascontiguousarray(head).tobytes()).hexdigest()


class _IdentityScaler:
    def fit(self, x: np.ndarray) -> "_IdentityScaler":
        _ = np.asarray(x, dtype=np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


def _build_proto_from_x(
    x_ref: np.ndarray,
    *,
    y_dim: int,
    horizon: int,
    params: PhotonicQRCParams,
) -> PhotonicQRCForecaster:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("photonic_qrc requires scikit-learn.")
    x = np.asarray(x_ref, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x_ref must be [N,L,D], got {x.shape}.")
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    _, _, x_dim = x.shape
    d_raw = int(max(1, min(int(y_dim), int(x_dim))))
    if int(params.factor_cap) > 0:
        d_core = int(max(1, min(d_raw, int(params.factor_cap))))
    else:
        d_core = int(d_raw)
    m = int(np.clip(int(params.modes), 4, 20))
    in_pca = int(np.clip(int(params.in_pca), 1, min(16, d_core)))
    budget = int(max(1, int(params.budget)))
    n_phase = int(max(2 * m, budget))

    x_core = x[:, :, :d_core]
    x_scaler = StandardScaler().fit(x_core.reshape(-1, d_core))
    x_s = x_scaler.transform(x_core.reshape(-1, d_core))
    proj_mean, proj_comp = _fit_linear_projector(x_s, n_comp=in_pca)

    rng = np.random.default_rng(int(params.seed))
    u0 = _haar_like_unitary(m, seed=int(params.seed))
    phase_mix = rng.normal(0.0, 1.0 / np.sqrt(max(1, n_phase)), size=(m, n_phase)).astype(np.float64)
    phase_init = rng.uniform(-np.pi, np.pi, size=(n_phase,)).astype(np.float64)
    idx_in = rng.choice(n_phase, size=min(4, n_phase), replace=False)
    idx_budget = rng.choice(n_phase, size=min(budget, n_phase), replace=False)
    w_input = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(idx_in.size, in_pca)).astype(np.float64)
    w_vr = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(m, in_pca)).astype(np.float64)
    w_vi = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(m, in_pca)).astype(np.float64)
    b_vr = rng.normal(0.0, 0.1, size=(m,)).astype(np.float64)
    b_vi = rng.normal(0.0, 0.1, size=(m,)).astype(np.float64)

    pair_i, pair_j = _pair_indices(m)
    triple_i = np.array([], dtype=int)
    triple_j = np.array([], dtype=int)
    triple_k = np.array([], dtype=int)
    if int(params.higher_order) >= 2:
        all_triples = np.array(
            [(i, j, k) for i in range(m) for j in range(i + 1, m) for k in range(j + 1, m)],
            dtype=int,
        )
        if all_triples.size > 0:
            max_triples = min(256, int(all_triples.shape[0]))
            pick = np.sort(rng.choice(all_triples.shape[0], size=max_triples, replace=False))
            tri = all_triples[pick]
            triple_i = tri[:, 0]
            triple_j = tri[:, 1]
            triple_k = tri[:, 2]
    if int(params.higher_order) <= 0:
        feat_dim = m
    elif int(params.higher_order) == 1:
        feat_dim = m + pair_i.size
    else:
        feat_dim = m + pair_i.size + int(triple_i.size)
    w_fb = rng.normal(0.0, 1.0 / np.sqrt(max(1, feat_dim)), size=(idx_budget.size, feat_dim)).astype(np.float64)
    if str(params.budget_alloc).strip().lower() == "weighted":
        w = rng.uniform(0.25, 1.0, size=(idx_budget.size,)).astype(np.float64)
        budget_weights = w / max(1e-12, float(np.mean(w)))
    else:
        budget_weights = np.ones((idx_budget.size,), dtype=np.float64)
    shot_uniforms = rng.uniform(
        low=0.0,
        high=1.0,
        size=(int(max(1, params.shots)), int(max(1, params.n_photons))),
    ).astype(np.float64)

    y_scaler = StandardScaler().fit(np.vstack([np.zeros((1, int(y_dim * horizon))), np.ones((1, int(y_dim * horizon)))]))
    return PhotonicQRCForecaster(
        ridge=Ridge(alpha=float(max(1e-8, params.ridge_alpha)), random_state=int(params.seed)),
        x_scaler=x_scaler,
        phi_scaler=_IdentityScaler(),
        y_scaler=y_scaler,
        y_dim=int(y_dim),
        horizon=int(horizon),
        d_factors=d_core,
        params=params,
        proj_mean=proj_mean,
        proj_comp=proj_comp,
        u0=u0,
        phase_mix=phase_mix,
        phase_init=phase_init,
        idx_in=idx_in,
        idx_budget=idx_budget,
        w_input=w_input,
        w_vr=w_vr,
        w_vi=w_vi,
        b_vr=b_vr,
        b_vi=b_vi,
        w_fb=w_fb,
        pair_i=pair_i,
        pair_j=pair_j,
        triple_i=triple_i,
        triple_j=triple_j,
        triple_k=triple_k,
        budget_weights=budget_weights,
        shot_uniforms=shot_uniforms,
        qrc_mode=str(params.qrc_mode),
        qrc_target=str(params.qrc_target),
        qrc_baseline=str(params.qrc_baseline),
        target_transform=str(params.target_transform),
        tau=float(params.tau),
        residual_scale=1.0,
        residual_clip=10.0 if params.resid_clip is None else float(max(1e-8, abs(float(params.resid_clip)))),
        higher_order=int(params.higher_order),
        shots_per_eval=int(max(1, params.shots)),
    )


def compute_photonic_memory_proxy(
    x_seq: np.ndarray,
    *,
    params: PhotonicQRCParams,
    max_k: int = 10,
    ridge_alpha: float = 1.0,
    max_samples: int = 256,
) -> dict[str, Any]:
    x = np.asarray(x_seq, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0:
        return {"mc_score": 0.0, "stability_penalty": 0.0, "r2_by_k": []}
    n = int(max(8, min(int(max_samples), x.shape[0])))
    xs = np.asarray(x[:n], dtype=np.float64)
    y_dim = int(max(1, min(xs.shape[2], 8)))
    proto = _build_proto_from_x(xs, y_dim=y_dim, horizon=1, params=params)
    phi = proto._window_states(xs)
    phi = np.clip(np.nan_to_num(phi, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    phi_s = StandardScaler().fit_transform(phi)
    x_core = np.clip(np.nan_to_num(xs[:, :, : proto.d_factors], nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    x_core_s = proto.x_scaler.transform(x_core.reshape(-1, x_core.shape[2])).reshape(x_core.shape)
    u_seq = _project_linear(x_core_s.reshape(-1, x_core.shape[2]), proto.proj_mean, proto.proj_comp).reshape(
        x_core.shape[0], x_core.shape[1], proto.proj_comp.shape[1]
    )
    u_scalar = np.asarray(u_seq[:, :, 0], dtype=np.float64)
    l = int(u_scalar.shape[1])
    k_max = int(max(1, min(int(max_k), max(1, l - 1))))
    split = int(max(8, min(n - 4, int(0.8 * n))))
    r2_list: list[float] = []
    for k in range(1, k_max + 1):
        idx = int(l - 1 - k)
        if idx < 0:
            break
        yk = u_scalar[:, idx]
        xtr = phi_s[:split]
        xva = phi_s[split:]
        ytr = yk[:split]
        yva = yk[split:]
        if xva.shape[0] < 4:
            continue
        reg = Ridge(alpha=float(max(1e-10, ridge_alpha)), random_state=int(params.seed))
        reg.fit(xtr, ytr)
        yp = reg.predict(xva)
        ss_res = float(np.sum((yva - yp) ** 2))
        ss_tot = float(np.sum((yva - np.mean(yva)) ** 2))
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-12)))
        r2_list.append(float(np.clip(np.nan_to_num(r2, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)))
    mc_score = float(np.sum(np.maximum(np.asarray(r2_list, dtype=np.float64), 0.0)))
    feat_std = np.std(phi_s, axis=0)
    med_std = float(np.median(feat_std))
    mean_std = float(np.mean(feat_std))
    flat = phi_s.reshape(phi_s.shape[0], -1)
    if flat.shape[0] > 2:
        s0 = flat[:-1, 0]
        s1 = flat[1:, 0]
        acf = float(np.corrcoef(s0, s1)[0, 1]) if np.std(s0) > 1e-12 and np.std(s1) > 1e-12 else 0.0
    else:
        acf = 0.0
    instability = 0.0
    if med_std < 1e-3:
        instability += float((1e-3 - med_std) * 20.0)
    if mean_std > 5.0:
        instability += float((mean_std - 5.0) * 0.1)
    if not np.isfinite(acf):
        acf = 0.0
    if abs(acf) > 0.995:
        instability += float((abs(acf) - 0.995) * 100.0)
    return {
        "mc_score": float(mc_score),
        "stability_penalty": float(instability),
        "mc_score_adj": float(mc_score - instability),
        "r2_by_k": [float(v) for v in r2_list],
        "feature_std_median": med_std,
        "feature_std_mean": mean_std,
        "state_autocorr_lag1": float(acf),
    }


@dataclass
class PhotonicQRCParams:
    modes: int = 12
    n_photons: int = 2
    budget: int = 32
    gain: float = 0.5
    feature: FeatureType = "coincidence"
    shots: int = DEFAULT_PQRC_SHOTS
    pseudocount: float = 0.5
    ridge_alpha: float = 1e-2
    qrc_mode: QRCMode = "auto"
    qrc_target: QRCTargetMode = "delta"
    qrc_baseline: QRCBaselineMode = "persistence"
    target_transform: TargetTransformMode = "log"
    y_floor_mode: str = "train_p001"
    y_floor_value: float | None = None
    tau: float = 0.05
    resid_clip: float | None = None
    residvar_penalty: float = 0.1
    feat_norm: FeatNormMode = "standard"
    in_pca: int = 4
    factor_cap: int = 0
    input_scale: float = 0.75
    min_residual_scale: float = 0.05
    higher_order: int = 1
    budget_alloc: BudgetAllocMode = "uniform"
    seed: int = 0
    feedback_enabled: bool = True

    def __post_init__(self) -> None:
        self.modes = int(np.clip(int(self.modes), 4, 20))
        self.shots = _sanitize_shots(self.shots, field_name="PhotonicQRCParams.shots")
        self.pseudocount = float(max(0.0, float(self.pseudocount)))
        self.budget = int(max(1, int(self.budget)))
        self.factor_cap = int(max(0, int(self.factor_cap)))
        self.min_residual_scale = float(np.clip(self.min_residual_scale, 1e-4, 3.0))
        self.tau = float(np.clip(self.tau, 1e-8, 50.0))
        self.residvar_penalty = float(np.clip(self.residvar_penalty, 0.0, 10.0))
        self.higher_order = int(np.clip(int(self.higher_order), 0, 2))
        feature = str(self.feature).strip().lower()
        if feature not in {"coincidence", "clickprob", "pseudo_coincidence"}:
            raise ValueError("feature must be one of: coincidence, clickprob, pseudo_coincidence.")
        self.feature = feature  # type: ignore[assignment]
        mode = str(self.qrc_mode).strip().lower()
        if mode not in {"residual", "gated_residual", "absolute", "auto"}:
            raise ValueError("qrc_mode must be one of: residual, gated_residual, absolute, auto.")
        self.qrc_mode = mode  # type: ignore[assignment]
        target = str(self.qrc_target).strip().lower()
        if target not in {"level", "delta", "log_return", "norm_resid"}:
            raise ValueError("qrc_target must be one of: level, delta, log_return, norm_resid.")
        self.qrc_target = target  # type: ignore[assignment]
        baseline = str(self.qrc_baseline).strip().lower()
        if baseline == "persistence_naive":
            baseline = "persistence"
        if baseline not in {"persistence", "seasonal13"}:
            raise ValueError("qrc_baseline must be one of: persistence, persistence_naive, seasonal13.")
        self.qrc_baseline = baseline  # type: ignore[assignment]
        tt = str(getattr(self, "target_transform", "log")).strip().lower()
        if tt not in {"none", "log"}:
            raise ValueError("target_transform must be one of: none, log.")
        self.target_transform = tt  # type: ignore[assignment]
        yfm = str(getattr(self, "y_floor_mode", "train_p001")).strip().lower()
        if yfm not in {"train_p01", "train_p001", "fixed"}:
            raise ValueError("y_floor_mode must be one of: train_p01, train_p001, fixed.")
        self.y_floor_mode = yfm  # type: ignore[assignment]
        if self.y_floor_mode == "fixed" and self.y_floor_value is not None:
            self.y_floor_value = float(max(1e-12, float(self.y_floor_value)))
        fn = str(self.feat_norm).strip().lower()
        if fn not in {"none", "standard"}:
            raise ValueError("feat_norm must be one of: none, standard.")
        self.feat_norm = fn  # type: ignore[assignment]
        ba = str(self.budget_alloc).strip().lower()
        if ba not in {"uniform", "weighted"}:
            raise ValueError("budget_alloc must be one of: uniform, weighted.")
        self.budget_alloc = ba  # type: ignore[assignment]


@dataclass
class PhotonicQRCForecaster:
    ridge: Ridge
    x_scaler: StandardScaler
    phi_scaler: Any
    y_scaler: StandardScaler
    y_dim: int
    horizon: int
    d_factors: int
    params: PhotonicQRCParams
    proj_mean: np.ndarray
    proj_comp: np.ndarray
    u0: np.ndarray
    phase_mix: np.ndarray
    phase_init: np.ndarray
    idx_in: np.ndarray
    idx_budget: np.ndarray
    w_input: np.ndarray
    w_vr: np.ndarray
    w_vi: np.ndarray
    b_vr: np.ndarray
    b_vi: np.ndarray
    w_fb: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    triple_i: np.ndarray
    triple_j: np.ndarray
    triple_k: np.ndarray
    budget_weights: np.ndarray
    shot_uniforms: np.ndarray | None = None
    qrc_mode: str = "residual"
    qrc_target: str = "delta"
    qrc_baseline: str = "persistence"
    target_transform: str = "none"
    y_positive_transform: PositiveLogTransform | None = None
    tau: float = 1e-3
    residual_scale: float = 1.0
    residual_clip: float = 10.0
    target_eps: float = 1e-6
    log_offset: np.ndarray | None = None
    norm_scale_floor: float = 1e-6
    higher_order: int = 1
    mode_val_mape: dict[str, float] | None = None
    feature_stats: dict[str, Any] | None = None
    train_qevals: int = 0
    train_total_shots: int = 0
    infer_qevals: int = 0
    infer_total_shots: int = 0
    shots_per_eval: int = DEFAULT_PQRC_SHOTS
    last_q_features_: np.ndarray | None = None
    last_gate_: np.ndarray | None = None
    feedback_sanity: dict[str, Any] | None = None
    val_target_residual_mae: float | None = None
    training_curve_: list[dict[str, Any]] | None = None
    prob_min_before: float = np.inf
    prob_min_after: float = np.inf
    prob_zero_frac_before_sum: float = 0.0
    prob_zero_frac_after_sum: float = 0.0
    prob_stats_steps: int = 0

    def _reset_prob_stats(self) -> None:
        self.prob_min_before = np.inf
        self.prob_min_after = np.inf
        self.prob_zero_frac_before_sum = 0.0
        self.prob_zero_frac_after_sum = 0.0
        self.prob_stats_steps = 0

    def _update_prob_stats(self, before: np.ndarray, after: np.ndarray) -> None:
        b = np.clip(np.nan_to_num(np.asarray(before, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        a = np.clip(np.nan_to_num(np.asarray(after, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if b.size == 0 or a.size == 0:
            return
        self.prob_min_before = float(min(self.prob_min_before, float(np.min(b))))
        self.prob_min_after = float(min(self.prob_min_after, float(np.min(a))))
        self.prob_zero_frac_before_sum += float(np.mean(b <= 0.0))
        self.prob_zero_frac_after_sum += float(np.mean(a <= 0.0))
        self.prob_stats_steps += 1

    def _snapshot_prob_stats(self) -> dict[str, float]:
        steps = int(max(1, self.prob_stats_steps))
        return {
            "min_prob_before": float(self.prob_min_before if np.isfinite(self.prob_min_before) else 0.0),
            "min_prob_after": float(self.prob_min_after if np.isfinite(self.prob_min_after) else 0.0),
            "fraction_zero_bins_before": float(self.prob_zero_frac_before_sum / float(steps)),
            "fraction_zero_bins_after": float(self.prob_zero_frac_after_sum / float(steps)),
        }

    def _baseline_from_x(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
        n, l, _ = x.shape
        y_dim = int(self.y_dim)
        h = int(self.horizon)
        base_persist = np.repeat(x[:, -1:, :y_dim], h, axis=1)
        baseline_mode = str(self.qrc_baseline).strip().lower()
        if baseline_mode in {"persistence_naive", "persistence"}:
            return base_persist
        out = np.zeros((n, h, y_dim), dtype=np.float64)
        for h_idx in range(h):
            step = int(h_idx + 1)
            lag = int(13 - step)
            if lag <= 0:
                out[:, h_idx, :] = x[:, -1, :y_dim]
            elif lag < l:
                out[:, h_idx, :] = x[:, -(lag + 1), :y_dim]
            else:
                out[:, h_idx, :] = x[:, -1, :y_dim]
        return out

    def _norm_scale_from_x(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        s = np.std(x[:, :, : self.y_dim], axis=1)
        s = np.clip(np.nan_to_num(s, nan=float(self.norm_scale_floor), posinf=50.0, neginf=float(self.norm_scale_floor)), float(self.norm_scale_floor), 50.0)
        return s

    def _target_from_level(self, y_level: np.ndarray, baseline: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            y_log = self.y_positive_transform.forward(np.asarray(y_level, dtype=np.float64))
            b_log = self.y_positive_transform.forward(np.asarray(baseline, dtype=np.float64))
            return y_log - b_log
        mode = str(self.qrc_target).strip().lower()
        y = np.asarray(y_level, dtype=np.float64)
        b = np.asarray(baseline, dtype=np.float64)
        eps = float(max(1e-12, self.target_eps))
        if mode == "level":
            return y
        if mode == "delta":
            return y - b
        if mode == "norm_resid":
            scale = self._norm_scale_from_x(x_seq)
            return (y - b) / scale[:, None, :]
        # Robust log-return for signed factor series: apply a per-dimension positive shift.
        offset = np.asarray(self.log_offset if self.log_offset is not None else np.zeros((self.y_dim,), dtype=np.float64), dtype=np.float64)
        num = np.maximum(y + offset[None, None, :], eps)
        den = np.maximum(b + offset[None, None, :], eps)
        return np.log(num / den)

    def _level_from_target(self, y_target: np.ndarray, baseline: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            b_log = self.y_positive_transform.forward(np.asarray(baseline, dtype=np.float64))
            y_log = b_log + np.asarray(y_target, dtype=np.float64)
            out = self.y_positive_transform.inverse(y_log)
            return self.y_positive_transform.clamp(out)
        mode = str(self.qrc_target).strip().lower()
        yt = np.asarray(y_target, dtype=np.float64)
        b = np.asarray(baseline, dtype=np.float64)
        eps = float(max(1e-12, self.target_eps))
        if mode == "level":
            return yt
        if mode == "delta":
            return b + yt
        if mode == "norm_resid":
            scale = self._norm_scale_from_x(x_seq)
            return b + yt * scale[:, None, :]
        offset = np.asarray(self.log_offset if self.log_offset is not None else np.zeros((self.y_dim,), dtype=np.float64), dtype=np.float64)
        return np.exp(np.clip(yt, -20.0, 20.0)) * np.maximum(b + offset[None, None, :], eps) - offset[None, None, :]

    def _step_features(
        self,
        u_t: np.ndarray,
        phase_state: np.ndarray,
        *,
        feedback_enabled: bool | None = None,
        gain: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        phase_next = np.asarray(phase_state, dtype=np.float64).copy()
        drive = self.w_input @ u_t
        drive = np.clip(np.nan_to_num(drive, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)
        phase_next[self.idx_in] = _wrap_phase(phase_next[self.idx_in] + float(self.params.input_scale) * drive)

        mode_phase = self.phase_mix @ phase_next
        phase_diag = np.exp(1j * mode_phase)
        # U = U0 @ D(phi): D applies phase shifts after fixed random mixing.
        u_mat = self.u0 * phase_diag[None, :]

        vr = self.w_vr @ u_t + self.b_vr
        vi = self.w_vi @ u_t + self.b_vi
        v = vr + 1j * vi
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            v = np.zeros_like(v, dtype=np.complex128)
            v[0] = 1.0 + 0.0j
        else:
            v = v / norm

        z = u_mat @ v
        p = np.abs(z) ** 2
        ps = float(np.sum(p))
        if ps <= 1e-12 or not np.isfinite(ps):
            p = np.ones_like(p, dtype=np.float64) / float(max(1, p.shape[0]))
        else:
            p = np.clip(p / ps, 0.0, 1.0)
            p = p / max(1e-12, float(np.sum(p)))

        if int(self.params.shots) > 0:
            if self.shot_uniforms is not None:
                full_feat = _sample_click_features_deterministic(
                    p=p,
                    nph=int(self.params.n_photons),
                    shot_uniforms=self.shot_uniforms,
                )
            else:
                local_rng = np.random.default_rng(int(self.params.seed))
                full_feat = _sample_click_features(
                    p=p,
                    nph=int(self.params.n_photons),
                    shots=int(self.params.shots),
                    rng=local_rng,
                )
            click_prob = full_feat[: int(self.params.modes)]
            pair_feat_sampled = full_feat[int(self.params.modes) :]
        else:
            click_prob = _analytic_click_prob(p, nph=int(self.params.n_photons))
            pair_feat_sampled = click_prob[self.pair_i] * click_prob[self.pair_j]

        click_before = np.clip(np.nan_to_num(click_prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        click_after = _smooth_histogram_probs(
            click_before,
            pseudocount=float(self.params.pseudocount),
            shots=int(self.params.shots),
        )

        feature_mode = str(self.params.feature).strip().lower()
        if feature_mode == "pseudo_coincidence":
            pair_before = click_before[self.pair_i] * click_before[self.pair_j]
        else:
            pair_before = np.clip(np.nan_to_num(pair_feat_sampled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        pair_after = _smooth_histogram_probs(
            pair_before,
            pseudocount=float(self.params.pseudocount),
            shots=int(self.params.shots),
        )

        # Coincidence-inspired coarse-grained feature map:
        # arXiv:2602.17440 uses coarse-grained coincidence/readout statistics with threshold-like detection.
        pieces: list[np.ndarray] = []
        if int(self.higher_order) <= 0:
            pieces = [click_after]
        elif int(self.higher_order) == 1:
            pieces = [click_after, pair_after]
        else:
            triple_feat = click_after[self.triple_i] * click_after[self.triple_j] * click_after[self.triple_k]
            pieces = [click_after, pair_after, triple_feat]
        if feature_mode == "clickprob" and int(self.higher_order) <= 0:
            feat = click_after
            self._update_prob_stats(click_before, click_after)
        else:
            feat = np.concatenate(pieces, axis=0)
            base_before = click_before if int(self.higher_order) <= 0 else np.concatenate([click_before, pair_before], axis=0)
            base_after = click_after if int(self.higher_order) <= 0 else np.concatenate([click_after, pair_after], axis=0)
            self._update_prob_stats(base_before, base_after)
        feat = np.clip(np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        delta_mag = 0.0
        fb_on = bool(self.params.feedback_enabled if feedback_enabled is None else feedback_enabled)
        g = float(self.params.gain if gain is None else gain)
        if fb_on and g > 0.0 and self.idx_budget.size > 0:
            # Measurement-conditioned phase feedback:
            # inspired by the measurement-feedback recurrence protocol in arXiv:2512.02928.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                raw = self.w_fb @ feat
            raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            delta = float(g) * np.tanh(raw)
            if self.budget_weights.size == delta.size:
                delta = delta * self.budget_weights
            delta = np.clip(delta, -0.35, 0.35)
            delta_mag = float(np.mean(np.abs(delta)))
            phase_next[self.idx_budget] = _wrap_phase(phase_next[self.idx_budget] + delta[: self.idx_budget.size])
        return feat.astype(np.float64), phase_next, float(delta_mag)

    def _window_states(self, x_seq: np.ndarray, *, feedback_enabled: bool | None = None, gain: float | None = None) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
        n, l, d = x.shape
        if d < self.y_dim:
            raise ValueError(f"x_seq feature dim {d} must be >= target dim {self.y_dim}.")
        if d < self.d_factors:
            raise ValueError(f"x_seq feature dim {d} must be >= capped factor dim {self.d_factors}.")

        x_core = np.clip(np.nan_to_num(x[:, :, : self.d_factors], nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        x_core_s = self.x_scaler.transform(x_core.reshape(-1, self.d_factors)).reshape(n, l, self.d_factors)
        u_seq = _project_linear(x_core_s.reshape(-1, self.d_factors), self.proj_mean, self.proj_comp).reshape(
            n,
            l,
            self.proj_comp.shape[1],
        )
        x_target = np.clip(np.nan_to_num(x[:, :, : self.y_dim], nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

        m = int(self.params.modes)
        pair_dim = (m * (m - 1)) // 2
        triple_dim = int(self.triple_i.size)
        if int(self.higher_order) <= 0:
            fdim = m
        elif int(self.higher_order) == 1:
            fdim = m + pair_dim
        else:
            fdim = m + pair_dim + triple_dim

        u_dim = int(self.proj_comp.shape[1])
        state_dim = int(3 * fdim + 2 * u_dim + 2 * self.y_dim + 2)
        phi = np.zeros((n, state_dim), dtype=np.float64)
        for i in range(n):
            phase_state = self.phase_init.copy()
            mean_f = np.zeros((fdim,), dtype=np.float64)
            sq_f = np.zeros((fdim,), dtype=np.float64)
            last_f = np.zeros((fdim,), dtype=np.float64)
            mean_delta = 0.0
            last_delta = 0.0
            for t in range(l):
                f, phase_state, delta_mag = self._step_features(
                    u_seq[i, t],
                    phase_state=phase_state,
                    feedback_enabled=feedback_enabled,
                    gain=gain,
                )
                last_f = f
                mean_f += f
                sq_f += f * f
                mean_delta += float(delta_mag)
                last_delta = float(delta_mag)
            mean_f /= float(max(1, l))
            mean_delta /= float(max(1, l))
            var_f = np.maximum(sq_f / float(max(1, l)) - mean_f * mean_f, 0.0)
            std_f = np.sqrt(var_f)
            u_last = u_seq[i, -1]
            if l > 1:
                u_delta = u_seq[i, -1] - u_seq[i, -2]
            else:
                u_delta = np.zeros_like(u_last, dtype=np.float64)
            x_last = x_target[i, -1, :]
            if l > 1:
                x_delta = x_target[i, -1, :] - x_target[i, -2, :]
            else:
                x_delta = np.zeros_like(x_last, dtype=np.float64)
            fb_stats = np.array([mean_delta, last_delta], dtype=np.float64)
            state_i = np.concatenate([last_f, mean_f, std_f, u_last, u_delta, x_last, x_delta, fb_stats], axis=0)
            state_i = np.clip(np.nan_to_num(state_i, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
            phi[i] = state_i
        return phi

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        n = int(x_seq.shape[0])
        phi = self._window_states(x_seq)
        self.last_q_features_ = np.asarray(phi, dtype=np.float64)
        qevals = int(max(0, x_seq.shape[0] * x_seq.shape[1]))
        shots = int(max(1, self.shots_per_eval))
        self.infer_qevals += qevals
        self.infer_total_shots += qevals * shots
        phi_s = self.phi_scaler.transform(phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                y_out_s = self.ridge.predict(phi_s)
        y_out = self.y_scaler.inverse_transform(y_out_s).reshape(n, self.horizon, self.y_dim)
        y_out = np.clip(
            np.nan_to_num(y_out, nan=0.0, posinf=float(self.residual_clip), neginf=-float(self.residual_clip)),
            -float(self.residual_clip),
            float(self.residual_clip),
        )
        mode = str(self.qrc_mode).strip().lower()
        if mode == "absolute":
            self.last_gate_ = np.ones_like(y_out, dtype=np.float64)
            if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
                out = self.y_positive_transform.inverse(y_out)
                return self.y_positive_transform.clamp(out)
            return y_out
        base = self._baseline_from_x(x_seq)
        y_res = y_out
        if self.residual_clip is not None:
            y_res = np.clip(y_res, -float(self.residual_clip), float(self.residual_clip))
        gate = np.ones_like(y_res, dtype=np.float64)
        if mode == "gated_residual":
            gate = np.abs(y_res) / (np.abs(y_res) + float(max(1e-12, self.tau)))
            gate = np.clip(np.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            y_res = gate * y_res
        self.last_gate_ = gate
        y_target = float(self.residual_scale) * y_res
        out = self._level_from_target(y_target, baseline=base, x_seq=x_seq)
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            out = self.y_positive_transform.clamp(out)
        return out

    def run_feedback_sanity(self, x_seq: np.ndarray, max_n: int = 64) -> dict[str, Any]:
        if x_seq.ndim != 3 or x_seq.shape[0] <= 0:
            return {
                "feedback_enabled": bool(self.params.feedback_enabled),
                "pqrc_gain": float(self.params.gain),
                "checksum_gain0": "",
                "checksum_gain1": "",
                "phi_diff_gain0_vs_gain1": 0.0,
            }
        n = int(max(1, min(int(max_n), x_seq.shape[0])))
        xs = np.asarray(x_seq[:n], dtype=np.float64)
        phi0 = self._window_states(xs, feedback_enabled=True, gain=0.0)
        phi1 = self._window_states(xs, feedback_enabled=True, gain=1.0)
        diff = float(np.mean(np.abs(phi1 - phi0)))
        out = {
            "feedback_enabled": bool(self.params.feedback_enabled),
            "pqrc_gain": float(self.params.gain),
            "checksum_gain0": _checksum_matrix(phi0),
            "checksum_gain1": _checksum_matrix(phi1),
            "phi_diff_gain0_vs_gain1": diff,
            "feedback_inert": bool(diff < 1e-9),
            "phi_gain0_mean": float(np.mean(phi0)),
            "phi_gain0_std": float(np.std(phi0)),
            "phi_gain1_mean": float(np.mean(phi1)),
            "phi_gain1_std": float(np.std(phi1)),
        }
        if diff < 1e-9:
            warnings.warn(
                "feedback not affecting features: mean_abs(phi[gain=1]-phi[gain=0]) < 1e-9",
                RuntimeWarning,
            )
        return out


def train_photonic_qrc_forecaster(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: PhotonicQRCParams,
) -> PhotonicQRCForecaster:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("photonic_qrc requires scikit-learn.")

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xva = np.asarray(x_val, dtype=np.float64)
    yva = np.asarray(y_val, dtype=np.float64)
    xtr = np.clip(np.nan_to_num(xtr, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    xva = np.clip(np.nan_to_num(xva, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    ytr = np.clip(np.nan_to_num(ytr, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    yva = np.clip(np.nan_to_num(yva, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    if xtr.ndim != 3 or ytr.ndim != 3 or xva.ndim != 3 or yva.ndim != 3:
        raise ValueError("All inputs must be rank-3 [N,T,D].")
    if xtr.shape[2] < ytr.shape[2]:
        raise ValueError(f"x_train feature dim {xtr.shape[2]} < y_train dim {ytr.shape[2]}.")
    if ytr.shape[1] != yva.shape[1] or ytr.shape[2] != yva.shape[2]:
        raise ValueError("Train/val target shapes must match on horizon and dim.")

    n_train, _, x_dim = xtr.shape
    y_dim = int(ytr.shape[2])
    horizon = int(ytr.shape[1])
    d_raw = int(max(1, min(int(y_dim), int(x_dim))))
    if int(params.factor_cap) > 0:
        d_core = int(max(1, min(d_raw, int(params.factor_cap))))
    else:
        d_core = int(d_raw)
    m = int(np.clip(int(params.modes), 4, 20))
    in_pca = int(np.clip(int(params.in_pca), 1, min(16, d_core)))
    budget = int(max(1, params.budget))
    n_phase = int(max(2 * m, budget))

    # Input sequence uses low-rank compression of capped factor features.
    xtr_core = xtr[:, :, :d_core]
    x_scaler = StandardScaler().fit(xtr_core.reshape(-1, d_core))
    xtr_s = x_scaler.transform(xtr_core.reshape(-1, d_core))
    proj_mean, proj_comp = _fit_linear_projector(xtr_s, n_comp=in_pca)

    rng = np.random.default_rng(int(params.seed))
    u0 = _haar_like_unitary(m, seed=int(params.seed))
    phase_mix = rng.normal(0.0, 1.0 / np.sqrt(max(1, n_phase)), size=(m, n_phase)).astype(np.float64)
    phase_init = rng.uniform(-np.pi, np.pi, size=(n_phase,)).astype(np.float64)
    idx_in = rng.choice(n_phase, size=min(4, n_phase), replace=False)
    idx_budget = rng.choice(n_phase, size=min(budget, n_phase), replace=False)

    w_input = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(idx_in.size, in_pca)).astype(np.float64)
    w_vr = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(m, in_pca)).astype(np.float64)
    w_vi = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(m, in_pca)).astype(np.float64)
    b_vr = rng.normal(0.0, 0.1, size=(m,)).astype(np.float64)
    b_vi = rng.normal(0.0, 0.1, size=(m,)).astype(np.float64)

    pair_i, pair_j = _pair_indices(m)
    triple_i = np.array([], dtype=int)
    triple_j = np.array([], dtype=int)
    triple_k = np.array([], dtype=int)
    if int(params.higher_order) >= 2:
        all_triples = np.array(
            [(i, j, k) for i in range(m) for j in range(i + 1, m) for k in range(j + 1, m)],
            dtype=int,
        )
        if all_triples.size > 0:
            max_triples = min(256, int(all_triples.shape[0]))
            pick = np.sort(rng.choice(all_triples.shape[0], size=max_triples, replace=False))
            tri = all_triples[pick]
            triple_i = tri[:, 0]
            triple_j = tri[:, 1]
            triple_k = tri[:, 2]
    if int(params.higher_order) <= 0:
        feat_dim = m
    elif int(params.higher_order) == 1:
        feat_dim = m + pair_i.size
    else:
        feat_dim = m + pair_i.size + int(triple_i.size)
    w_fb = rng.normal(0.0, 1.0 / np.sqrt(max(1, feat_dim)), size=(idx_budget.size, feat_dim)).astype(np.float64)
    if str(params.budget_alloc).strip().lower() == "weighted":
        w = rng.uniform(0.25, 1.0, size=(idx_budget.size,)).astype(np.float64)
        budget_weights = w / max(1e-12, float(np.mean(w)))
    else:
        budget_weights = np.ones((idx_budget.size,), dtype=np.float64)
    shot_uniforms = rng.uniform(
        low=0.0,
        high=1.0,
        size=(int(max(1, params.shots)), int(max(1, params.n_photons))),
    ).astype(np.float64)

    proto = PhotonicQRCForecaster(
        ridge=Ridge(alpha=float(params.ridge_alpha), random_state=int(params.seed)),
        x_scaler=x_scaler,
        phi_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        y_dim=y_dim,
        horizon=horizon,
        d_factors=d_core,
        params=params,
        proj_mean=proj_mean,
        proj_comp=proj_comp,
        u0=u0,
        phase_mix=phase_mix,
        phase_init=phase_init,
        idx_in=idx_in,
        idx_budget=idx_budget,
        w_input=w_input,
        w_vr=w_vr,
        w_vi=w_vi,
        b_vr=b_vr,
        b_vi=b_vi,
        w_fb=w_fb,
        pair_i=pair_i,
        pair_j=pair_j,
        triple_i=triple_i,
        triple_j=triple_j,
        triple_k=triple_k,
        budget_weights=budget_weights,
        shot_uniforms=shot_uniforms,
        qrc_target=str(params.qrc_target),
        qrc_baseline=str(params.qrc_baseline),
        target_transform=str(params.target_transform),
        residual_scale=1.0,
        residual_clip=10.0 if params.resid_clip is None else float(max(1e-8, abs(float(params.resid_clip)))),
        higher_order=int(params.higher_order),
        shots_per_eval=int(max(1, params.shots)),
    )

    proto._reset_prob_stats()
    phi_train = proto._window_states(xtr)
    prob_stats_train = proto._snapshot_prob_stats()
    proto._reset_prob_stats()
    phi_val = proto._window_states(xva)
    prob_stats_val = proto._snapshot_prob_stats()
    if not np.isfinite(phi_train).all() or not np.isfinite(phi_val).all():
        raise RuntimeError("photonic_qrc produced non-finite state features.")
    phi_std = float(np.std(phi_train))
    if phi_std <= 1e-8:
        raise RuntimeError(
            f"photonic_qrc produced degenerate state features (std={phi_std:.3e}). "
            "Check shots/encoding/feature map."
        )

    feat_std_train = np.std(phi_train, axis=0)
    feat_std_val = np.std(phi_val, axis=0)
    feat_low_frac_train = float(np.mean(feat_std_train < 1e-8))
    feat_low_frac_val = float(np.mean(feat_std_val < 1e-8))
    feature_stats: dict[str, Any] = {
        "feat_std_train": feat_std_train.astype(float),
        "feat_std_val": feat_std_val.astype(float),
        "feat_std_train_min": float(np.min(feat_std_train)),
        "feat_std_train_median": float(np.median(feat_std_train)),
        "feat_std_train_max": float(np.max(feat_std_train)),
        "feat_std_val_min": float(np.min(feat_std_val)),
        "feat_std_val_median": float(np.median(feat_std_val)),
        "feat_std_val_max": float(np.max(feat_std_val)),
        "feat_low_frac_train": feat_low_frac_train,
        "feat_low_frac_val": feat_low_frac_val,
        "effective_d_factors": int(d_core),
        "factor_cap": int(params.factor_cap),
        "pqrc_pseudocount": float(params.pseudocount),
        "min_prob_before": float(min(prob_stats_train["min_prob_before"], prob_stats_val["min_prob_before"])),
        "min_prob_after": float(min(prob_stats_train["min_prob_after"], prob_stats_val["min_prob_after"])),
        "fraction_zero_bins_before": float(
            0.5 * (prob_stats_train["fraction_zero_bins_before"] + prob_stats_val["fraction_zero_bins_before"])
        ),
        "fraction_zero_bins_after": float(
            0.5 * (prob_stats_train["fraction_zero_bins_after"] + prob_stats_val["fraction_zero_bins_after"])
        ),
    }
    n_sanity = int(max(4, min(64, xtr.shape[0])))
    x_sanity = np.asarray(xtr[:n_sanity], dtype=np.float64)
    phi_gain0 = proto._window_states(x_sanity, feedback_enabled=True, gain=0.0)
    phi_gain1 = proto._window_states(x_sanity, feedback_enabled=True, gain=1.0)
    gain_diff = float(np.mean(np.abs(phi_gain1 - phi_gain0)))
    mode_alt = 16 if int(params.modes) != 16 else 8
    params_alt = PhotonicQRCParams(
        modes=int(np.clip(mode_alt, 4, 20)),
        n_photons=int(params.n_photons),
        budget=int(params.budget),
        gain=float(params.gain),
        feature=str(params.feature),  # type: ignore[arg-type]
        shots=int(params.shots),
        pseudocount=float(params.pseudocount),
        ridge_alpha=float(params.ridge_alpha),
        qrc_mode=str(params.qrc_mode),  # type: ignore[arg-type]
        qrc_target=str(params.qrc_target),  # type: ignore[arg-type]
        qrc_baseline=str(params.qrc_baseline),  # type: ignore[arg-type]
        target_transform=str(params.target_transform),  # type: ignore[arg-type]
        y_floor_mode=str(params.y_floor_mode),
        y_floor_value=params.y_floor_value,
        tau=float(params.tau),
        resid_clip=params.resid_clip,
        residvar_penalty=float(params.residvar_penalty),
        feat_norm=str(params.feat_norm),  # type: ignore[arg-type]
        in_pca=int(params.in_pca),
        factor_cap=int(params.factor_cap),
        input_scale=float(params.input_scale),
        min_residual_scale=float(params.min_residual_scale),
        higher_order=int(params.higher_order),
        budget_alloc=str(params.budget_alloc),  # type: ignore[arg-type]
        seed=int(params.seed),
        feedback_enabled=bool(params.feedback_enabled),
    )
    proto_alt = _build_proto_from_x(x_sanity, y_dim=y_dim, horizon=horizon, params=params_alt)
    phi_mode_alt = proto_alt._window_states(x_sanity)
    d_overlap = int(min(phi_gain1.shape[1], phi_mode_alt.shape[1]))
    mode_diff = float(np.mean(np.abs(phi_gain1[:, :d_overlap] - phi_mode_alt[:, :d_overlap]))) if d_overlap > 0 else 0.0
    feature_sanity = {
        "modes_ref": int(params.modes),
        "modes_alt": int(mode_alt),
        "checksum_ref": _checksum_matrix(phi_gain1),
        "checksum_gain0": _checksum_matrix(phi_gain0),
        "checksum_gain1": _checksum_matrix(phi_gain1),
        "checksum_modes_alt": _checksum_matrix(phi_mode_alt),
        "diff_gain0_vs_gain1": gain_diff,
        "diff_modes_ref_vs_alt": mode_diff,
    }
    feature_stats["feature_sanity"] = feature_sanity
    if gain_diff < 1e-9:
        warnings.warn("feature sanity check failed: gain=0 and gain=1 produce near-identical features.", RuntimeWarning)
    if mode_diff < 1e-9:
        warnings.warn("feature sanity check failed: pqrc_modes perturbation does not change features.", RuntimeWarning)
    if (
        feature_stats["feat_std_train_median"] < 1e-6
        or feature_stats["feat_std_val_median"] < 1e-6
        or feat_low_frac_train > 0.5
        or feat_low_frac_val > 0.5
    ):
        warnings.warn(
            "photonic_qrc feature-variance warning: low-variance feature map detected; "
            f"train_median_std={feature_stats['feat_std_train_median']:.3e}, "
            f"val_median_std={feature_stats['feat_std_val_median']:.3e}, "
            f"train_low_frac={feat_low_frac_train:.3f}, val_low_frac={feat_low_frac_val:.3f}",
            RuntimeWarning,
        )

    def _baseline_with_mode(x_seq: np.ndarray, mode: str) -> np.ndarray:
        x_local = np.asarray(x_seq, dtype=np.float64)
        n_local, l_local, _ = x_local.shape
        if str(mode).strip().lower() != "seasonal13":
            return np.repeat(x_local[:, -1:, :y_dim], horizon, axis=1)
        out_local = np.zeros((n_local, horizon, y_dim), dtype=np.float64)
        for h_idx in range(horizon):
            step = int(h_idx + 1)
            lag = int(13 - step)
            if lag <= 0:
                out_local[:, h_idx, :] = x_local[:, -1, :y_dim]
            elif lag < l_local:
                out_local[:, h_idx, :] = x_local[:, -(lag + 1), :y_dim]
            else:
                out_local[:, h_idx, :] = x_local[:, -1, :y_dim]
        return out_local

    baseline_requested = str(params.qrc_baseline).strip().lower()
    base_train = _baseline_with_mode(xtr, baseline_requested)
    base_val = _baseline_with_mode(xva, baseline_requested)
    baseline_effective = baseline_requested
    # If seasonal baseline is weaker than naive persistence on validation,
    # force persistence baseline to avoid noisy residual targets.
    if baseline_requested == "seasonal13":
        base_train_naive = _baseline_with_mode(xtr, "persistence")
        base_val_naive = _baseline_with_mode(xva, "persistence")
        seasonal_mape = _compute_mape(yva, base_val, eps=1e-6)
        naive_mape = _compute_mape(yva, base_val_naive, eps=1e-6)
        if naive_mape + 1e-10 < seasonal_mape:
            base_train = base_train_naive
            base_val = base_val_naive
            baseline_effective = "persistence"

    y_positive_transform: PositiveLogTransform | None = None
    if str(params.target_transform).strip().lower() == "log":
        y_positive_transform = PositiveLogTransform(
            floor_mode=str(params.y_floor_mode),
            floor_value=params.y_floor_value,
            eps_min=1e-8,
        ).fit(np.concatenate([ytr.reshape(-1), base_train.reshape(-1)], axis=0))
    proto.y_positive_transform = y_positive_transform
    if y_positive_transform is not None:
        y_train_abs = y_positive_transform.forward(ytr).reshape(n_train, -1)
        y_val_abs = y_positive_transform.forward(yva).reshape(xva.shape[0], -1)
    else:
        y_train_abs = ytr.reshape(n_train, -1)
        y_val_abs = yva.reshape(xva.shape[0], -1)
    if str(params.qrc_target).strip().lower() == "log_return":
        min_all = np.minimum(
            np.min(ytr, axis=(0, 1)),
            np.min(base_train, axis=(0, 1)),
        )
        log_offset = np.maximum(0.0, -min_all + 1e-3).astype(np.float64)
    else:
        log_offset = np.zeros((y_dim,), dtype=np.float64)
    proto.log_offset = log_offset
    y_train_target = proto._target_from_level(ytr, base_train, xtr).reshape(n_train, -1)
    y_val_target = proto._target_from_level(yva, base_val, xva).reshape(xva.shape[0], -1)
    y_train_res = y_train_target
    y_val_res = y_val_target
    y_train_res = np.clip(np.nan_to_num(y_train_res, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
    y_val_res = np.clip(np.nan_to_num(y_val_res, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
    y_train_abs = np.clip(np.nan_to_num(y_train_abs, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
    y_val_abs = np.clip(np.nan_to_num(y_val_abs, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
    if str(params.feat_norm).strip().lower() == "standard":
        phi_scaler: Any = StandardScaler().fit(phi_train)
    else:
        phi_scaler = _IdentityScaler().fit(phi_train)
    x_phi = phi_scaler.transform(phi_train)
    x_phi_val = phi_scaler.transform(phi_val)
    x_phi = np.clip(np.nan_to_num(x_phi, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)
    x_phi_val = np.clip(np.nan_to_num(x_phi_val, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)
    feature_stats["feat_norm"] = str(params.feat_norm)
    median_sigma_train = float(np.median(np.std(x_phi, axis=0)))
    median_sigma_val = float(np.median(np.std(x_phi_val, axis=0)))
    feature_stats["median_sigma_train"] = median_sigma_train
    feature_stats["median_sigma_val"] = median_sigma_val
    feature_stats["median_sigma_train_postnorm"] = median_sigma_train
    feature_stats["median_sigma_val_postnorm"] = median_sigma_val
    feature_stats["phi_train_checksum"] = _checksum_matrix(phi_train)
    feature_stats["phi_val_checksum"] = _checksum_matrix(phi_val)
    feature_stats["phi_train_norm_checksum"] = _checksum_matrix(x_phi)
    feature_stats["phi_val_norm_checksum"] = _checksum_matrix(x_phi_val)
    feature_stats["phi_train_mean"] = float(np.mean(phi_train))
    feature_stats["phi_train_std"] = float(np.std(phi_train))
    feature_stats["phi_val_mean"] = float(np.mean(phi_val))
    feature_stats["phi_val_std"] = float(np.std(phi_val))

    alpha_grid = [
        float(max(1e-10, params.ridge_alpha)),
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ]
    seen = set()
    alpha_grid = [a for a in alpha_grid if not (a in seen or seen.add(a))]
    scale_floor = float(np.clip(params.min_residual_scale, 1e-4, 3.0))
    scale_grid = [scale_floor, 0.1, 0.2, 0.3, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 2.5, 3.0]
    scale_grid = sorted(set(float(np.clip(s, scale_floor, 3.0)) for s in scale_grid))
    persistence_val_mae = float(np.mean(np.abs(base_val - yva)))
    persistence_val_mape = _compute_mape(yva, base_val, eps=1e-6)
    # Selection should always compare against naive last-observation persistence,
    # even when residual baseline uses seasonal13.
    base_val_naive = np.repeat(xva[:, -1:, :y_dim], horizon, axis=1)
    persistence_naive_val_mae = float(np.mean(np.abs(base_val_naive - yva)))
    persistence_naive_val_mape = _compute_mape(yva, base_val_naive, eps=1e-6)
    default_tau = 0.5 if str(params.qrc_target).strip().lower() == "norm_resid" else 0.05
    tau = float(params.tau if np.isfinite(params.tau) else default_tau)

    if str(params.qrc_mode).strip().lower() == "auto":
        # For residual-like targets, absolute mode frequently amplifies noise.
        # Feedback-enabled runs should prefer residual mode first; gated mode
        # often over-damps to persistence and can win penalty score despite worse MAPE.
        target_mode = str(params.qrc_target).strip().lower()
        if target_mode in {"delta", "norm_resid", "log_return"}:
            if bool(params.feedback_enabled) and float(params.gain) > 0.0:
                candidate_modes = ["residual", "gated_residual"]
            else:
                candidate_modes = ["gated_residual", "residual"]
        else:
            candidate_modes = ["residual", "gated_residual", "absolute"]
    else:
        candidate_modes = [str(params.qrc_mode).strip().lower()]

    mode_results: list[dict[str, Any]] = []
    training_curve_records: list[dict[str, Any]] = []
    epoch_counter = 0
    target_mode_norm = str(params.qrc_target).strip().lower()
    select_by_mae = bool(target_mode_norm in {"delta", "norm_resid", "log_return"})

    def _rank_tuple(rec: dict[str, Any]) -> tuple[float, float, float]:
        if select_by_mae:
            return (float(rec["val_mae"]), float(rec["val_mape"]), float(rec["score"]))
        return (float(rec["val_mape"]), float(rec["val_mae"]), float(rec["score"]))

    for mode in candidate_modes:
        y_tr_target = y_train_abs if mode == "absolute" else y_train_res
        y_scaler = StandardScaler().fit(y_tr_target)
        y_tr_s = y_scaler.transform(y_tr_target)
        y_tr_s = np.clip(np.nan_to_num(y_tr_s, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)

        best_for_mode: dict[str, Any] | None = None
        for alpha in alpha_grid:
            ridge = Ridge(alpha=float(alpha), random_state=int(params.seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    ridge.fit(x_phi, y_tr_s)
                    pred_train_s = ridge.predict(x_phi)
                    pred_val_s = ridge.predict(x_phi_val)
            if (not np.isfinite(pred_train_s).all()) or (not np.isfinite(pred_val_s).all()):
                continue

            pred_target_train = y_scaler.inverse_transform(pred_train_s).reshape(n_train, horizon, y_dim)
            pred_target_train = np.clip(
                np.nan_to_num(pred_target_train, nan=0.0, posinf=25.0, neginf=-25.0),
                -25.0,
                25.0,
            )
            pred_target_val = y_scaler.inverse_transform(pred_val_s).reshape(xva.shape[0], horizon, y_dim)
            pred_target_val = np.clip(
                np.nan_to_num(pred_target_val, nan=0.0, posinf=25.0, neginf=-25.0),
                -25.0,
                25.0,
            )

            if mode == "absolute":
                if y_positive_transform is not None:
                    pred_train = y_positive_transform.clamp(y_positive_transform.inverse(pred_target_train))
                    pred_val = y_positive_transform.clamp(y_positive_transform.inverse(pred_target_val))
                else:
                    pred_train = pred_target_train
                    pred_val = pred_target_val
                train_mape = _compute_mape(ytr, pred_train, eps=1e-6)
                train_mae = _compute_mae(ytr, pred_train)
                val_mape = _compute_mape(yva, pred_val, eps=1e-6)
                val_mae = _compute_mae(yva, pred_val)
                val_delta = float(np.mean(np.abs(pred_val - base_val)))
                collapse_penalty = float(max(0.0, 5e-5 - val_delta) * 5e3)
                persist_penalty = float(max(0.0, val_mape - persistence_val_mape) * 5.0)
                naive_penalty = float(max(0.0, val_mape - persistence_naive_val_mape) * 10.0)
                pred_target_flat = pred_val.reshape(xva.shape[0], -1)
                true_target_flat = y_val_abs
                std_ratio = float(np.std(pred_target_flat) / (np.std(true_target_flat) + 1e-12))
                resid_penalty = float(max(0.0, std_ratio - 1.2) * float(params.residvar_penalty) * 10.0)
                mode_penalty = float(0.25 * val_mape) if str(params.qrc_target).strip().lower() != "level" else 0.0
                target_resid_mae = float(np.mean(np.abs(pred_target_flat - true_target_flat)))
                cand = {
                    "mode": mode,
                    "ridge": ridge,
                    "y_scaler": y_scaler,
                    "alpha": float(alpha),
                    "residual_scale": 1.0,
                    "train_mape": float(train_mape),
                    "train_mae": float(train_mae),
                    "val_mape": float(val_mape),
                    "val_mae": float(val_mae),
                    "val_delta": float(val_delta),
                    "residual_std_ratio": std_ratio,
                    "target_residual_mae": target_resid_mae,
                    "score": float(val_mape + collapse_penalty + persist_penalty + naive_penalty + resid_penalty + mode_penalty),
                    "val_pred": pred_val,
                }
                epoch_counter += 1
                training_curve_records.append(
                    {
                        "epoch": int(epoch_counter),
                        "mode": str(mode),
                        "alpha": float(alpha),
                        "residual_scale": 1.0,
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        "train_mape": float(train_mape),
                        "val_mape": float(val_mape),
                        "score": float(cand["score"]),
                    }
                )
                if (best_for_mode is None) or (_rank_tuple(cand) < _rank_tuple(best_for_mode)):
                    best_for_mode = cand
            else:
                pred_res_train = pred_target_train
                pred_res = pred_target_val
                tau_mode = float(tau)
                if mode == "gated_residual":
                    target_scale = float(np.std(y_train_res))
                    # Scale tau to target residual amplitude so gating is not effectively always 1.
                    tau_mode = float(max(tau, 0.20 * max(1e-6, target_scale)))
                if mode == "gated_residual":
                    gate_train = np.abs(pred_res_train) / (np.abs(pred_res_train) + float(max(1e-12, tau_mode)))
                    gate_train = np.clip(np.nan_to_num(gate_train, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    pred_res_train = gate_train * pred_res_train
                    gate = np.abs(pred_res) / (np.abs(pred_res) + float(max(1e-12, tau_mode)))
                    gate = np.clip(np.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    pred_res = gate * pred_res
                for s in scale_grid:
                    pred_target_train_s = float(s) * pred_res_train
                    pred_target = float(s) * pred_res
                    if params.resid_clip is not None:
                        pred_target_train_s = np.clip(
                            pred_target_train_s,
                            -abs(float(params.resid_clip)),
                            abs(float(params.resid_clip)),
                        )
                        pred_target = np.clip(pred_target, -abs(float(params.resid_clip)), abs(float(params.resid_clip)))
                    pred_train = proto._level_from_target(pred_target_train_s, baseline=base_train, x_seq=xtr)
                    pred_val = proto._level_from_target(pred_target, baseline=base_val, x_seq=xva)
                    train_mape = _compute_mape(ytr, pred_train, eps=1e-6)
                    train_mae = _compute_mae(ytr, pred_train)
                    val_mape = _compute_mape(yva, pred_val, eps=1e-6)
                    val_mae = _compute_mae(yva, pred_val)
                    val_delta = float(np.mean(np.abs(pred_val - base_val)))
                    collapse_penalty = float(max(0.0, 5e-5 - val_delta) * 5e3)
                    persist_penalty = float(max(0.0, val_mape - persistence_val_mape) * 5.0)
                    true_target = y_val_target.reshape(xva.shape[0], horizon, y_dim)
                    std_ratio = float(np.std(pred_target) / (np.std(true_target) + 1e-12))
                    resid_penalty = float(max(0.0, std_ratio - 1.2) * float(params.residvar_penalty) * 10.0)
                    naive_penalty = float(max(0.0, val_mape - persistence_naive_val_mape) * 8.0)
                    target_resid_mae = float(np.mean(np.abs(pred_target - true_target)))
                    cand = {
                        "mode": mode,
                        "ridge": ridge,
                        "y_scaler": y_scaler,
                        "alpha": float(alpha),
                        "residual_scale": float(s),
                        "train_mape": float(train_mape),
                        "train_mae": float(train_mae),
                        "val_mape": float(val_mape),
                        "val_mae": float(val_mae),
                        "val_delta": float(val_delta),
                        "residual_std_ratio": std_ratio,
                        "target_residual_mae": target_resid_mae,
                        "tau_used": float(tau_mode),
                        "score": float(val_mape + collapse_penalty + persist_penalty + naive_penalty + resid_penalty),
                        "val_pred": pred_val,
                    }
                    epoch_counter += 1
                    training_curve_records.append(
                        {
                            "epoch": int(epoch_counter),
                            "mode": str(mode),
                            "alpha": float(alpha),
                            "residual_scale": float(s),
                            "train_mae": float(train_mae),
                            "val_mae": float(val_mae),
                            "train_mape": float(train_mape),
                            "val_mape": float(val_mape),
                            "score": float(cand["score"]),
                        }
                    )
                    if (best_for_mode is None) or (_rank_tuple(cand) < _rank_tuple(best_for_mode)):
                        best_for_mode = cand
        if best_for_mode is None:
            raise RuntimeError(f"photonic_qrc mode={mode} ridge selection failed.")
        pred_vs_persist = np.asarray(best_for_mode["val_pred"], dtype=float) - base_val
        collapse_delta = float(np.mean(np.abs(pred_vs_persist)))
        best_for_mode["collapse_delta"] = collapse_delta
        best_for_mode["collapsed"] = bool(collapse_delta <= 1e-8)
        mode_results.append(best_for_mode)

    # Primary objective is validation MAPE; keep score as tie-breaker so
    # anti-collapse penalties still influence close choices.
    mode_results = sorted(mode_results, key=_rank_tuple)
    improving_noncollapsed = [
        r
        for r in mode_results
        if (not bool(r["collapsed"])) and float(r["val_mape"]) < persistence_val_mape - 1e-8
    ]
    noncollapsed = [r for r in mode_results if not bool(r["collapsed"])]
    if improving_noncollapsed:
        improving_noncollapsed = sorted(improving_noncollapsed, key=_rank_tuple)
        chosen = improving_noncollapsed[0]
    elif noncollapsed:
        noncollapsed = sorted(noncollapsed, key=_rank_tuple)
        chosen = noncollapsed[0]
    else:
        chosen = sorted(mode_results, key=_rank_tuple)[0]
    # Guard against selecting absolute if it underperforms naive persistence.
    if (
        str(chosen.get("mode", "")).strip().lower() == "absolute"
        and float(chosen.get("val_mape", np.inf)) > float(persistence_naive_val_mape)
    ):
        alt = [
            r
            for r in mode_results
            if str(r.get("mode", "")).strip().lower() in {"residual", "gated_residual"}
            and not bool(r.get("collapsed", False))
        ]
        if alt:
            chosen = sorted(alt, key=_rank_tuple)[0]

    collapse_detected = bool(chosen.get("collapsed", False))
    if collapse_detected:
        warnings.warn(
            "photonic_qrc collapse detected across configured modes; returning best-available model and marking "
            "collapse diagnostics for downstream reporting.",
            RuntimeWarning,
        )

    target_std = float(np.std(y_train_abs if str(chosen["mode"]) == "absolute" else y_train_res))
    residual_clip = float(max(1.0, 6.0 * target_std))

    model = PhotonicQRCForecaster(
        ridge=chosen["ridge"],
        x_scaler=x_scaler,
        phi_scaler=phi_scaler,
        y_scaler=chosen["y_scaler"],
        y_dim=y_dim,
        horizon=horizon,
        d_factors=d_core,
        params=params,
        proj_mean=proj_mean,
        proj_comp=proj_comp,
        u0=u0,
        phase_mix=phase_mix,
        phase_init=phase_init,
        idx_in=idx_in,
        idx_budget=idx_budget,
        w_input=w_input,
        w_vr=w_vr,
        w_vi=w_vi,
        b_vr=b_vr,
        b_vi=b_vi,
        w_fb=w_fb,
        pair_i=pair_i,
        pair_j=pair_j,
        triple_i=triple_i,
        triple_j=triple_j,
        triple_k=triple_k,
        budget_weights=budget_weights,
        qrc_mode=str(chosen["mode"]),
        qrc_target=str(params.qrc_target),
        qrc_baseline=str(baseline_effective),
        target_transform=str(params.target_transform),
        y_positive_transform=y_positive_transform,
        tau=float(chosen.get("tau_used", tau)),
        residual_scale=float(chosen["residual_scale"]),
        residual_clip=float(residual_clip),
        target_eps=1e-6,
        log_offset=log_offset,
        higher_order=int(params.higher_order),
        mode_val_mape={str(r["mode"]): float(r["val_mape"]) for r in mode_results},
        feature_stats=feature_stats,
        train_qevals=int((xtr.shape[0] + xva.shape[0]) * xtr.shape[1]),
        train_total_shots=int((xtr.shape[0] + xva.shape[0]) * xtr.shape[1] * int(max(1, params.shots))),
        shots_per_eval=int(max(1, params.shots)),
    )
    if isinstance(model.feature_stats, dict):
        model.feature_stats["collapse_detected"] = bool(collapse_detected)
        model.feature_stats["qrc_target"] = str(params.qrc_target)
        model.feature_stats["target_transform"] = str(params.target_transform)
        model.feature_stats["y_floor_mode"] = str(params.y_floor_mode)
        if y_positive_transform is not None:
            model.feature_stats["y_floor"] = float(y_positive_transform.floor)
            model.feature_stats["y_shift"] = float(y_positive_transform.shift)
        model.feature_stats["qrc_baseline"] = str(baseline_effective)
        model.feature_stats["qrc_baseline_requested"] = str(baseline_requested)
        model.feature_stats["qrc_baseline_effective"] = str(baseline_effective)
        model.feature_stats["qrc_mode_used"] = str(chosen["mode"])
        model.feature_stats["qrc_gate_tau"] = float(tau)
        model.feature_stats["residual_std_ratio"] = float(chosen.get("residual_std_ratio", np.nan))
        model.feature_stats["target_residual_mae_val"] = float(chosen.get("target_residual_mae", np.nan))
        model.feature_stats["training_curve_epochs"] = int(len(training_curve_records))
        if training_curve_records:
            train_mae_arr = np.asarray([float(r.get("train_mae", np.nan)) for r in training_curve_records], dtype=np.float64)
            val_mae_arr = np.asarray([float(r.get("val_mae", np.nan)) for r in training_curve_records], dtype=np.float64)
            model.feature_stats["training_curve_best_train_mae"] = float(np.nanmin(train_mae_arr))
            model.feature_stats["training_curve_best_val_mae"] = float(np.nanmin(val_mae_arr))
    model.val_target_residual_mae = float(chosen.get("target_residual_mae", np.nan))
    model.training_curve_ = list(training_curve_records)

    val_pred = model.predict(xva)
    val_mae = _compute_mae(yva, val_pred)
    val_mape = _compute_mape(yva, val_pred, eps=1e-6)
    delta_vs_persist = float(np.mean(np.abs(val_pred - base_val)))
    if delta_vs_persist <= 1e-8:
        warnings.warn(
            "photonic_qrc selected mode remains near-persistence on validation; downstream collapse diagnostics "
            "will mark this run as FAILED.",
            RuntimeWarning,
        )
        if isinstance(model.feature_stats, dict):
            model.feature_stats["collapse_detected"] = True
    model.feedback_sanity = model.run_feedback_sanity(xva, max_n=64)
    if isinstance(model.feature_stats, dict):
        model.feature_stats["feedback_sanity"] = dict(model.feedback_sanity or {})
    print(
        "[photonic_qrc] "
        f"variant={'feedback' if params.feedback_enabled and params.gain > 0 else 'no_feedback'} "
        f"M={m} Nph={int(params.n_photons)} B={int(idx_budget.size)} "
        f"feature={params.feature} shots={int(params.shots)} in_pca={in_pca} "
        f"pseudocount={float(params.pseudocount):.3g} "
        f"d_core={int(d_core)} factor_cap={int(params.factor_cap)} "
        f"feat_norm={params.feat_norm} "
        f"mode={model.qrc_mode} target={model.qrc_target} baseline={model.qrc_baseline} "
        f"(requested={baseline_requested}) "
        f"tau={model.tau:.3g} alpha={float(chosen['alpha']):.3g} "
        f"residual_scale={model.residual_scale:.3f} phi_std={phi_std:.3e} "
        f"val_mape={val_mape:.6f} persistence_val_mape={persistence_val_mape:.6f} "
        f"naive_persistence_val_mape={persistence_naive_val_mape:.6f} "
        f"val_mae={val_mae:.6f} persistence_val_mae={persistence_val_mae:.6f} "
        f"naive_persistence_val_mae={persistence_naive_val_mae:.6f} "
        f"delta_vs_persist={delta_vs_persist:.3e}"
    )
    return model


def _make_windows(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    n = int(series.shape[0])
    for t in range(int(lookback), n - int(horizon)):
        x.append(series[t - int(lookback) : t])
        y.append(series[t : t + int(horizon)])
    x_arr = np.asarray(x, dtype=np.float64)[:, :, None]
    y_arr = np.asarray(y, dtype=np.float64)[:, :, None]
    return x_arr, y_arr


def _self_test(seed: int = 0) -> None:
    rng = np.random.default_rng(int(seed))
    n = 380
    lookback = 24
    horizon = 3
    s = np.zeros((n,), dtype=np.float64)
    for t in range(3, n):
        s[t] = (
            0.62 * s[t - 1]
            - 0.28 * s[t - 2]
            + 0.16 * np.sin(0.7 * s[t - 3])
            + 0.10 * np.cos(0.12 * float(t))
            + 0.05 * rng.normal()
        )
    x, y = _make_windows(s, lookback=lookback, horizon=horizon)
    n_train = int(0.7 * x.shape[0])
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    params = PhotonicQRCParams(
        modes=12,
        n_photons=2,
        budget=24,
        gain=0.55,
        feature="coincidence",
        shots=64,
        ridge_alpha=1e-3,
        in_pca=1,
        seed=int(seed),
        feedback_enabled=True,
    )
    model1 = train_photonic_qrc_forecaster(x_train, y_train, x_val, y_val, params=params)
    pred1 = model1.predict(x_val)
    model2 = train_photonic_qrc_forecaster(x_train, y_train, x_val, y_val, params=params)
    pred2 = model2.predict(x_val)

    per = np.repeat(x_val[:, -1:, :], horizon, axis=1)
    mae_q = float(np.mean(np.abs(pred1 - y_val)))
    mae_p = float(np.mean(np.abs(per - y_val)))
    det_diff = float(np.max(np.abs(pred1 - pred2)))
    print(f"[self-test photonic_qrc] mae_q={mae_q:.6f} mae_persistence={mae_p:.6f} det_maxdiff={det_diff:.3e}")
    if not np.isfinite(mae_q) or not np.isfinite(det_diff):
        raise RuntimeError("photonic_qrc self-test produced non-finite outputs.")
    if det_diff > 1e-10:
        raise RuntimeError(f"photonic_qrc determinism check failed (max diff {det_diff:.3e}).")
    if not (mae_q < mae_p):
        print(
            "[self-test photonic_qrc] warning: did not beat persistence on this draw; "
            "model remains valid if deterministic and finite."
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Photonic QRC feedback baseline self-test.")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.self_test):
        _self_test(seed=int(args.seed))
        return
    print("Run with --self_test to execute the deterministic synthetic sanity check.")


if __name__ == "__main__":
    main()
