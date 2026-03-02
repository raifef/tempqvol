from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

FeatureType = Literal["coincidence", "clickprob", "pseudo_coincidence"]
QRCMode = Literal["residual", "direct", "auto", "gated_residual", "absolute"]
FeatNormMode = Literal["none", "standard"]
QRCTargetMode = Literal["level", "delta", "log_return", "norm_resid"]
QRCBaselineMode = Literal["persistence", "persistence_naive", "seasonal13"]
BudgetAllocMode = Literal["uniform", "random", "weighted"]
TargetTransformMode = Literal["none", "log"]
DEFAULT_PQRC_SHOTS = 32


def _sanitize_shots(shots: int, *, field_name: str) -> int:
    s = int(shots)
    if s >= 0:
        return s
    warnings.warn(
        f"{field_name}={s} is negative; forcing shots={DEFAULT_PQRC_SHOTS}.",
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


@dataclass
class _PositiveShiftLog:
    floor_mode: str = "train_p001"
    floor_value: float | None = None
    eps_min: float = 1e-8
    floor: float = 1e-8
    shift: float = 0.0

    def fit(self, values: np.ndarray) -> "_PositiveShiftLog":
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            self.floor = float(self.eps_min)
            self.shift = 0.0
            return self
        mode = str(self.floor_mode).strip().lower()
        if mode == "fixed":
            raw_floor = float(self.floor_value if self.floor_value is not None else self.eps_min)
        elif mode == "train_p01":
            raw_floor = float(np.percentile(v, 1.0))
        else:
            raw_floor = float(np.percentile(v, 0.1))
        floor = float(max(self.eps_min, raw_floor))
        shift = float(max(0.0, -float(np.min(v)) + floor))
        self.floor = floor
        self.shift = shift
        return self

    def forward(self, y: np.ndarray) -> np.ndarray:
        y2 = np.asarray(y, dtype=np.float64)
        yp = np.maximum(y2 + float(self.shift), float(self.floor))
        return np.log(yp)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z2 = np.asarray(z, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.exp(np.clip(z2, -50.0, 50.0)) - float(self.shift)
        return np.asarray(y, dtype=np.float64)

    def clamp(self, y: np.ndarray) -> np.ndarray:
        y2 = np.asarray(y, dtype=np.float64)
        min_level = float(self.floor - self.shift)
        return np.maximum(y2, min_level)


@dataclass
class PhotonicMemoryParams:
    modes: int = 12
    n_photons: int = 2
    budget: int = 32
    gain: float = 0.5
    feature: FeatureType = "coincidence"
    shots: int = DEFAULT_PQRC_SHOTS
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
        self.modes = int(np.clip(int(self.modes), 4, 32))
        self.n_photons = int(max(1, int(self.n_photons)))
        self.budget = int(max(1, int(self.budget)))
        self.gain = float(max(0.0, float(self.gain)))
        self.shots = _sanitize_shots(self.shots, field_name="PhotonicMemoryParams.shots")
        self.ridge_alpha = float(max(1e-10, float(self.ridge_alpha)))
        self.tau = float(np.clip(float(self.tau), 1e-8, 100.0))
        self.residvar_penalty = float(np.clip(float(self.residvar_penalty), 0.0, 10.0))
        self.min_residual_scale = float(np.clip(float(self.min_residual_scale), 1e-4, 2.0))
        self.input_scale = float(np.clip(float(self.input_scale), 1e-5, 50.0))
        self.in_pca = int(max(1, int(self.in_pca)))
        self.factor_cap = int(max(0, int(self.factor_cap)))
        self.higher_order = int(np.clip(int(self.higher_order), 0, 4))
        feature = str(self.feature).strip().lower()
        if feature not in {"coincidence", "clickprob", "pseudo_coincidence"}:
            raise ValueError("feature must be one of: coincidence, clickprob, pseudo_coincidence.")
        self.feature = feature  # type: ignore[assignment]
        mode = str(self.qrc_mode).strip().lower()
        if mode not in {"residual", "direct", "auto", "gated_residual", "absolute"}:
            raise ValueError("qrc_mode must be one of: residual, direct, auto, gated_residual, absolute.")
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
        if ba == "weighted":
            ba = "uniform"
        if ba not in {"uniform", "random"}:
            raise ValueError("budget_alloc must be one of: uniform, random, weighted.")
        self.budget_alloc = ba  # type: ignore[assignment]


@dataclass
class PhotonicMemoryForecaster:
    ridge: Ridge
    x_scaler: StandardScaler
    phi_scaler: Any
    y_scaler: StandardScaler
    y_dim: int
    horizon: int
    params: PhotonicMemoryParams
    d_factors: int
    proj_mean: np.ndarray
    proj_comp: np.ndarray
    u0: np.ndarray
    phi0: np.ndarray
    w_in: np.ndarray
    v_fb: np.ndarray
    wedge_idxs: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    qrc_mode: str = "residual"
    qrc_target: str = "delta"
    qrc_baseline: str = "persistence"
    target_transform: str = "none"
    y_positive_transform: _PositiveShiftLog | None = None
    norm_scale: np.ndarray | None = None
    log_offset: np.ndarray | None = None
    residual_scale: float = 1.0
    residual_clip: float | None = None
    feature_stats: dict[str, Any] | None = None
    feedback_sanity: dict[str, Any] | None = None
    train_qevals: int = 0
    train_total_shots: int = 0
    infer_qevals: int = 0
    infer_total_shots: int = 0
    shots_per_eval: int = DEFAULT_PQRC_SHOTS
    use_feedback_delta: bool = False
    last_q_features_: np.ndarray | None = None
    last_gate_: np.ndarray | None = None
    val_target_residual_mae: float | None = None

    def _baseline_from_x(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
        n, l, _ = x.shape
        h = int(self.horizon)
        y_dim = int(self.y_dim)
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
        if self.norm_scale is not None:
            return np.asarray(self.norm_scale, dtype=np.float64)[None, :]
        x = np.asarray(x_seq, dtype=np.float64)
        s = np.std(x[:, :, : self.y_dim], axis=1)
        s = np.clip(np.nan_to_num(s, nan=1e-6, posinf=50.0, neginf=1e-6), 1e-6, 50.0)
        return s

    def _target_from_level(self, y_level: np.ndarray, baseline: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            y_log = self.y_positive_transform.forward(np.asarray(y_level, dtype=np.float64))
            b_log = self.y_positive_transform.forward(np.asarray(baseline, dtype=np.float64))
            if str(self.qrc_target).strip().lower() == "level":
                return y_log
            return y_log - b_log
        mode = str(self.qrc_target).strip().lower()
        y = np.asarray(y_level, dtype=np.float64)
        b = np.asarray(baseline, dtype=np.float64)
        if mode == "level":
            return y
        if mode == "delta":
            return y - b
        if mode == "norm_resid":
            scale = self._norm_scale_from_x(x_seq)
            return (y - b) / np.maximum(scale[:, None, :], 1e-6)
        offset = np.asarray(self.log_offset if self.log_offset is not None else np.zeros((self.y_dim,), dtype=np.float64), dtype=np.float64)
        num = np.maximum(y + offset[None, None, :], 1e-8)
        den = np.maximum(b + offset[None, None, :], 1e-8)
        return np.log(num / den)

    def _level_from_target(self, y_target: np.ndarray, baseline: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            b_log = self.y_positive_transform.forward(np.asarray(baseline, dtype=np.float64))
            if str(self.qrc_target).strip().lower() == "level":
                y_log = np.asarray(y_target, dtype=np.float64)
            else:
                y_log = b_log + np.asarray(y_target, dtype=np.float64)
            out = self.y_positive_transform.inverse(y_log)
            return self.y_positive_transform.clamp(out)
        mode = str(self.qrc_target).strip().lower()
        yt = np.asarray(y_target, dtype=np.float64)
        b = np.asarray(baseline, dtype=np.float64)
        if mode == "level":
            return yt
        if mode == "delta":
            return b + yt
        if mode == "norm_resid":
            scale = self._norm_scale_from_x(x_seq)
            return b + yt * scale[:, None, :]
        offset = np.asarray(self.log_offset if self.log_offset is not None else np.zeros((self.y_dim,), dtype=np.float64), dtype=np.float64)
        return np.exp(np.clip(yt, -30.0, 30.0)) * np.maximum(b + offset[None, None, :], 1e-8) - offset[None, None, :]

    def _direct_target_from_level(self, y_level: np.ndarray) -> np.ndarray:
        y = np.asarray(y_level, dtype=np.float64)
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            return self.y_positive_transform.forward(y)
        return y

    def _level_from_direct_target(self, y_direct: np.ndarray) -> np.ndarray:
        y = np.asarray(y_direct, dtype=np.float64)
        if str(self.target_transform).strip().lower() == "log" and self.y_positive_transform is not None:
            out = self.y_positive_transform.inverse(y)
            return self.y_positive_transform.clamp(out)
        return y

    def _input_sequence(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
        if x.shape[2] < self.d_factors:
            raise ValueError(f"x_seq dim {x.shape[2]} < required d_factors={self.d_factors}.")
        x_core = np.clip(np.nan_to_num(x[:, :, : self.d_factors], nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        n, l, d = x_core.shape
        x_s = self.x_scaler.transform(x_core.reshape(-1, d)).reshape(n, l, d)
        u_seq = _project_linear(x_s.reshape(-1, d), self.proj_mean, self.proj_comp).reshape(n, l, self.proj_comp.shape[1])
        return np.clip(np.nan_to_num(u_seq, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)

    def _measurement_features(self, p: np.ndarray, *, seed_token: int) -> np.ndarray:
        p2 = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        feature_mode = str(self.params.feature).strip().lower()
        if feature_mode == "clickprob":
            c_base = p2
        elif feature_mode == "pseudo_coincidence":
            c_base = p2[self.pair_i] * p2[self.pair_j]
        else:
            c_base = p2[self.pair_i] * p2[self.pair_j]
        c_base = np.clip(np.nan_to_num(c_base, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if int(self.shots_per_eval) > 0:
            rng = np.random.default_rng(int(seed_token))
            shots = int(max(1, int(self.shots_per_eval)))
            total_mass = float(np.sum(c_base))
            if np.isfinite(total_mass) and total_mass > 1e-12:
                probs = np.clip(c_base / total_mass, 0.0, 1.0)
                probs = probs / max(1e-12, float(np.sum(probs)))
                counts = rng.multinomial(shots, probs)
                c_meas = counts.astype(np.float64) / float(shots)
            else:
                c_meas = np.zeros_like(c_base, dtype=np.float64)
        else:
            c_meas = c_base
        order = int(max(1, int(self.params.higher_order)))
        if order <= 1:
            return c_meas.astype(np.float64)
        pieces = [c_meas]
        for k in range(2, order + 1):
            pieces.append(np.power(c_meas, int(k)))
        return np.concatenate(pieces, axis=0).astype(np.float64)

    def _step_features(
        self,
        u_t: np.ndarray,
        phase_state: np.ndarray,
        *,
        sample_seed: int,
        feedback_state: np.ndarray | None = None,
        feedback_enabled: bool | None = None,
        gain: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi = np.asarray(phase_state, dtype=np.float64)
        phase_drive = self.w_in @ np.asarray(u_t, dtype=np.float64)
        phase_drive = np.clip(np.nan_to_num(phase_drive, nan=0.0, posinf=2.5, neginf=-2.5), -2.5, 2.5)
        phi_in = _wrap_phase(phi + phase_drive)
        a = (self.u0 @ np.exp(1j * phi_in)) / np.sqrt(float(self.params.modes))
        p = np.abs(a) ** 2
        ps = float(np.sum(p))
        if ps <= 1e-12 or (not np.isfinite(ps)):
            p = np.ones_like(p, dtype=np.float64) / float(max(1, p.shape[0]))
        else:
            p = np.clip(p / ps, 0.0, 1.0)
            p = p / max(1e-12, float(np.sum(p)))
        c_t = self._measurement_features(p, seed_token=int(sample_seed))
        c_center = c_t - float(np.mean(c_t))
        c_scale = float(np.std(c_center))
        if np.isfinite(c_scale) and c_scale > 1e-8:
            c_fb = c_center / c_scale
        else:
            c_fb = c_center

        fb_on = bool(self.params.feedback_enabled if feedback_enabled is None else feedback_enabled)
        g = float(self.params.gain if gain is None else gain)
        h_prev = np.zeros((self.wedge_idxs.size,), dtype=np.float64)
        if feedback_state is not None and self.wedge_idxs.size > 0:
            h_prev = np.asarray(feedback_state, dtype=np.float64).reshape(-1)[: self.wedge_idxs.size]
        if fb_on:
            # Carry non-wedge phases through the driven interferometer state.
            # Feedback writes a measurement-conditioned update only on wedge indices.
            phi_next = phi_in.copy()
            h_next = np.zeros((self.wedge_idxs.size,), dtype=np.float64)
            if self.wedge_idxs.size > 0:
                if g > 0.0 and self.v_fb.size > 0:
                    raw = self.v_fb @ c_fb
                    if h_prev.size == raw.size:
                        raw = raw + 0.35 * h_prev
                    raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
                    h = np.tanh(raw)
                    h_next = h[: self.wedge_idxs.size]
                    phi_next[self.wedge_idxs] = _wrap_phase(self.phi0[self.wedge_idxs] + float(g) * h[: self.wedge_idxs.size])
                else:
                    phi_next[self.wedge_idxs] = self.phi0[self.wedge_idxs]
        else:
            phi_next = self.phi0.copy()
            h_next = np.zeros((self.wedge_idxs.size,), dtype=np.float64)
        return c_t.astype(np.float64), phi_next.astype(np.float64), h_next.astype(np.float64)

    def _window_states(self, x_seq: np.ndarray, *, feedback_enabled: bool | None = None, gain: float | None = None) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
        fb_on = bool(self.params.feedback_enabled if feedback_enabled is None else feedback_enabled)
        n, l, _ = x.shape
        u_seq = self._input_sequence(x)
        c_dim = int(max(1, self.v_fb.shape[1] if self.v_fb.ndim == 2 and self.v_fb.shape[1] > 0 else self.params.modes))
        w_dim = int(self.wedge_idxs.size)
        out = np.zeros((n, 2 * c_dim + 3 * w_dim + 2), dtype=np.float64)
        for i in range(n):
            phi = self.phi0.copy()
            c_last = np.zeros((c_dim,), dtype=np.float64)
            c_sum = np.zeros((c_dim,), dtype=np.float64)
            h_state = np.zeros((w_dim,), dtype=np.float64)
            h_last = np.zeros((w_dim,), dtype=np.float64)
            h_sum = np.zeros((w_dim,), dtype=np.float64)
            for t in range(l):
                c_t, phi, h_next = self._step_features(
                    u_t=float(self.params.input_scale) * u_seq[i, t],
                    phase_state=phi,
                    sample_seed=int(self.params.seed + 1_000_003 * (i + 1) + 7_919 * (t + 1)),
                    feedback_state=h_state,
                    feedback_enabled=feedback_enabled,
                    gain=gain,
                )
                c_last = c_t
                c_sum += c_t
                h_state = h_next
                h_last = h_next
                if w_dim > 0:
                    h_sum += h_next
            c_mean = c_sum / float(max(1, l))
            if w_dim > 0:
                phi_drift = _wrap_phase(phi[self.wedge_idxs] - self.phi0[self.wedge_idxs]) / np.pi
                h_mean = h_sum / float(max(1, l))
            else:
                phi_drift = np.zeros((0,), dtype=np.float64)
                h_mean = np.zeros((0,), dtype=np.float64)
            if not fb_on:
                # No-feedback baseline intentionally uses only instantaneous readout.
                c_mean = np.zeros_like(c_mean)
            phi_stats = np.array(
                [
                    float(np.mean(np.abs(phi_drift))) if phi_drift.size else 0.0,
                    float(np.std(phi_drift)) if phi_drift.size else 0.0,
                ],
                dtype=np.float64,
            )
            out[i] = np.concatenate([c_last, c_mean, phi_drift, phi_stats, h_last, h_mean], axis=0)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        n = int(x.shape[0])
        l = int(x.shape[1])
        phi_fb = self._window_states(x)
        if bool(self.use_feedback_delta) and bool(self.params.feedback_enabled) and float(self.params.gain) > 0.0:
            phi_nf = self._window_states(x, feedback_enabled=False, gain=0.0)
            phi = np.concatenate([phi_fb, phi_fb - phi_nf], axis=1)
        else:
            phi = phi_fb
        self.last_q_features_ = np.asarray(phi, dtype=np.float64)
        eval_multiplier = 2 if (bool(self.use_feedback_delta) and bool(self.params.feedback_enabled) and float(self.params.gain) > 0.0) else 1
        qevals = int(max(0, n * l * eval_multiplier))
        shots = int(max(0, self.shots_per_eval))
        self.infer_qevals += qevals
        self.infer_total_shots += int(qevals * shots)

        phi_s = self.phi_scaler.transform(phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred_s = self.ridge.predict(phi_s)
        pred_target = self.y_scaler.inverse_transform(pred_s).reshape(n, self.horizon, self.y_dim)
        pred_target = np.clip(np.nan_to_num(pred_target, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)

        if str(self.qrc_mode).strip().lower() == "direct":
            self.last_gate_ = np.ones_like(pred_target, dtype=np.float64)
            return self._level_from_direct_target(pred_target)

        if self.residual_clip is not None:
            pred_target = np.clip(pred_target, -abs(float(self.residual_clip)), abs(float(self.residual_clip)))
        self.last_gate_ = np.ones_like(pred_target, dtype=np.float64)
        y_target = float(self.residual_scale) * pred_target
        base = self._baseline_from_x(x)
        out = self._level_from_target(y_target=y_target, baseline=base, x_seq=x)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

    def run_feedback_sanity(self, x_seq: np.ndarray, max_n: int = 64) -> dict[str, Any]:
        x = np.asarray(x_seq, dtype=np.float64)
        if x.ndim != 3 or x.shape[0] <= 0:
            return {
                "feedback_enabled": bool(self.params.feedback_enabled),
                "pqrc_gain": float(self.params.gain),
                "checksum_gain0": "",
                "checksum_gain1": "",
                "phi_diff_gain0_vs_gain1": 0.0,
                "feedback_inert": True,
            }
        n = int(max(1, min(int(max_n), x.shape[0])))
        xs = x[:n]
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
                "photonic_memory feedback appears inert: gain=0 and gain=1 produce near-identical features.",
                RuntimeWarning,
            )
        return out


def _build_proto_from_x(
    x_ref: np.ndarray,
    *,
    y_dim: int,
    horizon: int,
    d_factors: int,
    params: PhotonicMemoryParams,
) -> PhotonicMemoryForecaster:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("photonic_memory requires scikit-learn.")
    x = np.asarray(x_ref, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x_ref must be [N,L,D], got {x.shape}.")
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    _, _, x_dim = x.shape
    d_raw = int(max(1, min(int(d_factors), int(x_dim))))
    if int(params.factor_cap) > 0:
        d_core = int(max(1, min(d_raw, int(params.factor_cap))))
    else:
        d_core = int(d_raw)
    in_pca = int(np.clip(int(params.in_pca), 1, min(16, d_core)))
    modes = int(params.modes)

    x_core = x[:, :, :d_core]
    x_scaler = StandardScaler().fit(x_core.reshape(-1, d_core))
    x_s = x_scaler.transform(x_core.reshape(-1, d_core))
    proj_mean, proj_comp = _fit_linear_projector(x_s, n_comp=in_pca)

    rng = np.random.default_rng(int(params.seed))
    u0 = _haar_like_unitary(modes, seed=int(params.seed))
    phi0 = rng.uniform(-np.pi, np.pi, size=(modes,)).astype(np.float64)
    w_in = rng.normal(0.0, 1.0 / np.sqrt(max(1, in_pca)), size=(modes, in_pca)).astype(np.float64)

    pair_i, pair_j = _pair_indices(modes)
    base_c_dim = int(modes) if str(params.feature).strip().lower() == "clickprob" else int(pair_i.size)
    ho = int(max(1, int(params.higher_order)))
    c_dim = int(base_c_dim * ho)

    wedge_size = int(max(1, min(int(params.budget), modes)))
    if str(params.budget_alloc).strip().lower() == "random":
        wedge_idxs = np.sort(rng.choice(modes, size=wedge_size, replace=False)).astype(int)
    else:
        wedge_idxs = np.arange(wedge_size, dtype=int)
    v_fb = rng.normal(0.0, 2.0 / np.sqrt(max(1, c_dim)), size=(wedge_size, c_dim)).astype(np.float64)

    y_scaler = StandardScaler().fit(
        np.vstack([np.zeros((1, int(y_dim * horizon))), np.ones((1, int(y_dim * horizon)))])
    )
    return PhotonicMemoryForecaster(
        ridge=Ridge(alpha=float(max(1e-8, params.ridge_alpha)), random_state=int(params.seed)),
        x_scaler=x_scaler,
        phi_scaler=_IdentityScaler(),
        y_scaler=y_scaler,
        y_dim=int(y_dim),
        horizon=int(horizon),
        params=params,
        d_factors=d_core,
        proj_mean=proj_mean,
        proj_comp=proj_comp,
        u0=u0,
        phi0=phi0,
        w_in=w_in,
        v_fb=v_fb,
        wedge_idxs=wedge_idxs,
        pair_i=pair_i,
        pair_j=pair_j,
        qrc_mode=("direct" if str(params.qrc_mode).strip().lower() in {"direct", "absolute"} else "residual"),
        qrc_target=str(params.qrc_target),
        qrc_baseline=str(params.qrc_baseline),
        target_transform=str(params.target_transform),
        residual_scale=float(params.min_residual_scale),
        residual_clip=(None if params.resid_clip is None else float(abs(float(params.resid_clip)))),
        shots_per_eval=int(max(0, params.shots)),
    )


def compute_photonic_memory_proxy(
    x_train_sub: np.ndarray,
    *,
    params: PhotonicMemoryParams,
    max_k: int = 10,
    ridge_alpha: float = 1.0,
    max_samples: int = 256,
) -> dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("photonic_memory requires scikit-learn.")
    x = np.asarray(x_train_sub, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0:
        return {"mc_score": 0.0, "stability_penalty": 0.0, "mc_score_adj": 0.0, "r2_by_k": []}
    n = int(max(8, min(int(max_samples), x.shape[0])))
    xs = np.asarray(x[:n], dtype=np.float64)
    d_core = int(max(1, min(xs.shape[2], 8)))
    proto = _build_proto_from_x(xs, y_dim=d_core, horizon=1, d_factors=d_core, params=params)
    phi = proto._window_states(xs)
    phi = np.clip(np.nan_to_num(phi, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    phi_s = StandardScaler().fit_transform(phi)

    u_seq = proto._input_sequence(xs)
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


def train_photonic_memory_forecaster(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: PhotonicMemoryParams,
    d_factors: int,
    horizon: int,
) -> PhotonicMemoryForecaster:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("photonic_memory requires scikit-learn.")

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
    if ytr.shape[1] != int(horizon) or yva.shape[1] != int(horizon):
        raise ValueError(
            f"horizon mismatch: expected {int(horizon)}, got y_train={ytr.shape[1]} y_val={yva.shape[1]}."
        )
    if ytr.shape[2] != yva.shape[2]:
        raise ValueError("Train/val target dimensions must match.")
    if xtr.shape[2] < int(d_factors) or xva.shape[2] < int(d_factors):
        raise ValueError(f"x feature dimension must be >= d_factors={int(d_factors)}.")

    y_dim = int(ytr.shape[2])
    proto = _build_proto_from_x(
        xtr,
        y_dim=y_dim,
        horizon=int(horizon),
        d_factors=int(d_factors),
        params=params,
    )
    phi_train = proto._window_states(xtr)
    phi_val = proto._window_states(xva)
    use_feedback_delta = bool(params.feedback_enabled and float(params.gain) > 0.0)
    if use_feedback_delta:
        # Differential state features isolate what the measurement-conditioned recurrence adds
        # over the no-feedback trajectory on the same window.
        phi_train_nf = proto._window_states(xtr, feedback_enabled=False, gain=0.0)
        phi_val_nf = proto._window_states(xva, feedback_enabled=False, gain=0.0)
        phi_train_delta = phi_train - phi_train_nf
        phi_val_delta = phi_val - phi_val_nf
        phi_train = np.concatenate([phi_train, phi_train_delta], axis=1)
        phi_val = np.concatenate([phi_val, phi_val_delta], axis=1)
    if not np.isfinite(phi_train).all() or not np.isfinite(phi_val).all():
        raise RuntimeError("photonic_memory produced non-finite state features.")
    feat_std_train = np.std(phi_train, axis=0)
    feat_std_val = np.std(phi_val, axis=0)
    feature_stats: dict[str, Any] = {
        "feat_std_train": feat_std_train.astype(float),
        "feat_std_val": feat_std_val.astype(float),
        "feature_std_median": float(np.median(feat_std_train)),
        "effective_d_factors": int(proto.d_factors),
        "factor_cap": int(params.factor_cap),
    }
    if use_feedback_delta:
        feature_stats["feedback_delta_mean_abs"] = float(np.mean(np.abs(phi_train_delta)))
        feature_stats["feedback_delta_std_median"] = float(np.median(np.std(phi_train_delta, axis=0)))

    n_sanity = int(max(4, min(64, xtr.shape[0])))
    x_sanity = np.asarray(xtr[:n_sanity], dtype=np.float64)
    phi_gain0 = proto._window_states(x_sanity, feedback_enabled=True, gain=0.0)
    phi_gain1 = proto._window_states(x_sanity, feedback_enabled=True, gain=1.0)
    gain_diff = float(np.mean(np.abs(phi_gain1 - phi_gain0)))
    feature_sanity = {
        "checksum_gain0": _checksum_matrix(phi_gain0),
        "checksum_gain1": _checksum_matrix(phi_gain1),
        "diff_gain0_vs_gain1": gain_diff,
        "phi_gain0_mean": float(np.mean(phi_gain0)),
        "phi_gain0_std": float(np.std(phi_gain0)),
        "phi_gain1_mean": float(np.mean(phi_gain1)),
        "phi_gain1_std": float(np.std(phi_gain1)),
    }
    feature_stats["feature_sanity"] = feature_sanity

    baseline_requested = str(params.qrc_baseline).strip().lower()
    base_train = proto._baseline_from_x(xtr)
    base_val = proto._baseline_from_x(xva)
    proto.qrc_baseline = baseline_requested

    y_positive_transform: _PositiveShiftLog | None = None
    if str(params.target_transform).strip().lower() == "log":
        y_positive_transform = _PositiveShiftLog(
            floor_mode=str(params.y_floor_mode),
            floor_value=params.y_floor_value,
            eps_min=1e-8,
        ).fit(np.concatenate([ytr.reshape(-1), base_train.reshape(-1)], axis=0))
    proto.y_positive_transform = y_positive_transform
    if str(params.qrc_target).strip().lower() == "log_return":
        min_all = np.minimum(np.min(ytr, axis=(0, 1)), np.min(base_train, axis=(0, 1)))
        log_offset = np.maximum(0.0, -min_all + 1e-3).astype(np.float64)
    else:
        log_offset = np.zeros((y_dim,), dtype=np.float64)
    proto.log_offset = log_offset
    proto.norm_scale = np.clip(np.std(ytr - base_train, axis=(0, 1)), 1e-6, 50.0)

    if str(params.feat_norm).strip().lower() == "standard":
        phi_scaler: Any = StandardScaler().fit(phi_train)
    else:
        phi_scaler = _IdentityScaler().fit(phi_train)
    x_phi = phi_scaler.transform(phi_train)
    x_phi_val = phi_scaler.transform(phi_val)
    x_phi = np.clip(np.nan_to_num(x_phi, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)
    x_phi_val = np.clip(np.nan_to_num(x_phi_val, nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)
    feature_stats["phi_train_checksum"] = _checksum_matrix(phi_train)
    feature_stats["phi_val_checksum"] = _checksum_matrix(phi_val)
    feature_stats["phi_train_norm_checksum"] = _checksum_matrix(x_phi)
    feature_stats["phi_val_norm_checksum"] = _checksum_matrix(x_phi_val)

    qrc_mode_req = str(params.qrc_mode).strip().lower()
    if qrc_mode_req == "auto":
        candidate_modes = ["residual", "direct"]
    elif qrc_mode_req in {"direct", "absolute"}:
        candidate_modes = ["direct"]
    else:
        candidate_modes = ["residual"]

    best: dict[str, Any] | None = None
    for mode in candidate_modes:
        if mode == "direct":
            y_train_target = proto._direct_target_from_level(ytr).reshape(ytr.shape[0], -1)
            y_val_target = proto._direct_target_from_level(yva).reshape(yva.shape[0], -1)
        else:
            y_train_target = proto._target_from_level(ytr, base_train, xtr).reshape(ytr.shape[0], -1)
            y_val_target = proto._target_from_level(yva, base_val, xva).reshape(yva.shape[0], -1)
        y_train_target = np.clip(np.nan_to_num(y_train_target, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
        y_val_target = np.clip(np.nan_to_num(y_val_target, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
        y_scaler = StandardScaler().fit(y_train_target)
        y_train_s = np.clip(np.nan_to_num(y_scaler.transform(y_train_target), nan=0.0, posinf=12.0, neginf=-12.0), -12.0, 12.0)

        ridge = Ridge(alpha=float(max(1e-8, params.ridge_alpha)), random_state=int(params.seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                ridge.fit(x_phi, y_train_s)
                pred_val_s = ridge.predict(x_phi_val)
        pred_val_target = y_scaler.inverse_transform(pred_val_s).reshape(yva.shape[0], int(horizon), y_dim)
        pred_val_target = np.clip(np.nan_to_num(pred_val_target, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)

        if mode == "direct":
            y_val_pred = proto._level_from_direct_target(pred_val_target)
            target_resid_mae = float(
                np.mean(np.abs(pred_val_target.reshape(yva.shape[0], -1) - y_val_target))
            )
        else:
            pred_res = float(params.min_residual_scale) * pred_val_target
            if params.resid_clip is not None:
                pred_res = np.clip(pred_res, -abs(float(params.resid_clip)), abs(float(params.resid_clip)))
            y_val_pred = proto._level_from_target(pred_res, base_val, xva)
            target_resid_mae = float(
                np.mean(
                    np.abs(
                        pred_res
                        - proto._target_from_level(yva, base_val, xva).reshape(yva.shape[0], int(horizon), y_dim)
                    )
                )
            )
        val_mape = _compute_mape(yva, y_val_pred, eps=1e-6)
        val_mae = _compute_mae(yva, y_val_pred)
        cand = {
            "mode": str(mode),
            "ridge": ridge,
            "y_scaler": y_scaler,
            "val_mape": float(val_mape),
            "val_mae": float(val_mae),
            "target_resid_mae": float(target_resid_mae),
        }
        if best is None or (cand["val_mape"], cand["val_mae"]) < (best["val_mape"], best["val_mae"]):
            best = cand
    if best is None:
        raise RuntimeError("photonic_memory failed during mode selection.")

    model = PhotonicMemoryForecaster(
        ridge=best["ridge"],
        x_scaler=proto.x_scaler,
        phi_scaler=phi_scaler,
        y_scaler=best["y_scaler"],
        y_dim=y_dim,
        horizon=int(horizon),
        params=params,
        d_factors=proto.d_factors,
        proj_mean=proto.proj_mean,
        proj_comp=proto.proj_comp,
        u0=proto.u0,
        phi0=proto.phi0,
        w_in=proto.w_in,
        v_fb=proto.v_fb,
        wedge_idxs=proto.wedge_idxs,
        pair_i=proto.pair_i,
        pair_j=proto.pair_j,
        qrc_mode=str(best["mode"]),
        qrc_target=str(params.qrc_target),
        qrc_baseline=str(params.qrc_baseline),
        target_transform=str(params.target_transform),
        y_positive_transform=y_positive_transform,
        norm_scale=proto.norm_scale,
        log_offset=log_offset,
        residual_scale=float(params.min_residual_scale),
        residual_clip=(None if params.resid_clip is None else float(abs(float(params.resid_clip)))),
        feature_stats=feature_stats,
        train_qevals=int((xtr.shape[0] + xva.shape[0]) * xtr.shape[1] * (2 if use_feedback_delta else 1)),
        train_total_shots=int((xtr.shape[0] + xva.shape[0]) * xtr.shape[1] * (2 if use_feedback_delta else 1) * int(max(0, params.shots))),
        shots_per_eval=int(max(0, params.shots)),
        use_feedback_delta=use_feedback_delta,
        val_target_residual_mae=float(best["target_resid_mae"]),
    )
    model.feedback_sanity = model.run_feedback_sanity(xva, max_n=64)
    if isinstance(model.feature_stats, dict):
        model.feature_stats["feedback_sanity"] = dict(model.feedback_sanity)
        model.feature_stats["qrc_mode_used"] = str(model.qrc_mode)
        model.feature_stats["qrc_target"] = str(model.qrc_target)
        model.feature_stats["qrc_baseline"] = str(model.qrc_baseline)
    print(
        "[photonic_memory] "
        f"variant={'feedback' if params.feedback_enabled and params.gain > 0 else 'no_feedback'} "
        f"M={int(params.modes)} B={int(model.wedge_idxs.size)} feature={params.feature} shots={int(params.shots)} "
        f"d_core={int(model.d_factors)} factor_cap={int(params.factor_cap)} "
        f"mode={model.qrc_mode} target={model.qrc_target} baseline={model.qrc_baseline} "
        f"gain={float(params.gain):.3g} feedback_enabled={bool(params.feedback_enabled)} "
        f"val_mape={float(best['val_mape']):.6f} val_mae={float(best['val_mae']):.6f} "
        f"feature_std_median={float(feature_stats['feature_std_median']):.3e}"
    )
    return model
