from __future__ import annotations

from dataclasses import dataclass
import importlib
import warnings

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - explicit runtime dependency
    raise RuntimeError("PyTorch is required for submission_swaptions.models_quantum.") from exc

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - explicit runtime dependency
    raise RuntimeError("scikit-learn is required for submission_swaptions.models_quantum.") from exc


def _load_merlin_with_quantumlayer():
    try:
        ml = importlib.import_module("merlin")
    except Exception as exc:  # pragma: no cover - requested hard failure message
        raise RuntimeError(
            "MerLin import failed. Please install merlinquantum (and perceval-quandela) to use quantum annotators."
        ) from exc

    quantum_layer = getattr(ml, "QuantumLayer", None)
    if quantum_layer is None:
        try:
            from merlin import QuantumLayer as imported_quantum_layer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Imported 'merlin' but 'QuantumLayer' is missing. This usually means a different package named "
                "'merlin' is installed in this environment. Install merlinquantum/perceval-quandela in the active "
                "venv and ensure that `import merlin; merlin.QuantumLayer` works."
            ) from exc
        setattr(ml, "QuantumLayer", imported_quantum_layer)
    return ml


ML = _load_merlin_with_quantumlayer()


@dataclass
class QuantumBudget:
    shots: int
    train_qevals: int = 0
    infer_qevals: int = 0

    def add_qevals(self, n: int, phase: str) -> None:
        n_int = int(max(0, n))
        if phase == "train":
            self.train_qevals += n_int
        elif phase == "infer":
            self.infer_qevals += n_int
        else:
            raise ValueError(f"Unknown phase: {phase}")

    @property
    def total_qevals(self) -> int:
        return int(self.train_qevals + self.infer_qevals)

    @property
    def train_total_shots(self) -> int:
        return int(self.train_qevals * self.shots)

    @property
    def infer_total_shots(self) -> int:
        return int(self.infer_qevals * self.shots)

    @property
    def total_shots(self) -> int:
        return int(self.train_total_shots + self.infer_total_shots)

    def as_dict(self) -> dict[str, int]:
        return {
            "shots_per_eval": int(self.shots),
            "train_qevals": int(self.train_qevals),
            "infer_qevals": int(self.infer_qevals),
            "total_qevals": int(self.total_qevals),
            "train_total_shots": int(self.train_total_shots),
            "infer_total_shots": int(self.infer_total_shots),
            "total_shots": int(self.total_shots),
        }


class QuantumConcreteFeatures:
    """
    Concrete quantum feature extractor for fixed MerLin circuits.

    Modes:
    - q_featuremap_features: deterministic sampled feature vectors q(u)
    - q_kernel_gram: optional kernel Gram path (stubbed via q(u) dot products)
    """

    def __init__(
        self,
        input_dim: int,
        q_dim: int = 64,
        shots: int = 256,
        angle_scale: float = 1.0,
        seed: int = 0,
        budget: QuantumBudget | None = None,
    ) -> None:
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1.")
        if int(input_dim) > 20:
            raise ValueError("MerLin QuantumLayer.simple currently supports input_dim <= 20.")
        self.input_dim = int(input_dim)
        self.q_dim = int(max(2, q_dim))
        self.shots = int(max(1, shots))
        self.angle_scale = float(angle_scale)
        self.seed = int(seed)
        self.budget = budget if budget is not None else QuantumBudget(shots=self.shots)

        self.quantum = ML.QuantumLayer.simple(input_size=int(self.input_dim))
        self.raw_output_dim = int(getattr(self.quantum, "output_size"))
        self.input_scaler: StandardScaler | None = None

        rng = np.random.default_rng(self.seed)
        self._proj_w = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, self.raw_output_dim)),
            size=(self.raw_output_dim, self.q_dim),
        )
        self._proj_b = rng.normal(0.0, 0.15, size=(self.q_dim,))

        self.last_feature_summary: dict[str, float] = {}

    def fit(self, u_train: np.ndarray) -> "QuantumConcreteFeatures":
        self.input_scaler = StandardScaler().fit(u_train)
        return self

    def _prepare_u(self, u: np.ndarray) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError(f"Expected [N, D] input vectors, got {u.shape}.")
        if u.shape[1] != self.input_dim:
            raise ValueError(f"Input dim mismatch: got {u.shape[1]}, expected {self.input_dim}.")
        if self.input_scaler is None:
            raise RuntimeError("QuantumConcreteFeatures must be fitted before transform.")

        u_std = self.input_scaler.transform(np.nan_to_num(u, nan=0.0, posinf=20.0, neginf=-20.0))
        u_std = np.clip(u_std, -8.0, 8.0)
        u_ang = np.clip(self.angle_scale * u_std, -np.pi, np.pi)
        return u_ang

    def transform(self, u: np.ndarray, phase: str = "infer", batch_size: int = 64) -> np.ndarray:
        u_ang = self._prepare_u(u)
        feats = []
        for i in range(0, u_ang.shape[0], int(max(1, batch_size))):
            xb = u_ang[i : i + int(max(1, batch_size))]
            self.budget.add_qevals(int(xb.shape[0]), phase=phase)
            x_t = torch.from_numpy(xb.astype(np.float32))
            with torch.no_grad():
                q_raw = self.quantum(x_t, shots=self.shots).detach().cpu().numpy()
            q_raw = np.nan_to_num(q_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            q_raw = np.clip(q_raw, -1.0, 1.0)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                feat = np.tanh(q_raw @ self._proj_w + self._proj_b[None, :])
            feats.append(np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0))

        q = np.vstack(feats)
        self.last_feature_summary = {
            "q_dim": float(self.q_dim),
            "q_mean": float(np.mean(q)),
            "q_std": float(np.std(q)),
            "q_min": float(np.min(q)),
            "q_max": float(np.max(q)),
        }
        return q

    def compute_kernel_gram(
        self,
        u_left: np.ndarray,
        u_right: np.ndarray | None = None,
        phase: str = "infer",
        batch_size: int = 64,
    ) -> np.ndarray:
        # Optional stub path: if MerLin kernel APIs are unavailable, use concrete-feature dot products.
        q_left = self.transform(u_left, phase=phase, batch_size=batch_size)
        q_right = q_left if u_right is None else self.transform(u_right, phase=phase, batch_size=batch_size)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = q_left @ q_right.T
        gram = np.nan_to_num(gram, nan=0.0, posinf=1e3, neginf=-1e3)
        return gram / max(1, self.q_dim)


class QuantumAnnotatorResidual:
    """
    Quantum residual annotator with concrete features and non-collapsible readout:
      residual_pred = Ridge(q_features)
      final_pred = base_pred + residual_pred
    """

    def __init__(
        self,
        input_dim: int,
        mode: str = "q_bottleneck",
        shots: int = 256,
        ridge_alphas: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0),
        qrc_feature_dim: int = 24,
        qrc_alpha: float = 0.70,
        qrc_rho: float = 0.90,
        qrc_stride: int = 1,
        train_qeval_budget: int = 5000,
        batch_size: int = 32,
        seed: int = 0,
        quantum_feature_mode: str = "concrete_features",
        q_dim: int = 64,
        q_angle_scale: float = 1.0,
        q_seed: int = 0,
        min_qfeat_std: float = 1e-8,
    ) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"q_bottleneck", "qrc"}:
            raise ValueError("mode must be one of: q_bottleneck, qrc.")
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1.")
        if int(input_dim) > 20:
            raise ValueError("MerLin QuantumLayer.simple currently supports input_dim <= 20.")

        self.input_dim = int(input_dim)
        self.mode = mode_norm
        self.shots = int(max(1, shots))
        self.ridge_alphas = tuple(float(a) for a in ridge_alphas)
        self.qrc_feature_dim = int(max(2, qrc_feature_dim))
        self.qrc_alpha = float(qrc_alpha)
        self.qrc_rho = float(qrc_rho)
        self.qrc_stride = max(1, int(qrc_stride))
        self.train_qeval_budget = max(1, int(train_qeval_budget))
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.quantum_feature_mode = str(quantum_feature_mode).strip().lower()
        self.q_dim = int(max(2, q_dim))
        self.q_angle_scale = float(q_angle_scale)
        self.q_seed = int(q_seed)
        self.min_qfeat_std = float(max(0.0, min_qfeat_std))

        self.budget = QuantumBudget(shots=self.shots)
        self._concrete = QuantumConcreteFeatures(
            input_dim=self.input_dim,
            q_dim=self.q_dim,
            shots=self.shots,
            angle_scale=self.q_angle_scale,
            seed=self.q_seed,
            budget=self.budget,
        )

        self._x_scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None
        self._ridge: Ridge | None = None
        self._residual_shape: tuple[int, int] | None = None
        self._q_input_proj: np.ndarray | None = None

        self.last_q_features_train: np.ndarray | None = None
        self.last_q_features_val: np.ndarray | None = None
        self.last_q_features_infer: np.ndarray | None = None
        self.last_kernel_gram: np.ndarray | None = None

    def _assert_non_degenerate_q_features(self, q: np.ndarray, *, stage: str) -> None:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0 or (not np.isfinite(q_arr).all()):
            raise RuntimeError(
                "Quantum model produced degenerate q_features; refusing to continue. "
                "Check encoding scale / shots / featuremap."
            )
        q_std = float(np.std(q_arr))
        if q_std <= float(self.min_qfeat_std):
            raise RuntimeError(
                "Quantum model produced degenerate q_features; refusing to continue. "
                f"Check encoding scale / shots / featuremap. stage={stage} std={q_std:.3e}"
            )

    def _project_to_input_dim(self, u: np.ndarray) -> np.ndarray:
        if u.shape[1] == self.input_dim:
            return u
        if self._q_input_proj is None:
            rng = np.random.default_rng(self.seed + 73)
            self._q_input_proj = rng.normal(
                0.0,
                1.0 / np.sqrt(max(1, u.shape[1])),
                size=(u.shape[1], self.input_dim),
            )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = np.tanh(u @ self._q_input_proj)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_u(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        if x_seq.shape[2] < 1:
            raise ValueError("x_seq has empty feature dimension.")

        if self.mode == "q_bottleneck":
            u = x_seq[:, -1, :]
        else:
            last = x_seq[:, -1, :]
            mean = x_seq.mean(axis=1)
            std = x_seq.std(axis=1)
            delta = x_seq[:, -1, :] - x_seq[:, 0, :]
            u = np.concatenate([last, mean, std, delta], axis=1)
        u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)
        u = np.clip(u, -10.0, 10.0)
        return self._project_to_input_dim(u)

    def _extract_q_features(self, x_seq: np.ndarray, phase: str) -> np.ndarray:
        u = self._build_u(x_seq)
        mode = self.quantum_feature_mode
        if mode in {"concrete_features", "q_featuremap_features"}:
            q = self._concrete.transform(u, phase=phase, batch_size=self.batch_size)
        elif mode == "q_kernel_gram":
            # Kernel-gram path stub: derive features as K(u, train_anchor) in fit.
            q = self._concrete.transform(u, phase=phase, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unsupported quantum_feature_mode: {self.quantum_feature_mode}")
        return np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)

    def _fit_ridge_with_val(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Ridge:
        best: Ridge | None = None
        best_mae = float("inf")
        for alpha in self.ridge_alphas:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                model = Ridge(alpha=float(alpha), random_state=self.seed)
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
            mae = float(np.mean(np.abs(pred - y_val)))
            if mae < best_mae:
                best_mae = mae
                best = model
        if best is None:
            raise RuntimeError("Ridge selection failed.")
        return best

    def fit(
        self,
        x_train: np.ndarray,
        y_true_train: np.ndarray,
        y_classical_pred_train: np.ndarray,
        x_val: np.ndarray,
        y_true_val: np.ndarray,
        y_classical_pred_val: np.ndarray,
    ) -> "QuantumAnnotatorResidual":
        if y_true_train.shape != y_classical_pred_train.shape:
            raise ValueError("Train true/pred shapes must match.")
        if y_true_val.shape != y_classical_pred_val.shape:
            raise ValueError("Val true/pred shapes must match.")
        if x_train.ndim != 3 or x_val.ndim != 3:
            raise ValueError("x_train/x_val must be [N, L, D].")

        steps_per_sample = 1
        sample_budget = max(1, int(self.train_qeval_budget // max(1, steps_per_sample)))
        n_train_use = min(x_train.shape[0], sample_budget)
        n_val_use = min(x_val.shape[0], max(1, int(max(1, sample_budget) * 0.2)))

        x_train_use = x_train[:n_train_use]
        x_val_use = x_val[:n_val_use]
        y_train_use = y_true_train[:n_train_use]
        y_val_use = y_true_val[:n_val_use]
        y_train_base = y_classical_pred_train[:n_train_use]
        y_val_base = y_classical_pred_val[:n_val_use]

        y_res_train = (y_train_use - y_train_base).reshape(n_train_use, -1)
        y_res_val = (y_val_use - y_val_base).reshape(n_val_use, -1)
        self._residual_shape = (int(y_true_train.shape[1]), int(y_true_train.shape[2]))

        u_train = self._build_u(x_train_use)
        self._concrete.fit(u_train)
        phi_train = self._extract_q_features(x_train_use, phase="train")
        phi_val = self._extract_q_features(x_val_use, phase="train")
        self.last_q_features_train = phi_train.copy()
        self.last_q_features_val = phi_val.copy()

        self._assert_non_degenerate_q_features(phi_train, stage="train")
        self._assert_non_degenerate_q_features(phi_val, stage="val")

        self._x_scaler = StandardScaler().fit(phi_train)
        self._y_scaler = StandardScaler().fit(y_res_train)
        phi_train_s = self._x_scaler.transform(phi_train)
        phi_val_s = self._x_scaler.transform(phi_val)
        y_train_s = self._y_scaler.transform(y_res_train)
        y_val_s = self._y_scaler.transform(y_res_val)

        if self.quantum_feature_mode == "q_kernel_gram":
            self.last_kernel_gram = self._concrete.compute_kernel_gram(u_train, phase="train", batch_size=self.batch_size)

        self._ridge = self._fit_ridge_with_val(phi_train_s, y_train_s, phi_val_s, y_val_s)

        stats = self._concrete.last_feature_summary
        print(
            "[quantum] concrete features "
            f"mode={self.quantum_feature_mode} q_dim={self.q_dim} shots={self.shots} "
            f"mean={stats.get('q_mean', 0.0):.6f} std={stats.get('q_std', 0.0):.6f}"
        )
        return self

    def predict_residual(self, x_seq: np.ndarray) -> np.ndarray:
        if self._ridge is None or self._x_scaler is None or self._y_scaler is None or self._residual_shape is None:
            raise RuntimeError("QuantumAnnotatorResidual is not fitted.")
        n = x_seq.shape[0]
        h, d = self._residual_shape

        phi = self._extract_q_features(x_seq, phase="infer")
        self.last_q_features_infer = phi.copy()
        self._assert_non_degenerate_q_features(phi, stage="infer")
        phi_s = self._x_scaler.transform(phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            pred_s = self._ridge.predict(phi_s)
        pred = self._y_scaler.inverse_transform(pred_s).reshape(n, h, d)
        return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    def budget_dict(self) -> dict[str, int]:
        return self.budget.as_dict()

    def diagnostics(self) -> dict[str, np.ndarray | dict[str, float] | None]:
        return {
            "q_feature_summary": dict(self._concrete.last_feature_summary),
            "q_features_train": self.last_q_features_train,
            "q_features_val": self.last_q_features_val,
            "q_features_infer": self.last_q_features_infer,
            "kernel_gram": self.last_kernel_gram,
        }
