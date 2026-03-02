from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .ported_budget import BudgetCounter

# PORTED IMPLEMENTATIONS SOURCE:
# https://github.com/Quandela/HybridAIQuantum-Challenge
# Source path: src/qml/models.py (mirrored locally in aaquandela/qml/models.py)
#
# Each block below marked "COPIED-AND-ADAPTED" is directly ported from that codebase,
# then minimally adapted to work with the Swaptions pipeline tensors.

try:
    from .ported_quantum_layers import QuantumFeatureMap, QuantumReservoir

    _HAS_PORTED_QUANTUM = True
    _PORTED_QUANTUM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    QuantumFeatureMap = None  # type: ignore[assignment]
    QuantumReservoir = None  # type: ignore[assignment]
    _HAS_PORTED_QUANTUM = False
    _PORTED_QUANTUM_IMPORT_ERROR = str(exc)


BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0)

PORTED_ALGO_SOURCES = {
    "q_bottleneck": "src/qml/models.py::run_q_bottleneck",
    "qrc": "src/qml/models.py::run_qrc",
    "ctrl_rff": "src/qml/models.py::run_ctrl_rff",
    "ctrl_learned_featuremap": "src/qml/models.py::run_ctrl_learned_featuremap",
    "ctrl_classical_reservoir": "src/qml/models.py::run_ctrl_classical_reservoir",
}


@dataclass
class ModelForecast:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::ModelForecast
    model: str
    z_pred_test: np.ndarray
    best_blend_w: float
    budget: BudgetCounter
    algo_id: str = ""
    q_features: np.ndarray | None = None
    q_feature_summary: dict[str, float] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _sanitize_array(arr: np.ndarray, clip: float = 50.0) -> np.ndarray:
    clean = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(clean, -clip, clip)


def _require_quantum_backend() -> None:
    if not _HAS_PORTED_QUANTUM:
        raise RuntimeError(
            "Ported quantum models require merlinquantum/perceval-quandela. "
            f"Import error: {_PORTED_QUANTUM_IMPORT_ERROR}"
        )


def _sanitize_quantum_shots(shots: int, *, model_name: str) -> int:
    s = int(shots)
    if s >= 1:
        return s
    warnings.warn(
        f"{model_name}: received non-positive shots={s}; forcing shots=1 so quantum evaluations are executed.",
        RuntimeWarning,
    )
    return 1


def _select_subset_counts(n_train: int, n_val: int, sample_budget: int) -> tuple[int, int]:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_select_subset_counts
    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if sample_budget <= 0:
        return 1, min(1, n_val)

    if n_val <= 0:
        return min(n_train, sample_budget), 0

    total = min(sample_budget, n_train + n_val)
    n_val_use = min(n_val, max(1, int(round(0.2 * total))))
    n_train_use = min(n_train, max(1, total - n_val_use))

    if n_train_use + n_val_use > total:
        n_val_use = max(1, total - n_train_use)

    return n_train_use, n_val_use


def _count_trainable_params(*modules: nn.Module | None) -> int:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_count_trainable_params
    total = 0
    for module in modules:
        if module is None:
            continue
        total += int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(total)


def _split_residual_targets(x_seq: np.ndarray, y_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_split_residual_targets
    last = x_seq[:, -1, :]
    residual = y_seq - last[:, None, :]
    return last, residual.reshape(y_seq.shape[0], -1)


def _fit_ridge_with_val(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    fixed_alpha: float | None = None,
) -> Ridge:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_fit_ridge_with_val
    x_train = _sanitize_array(x_train, clip=50.0)
    y_train = _sanitize_array(y_train, clip=50.0)
    x_val = _sanitize_array(x_val, clip=50.0)
    y_val = _sanitize_array(y_val, clip=50.0)

    if fixed_alpha is not None:
        model = Ridge(alpha=float(fixed_alpha), random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, y_train)
        return model

    best_model: Ridge | None = None
    best_mae = float("inf")

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_mae:
            best_mae = mae
            best_model = model

    if best_model is None:
        raise RuntimeError("Ridge fit failed.")
    return best_model


def _repeat_persistence(last: np.ndarray, horizon: int) -> np.ndarray:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_repeat_persistence
    return np.repeat(last[:, None, :], horizon, axis=1)


def _ridge_predict(model: Ridge, x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return model.predict(_sanitize_array(x, clip=50.0))


def _summary_stats(arr: np.ndarray) -> dict[str, float]:
    a = _sanitize_array(np.asarray(arr, dtype=float), clip=1e3)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "l2norm": float(np.linalg.norm(a)),
    }


def _assert_non_degenerate_quantum_features(q_features: np.ndarray, *, model_name: str, min_std: float = 1e-8) -> None:
    q = np.asarray(q_features, dtype=float)
    if q.size == 0 or (not np.isfinite(q).all()):
        raise RuntimeError(
            "Quantum model produced degenerate q_features; refusing to continue. "
            "Check encoding scale / shots / featuremap."
        )
    q_std = float(np.std(q))
    if q_std <= float(min_std):
        raise RuntimeError(
            "Quantum model produced degenerate q_features; refusing to continue. "
            f"Check encoding scale / shots / featuremap. model={model_name} std={q_std:.3e}"
        )


def _best_blend_weight(z_true: np.ndarray, z_persist: np.ndarray, z_model: np.ndarray) -> float:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_best_blend_weight
    best_w = 0.0
    best_mae = float("inf")
    for w in BLEND_GRID:
        z_blend = (1.0 - w) * z_persist + w * z_model
        mae = float(np.mean(np.abs(z_blend - z_true)))
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w


def _quantum_features(
    q_map: nn.Module,
    x_in: np.ndarray,
    phase: str,
    batch_size: int,
) -> np.ndarray:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_quantum_features
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_in.shape[0], batch_size):
            xb = torch.from_numpy(x_in[i : i + batch_size].astype(np.float32))
            feats.append(q_map(xb, phase=phase).cpu().numpy())
    return _sanitize_array(np.vstack(feats), clip=20.0)


def _quantum_states(
    q_res: nn.Module,
    x_seq: np.ndarray,
    phase: str,
    batch_size: int,
) -> np.ndarray:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_quantum_states
    states: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_seq.shape[0], batch_size):
            xb = torch.from_numpy(x_seq[i : i + batch_size].astype(np.float32))
            state = q_res.forward_sequence(xb, phase=phase)
            states.append(state.cpu().numpy())
    return _sanitize_array(np.vstack(states), clip=20.0)


class FixedSummaryProjector:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::FixedSummaryProjector
    def __init__(self, input_dim: int, output_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 1.0 / np.sqrt(max(1, input_dim)), size=(input_dim, output_dim))
        self.b = rng.normal(0.0, 0.2, size=(output_dim,))

    def transform(self, x_flat: np.ndarray) -> np.ndarray:
        x = _sanitize_array(x_flat, clip=20.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lin = _sanitize_array(x @ self.w + self.b, clip=50.0)
        return np.tanh(lin)


class RandomFourierFeatures:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::RandomFourierFeatures
    def __init__(self, input_dim: int, output_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 1.0, size=(input_dim, output_dim))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=(output_dim,))
        self.scale = np.sqrt(2.0 / max(1, output_dim))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_s = _sanitize_array(x, clip=20.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            arg = _sanitize_array(x_s @ self.w + self.b, clip=200.0)
        return self.scale * np.cos(arg)


class TinyLearnedFeatureMap(nn.Module):
    # COPIED-AND-ADAPTED FROM src/qml/models.py::TinyLearnedFeatureMap
    def __init__(self, input_dim: int, d_q: int, d_phi: int, output_dim: int) -> None:
        super().__init__()
        d_hidden = max(8, int(d_q))
        self.map1 = nn.Linear(input_dim, d_hidden)
        self.map2 = nn.Linear(d_hidden, d_phi)
        self.head = nn.Linear(d_phi, output_dim)
        nn.init.xavier_uniform_(self.map1.weight)
        nn.init.zeros_(self.map1.bias)
        nn.init.xavier_uniform_(self.map2.weight)
        nn.init.zeros_(self.map2.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.map1(x))
        x = torch.tanh(self.map2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_map(x))


class ClassicalRandomReservoir:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::ClassicalRandomReservoir
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        alpha: float,
        rho: float,
        stride: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 1.0 / np.sqrt(max(1, input_dim)), size=(input_dim, feature_dim))
        self.b = rng.normal(0.0, 0.2, size=(feature_dim,))
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.stride = max(1, int(stride))

    def transform(self, x_seq: np.ndarray, budget: BudgetCounter | None = None, phase: str = "train") -> np.ndarray:
        x_seq = _sanitize_array(x_seq, clip=20.0)
        b, l, _ = x_seq.shape
        state = np.zeros((b, self.w.shape[1]), dtype=float)
        for t in range(0, l, self.stride):
            if budget is not None:
                budget.add_feat_evals(b, phase=phase)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lin = _sanitize_array(self.rho * (x_seq[:, t, :] @ self.w + self.b), clip=50.0)
            phi = np.tanh(lin)
            state = self.alpha * state + (1.0 - self.alpha) * phi
        return state


def _build_qrc_sequence(x_seq: np.ndarray, pool_mode: str) -> tuple[np.ndarray, int]:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_build_qrc_sequence
    if pool_mode == "none":
        return x_seq, int(x_seq.shape[1])
    if pool_mode == "summary":
        last = x_seq[:, -1, :]
        mean = x_seq.mean(axis=1)
        delta = x_seq[:, -1, :] - x_seq[:, 0, :]
        std = x_seq.std(axis=1)
        seq = np.stack([last, mean, delta, std], axis=1)
        return seq, 4
    raise ValueError(f"Unknown qrc pool mode: {pool_mode}")


def _prepare_summary_inputs(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    d_q: int,
    use_mask_features: bool,
    mask_train: np.ndarray | None,
    mask_val: np.ndarray | None,
    mask_test: np.ndarray | None,
    d_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # COPIED-AND-ADAPTED FROM src/qml/models.py::_prepare_summary_inputs
    x_scaler = StandardScaler()
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    x_train_flat_s = _sanitize_array(x_scaler.fit_transform(x_train_flat), clip=20.0)
    x_val_flat_s = _sanitize_array(x_scaler.transform(x_val_flat), clip=20.0)
    x_test_flat_s = _sanitize_array(x_scaler.transform(x_test_flat), clip=20.0)

    projector_main = FixedSummaryProjector(input_dim=x_train_flat_s.shape[1], output_dim=d_q, seed=seed)
    u_train = projector_main.transform(x_train_flat_s)
    u_val = projector_main.transform(x_val_flat_s)
    u_test = projector_main.transform(x_test_flat_s)

    if use_mask_features:
        if mask_train is None or mask_val is None or mask_test is None:
            raise ValueError("Mask features requested but mask vectors were not provided.")
        mask_scaler = StandardScaler()
        mask_train_s = _sanitize_array(mask_scaler.fit_transform(mask_train), clip=20.0)
        mask_val_s = _sanitize_array(mask_scaler.transform(mask_val), clip=20.0)
        mask_test_s = _sanitize_array(mask_scaler.transform(mask_test), clip=20.0)
        mask_proj = FixedSummaryProjector(input_dim=mask_train_s.shape[1], output_dim=d_mask, seed=seed + 17)
        m_train = mask_proj.transform(mask_train_s)
        m_val = mask_proj.transform(mask_val_s)
        m_test = mask_proj.transform(mask_test_s)
        u_train = np.concatenate([u_train, m_train], axis=1)
        u_val = np.concatenate([u_val, m_val], axis=1)
        u_test = np.concatenate([u_test, m_test], axis=1)

    # MerLin simple layer supports <=20 features, so compress if needed.
    if u_train.shape[1] > 20:
        comp = FixedSummaryProjector(input_dim=u_train.shape[1], output_dim=20, seed=seed + 29)
        u_train = comp.transform(u_train)
        u_val = comp.transform(u_val)
        u_test = comp.transform(u_test)

    return _sanitize_array(u_train, clip=20.0), _sanitize_array(u_val, clip=20.0), _sanitize_array(u_test, clip=20.0)


# COPIED-AND-ADAPTED FROM src/qml/models.py::run_q_bottleneck
def run_q_bottleneck(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    horizon: int,
    shots: int,
    train_qeval_budget: int,
    batch_size: int,
    seed: int,
    d_q: int,
    d_phi: int,
    train_quantum: bool,
    grad_qeval_multiplier: float,
    use_mask_features: bool,
    mask_train_vec: np.ndarray | None,
    mask_val_vec: np.ndarray | None,
    mask_test_vec: np.ndarray | None,
    qb_lr: float = 3e-3,
    qb_max_steps: int = 200,
    qb_patience: int = 20,
    d_mask: int = 8,
) -> ModelForecast:
    _require_quantum_backend()
    shots = _sanitize_quantum_shots(shots, model_name="ported_q_bottleneck")
    np.random.seed(seed)
    torch.manual_seed(seed)

    effective_mult = max(1.0, float(grad_qeval_multiplier))
    raw_sample_budget = max(1, int(train_qeval_budget // effective_mult))
    n_train_use, n_val_use = _select_subset_counts(x_train.shape[0], x_val.shape[0], raw_sample_budget)

    x_train_use = x_train[:n_train_use]
    y_train_use = y_train[:n_train_use]
    x_val_use = x_val[:n_val_use] if n_val_use > 0 else x_val
    y_val_use = y_val[:n_val_use] if n_val_use > 0 else y_val

    mask_train_use = mask_train_vec[:n_train_use] if mask_train_vec is not None else None
    mask_val_use = mask_val_vec[:n_val_use] if (mask_val_vec is not None and n_val_use > 0) else mask_val_vec

    u_train, u_val, u_test = _prepare_summary_inputs(
        x_train=x_train_use,
        x_val=x_val_use,
        x_test=x_test,
        seed=seed,
        d_q=d_q,
        use_mask_features=use_mask_features,
        mask_train=mask_train_use,
        mask_val=mask_val_use,
        mask_test=mask_test_vec,
        d_mask=d_mask,
    )

    budget = BudgetCounter(shots=shots)
    q_map = QuantumFeatureMap(  # type: ignore[operator]
        input_dim=int(u_train.shape[1]),
        feature_dim=d_phi,
        shots=shots,
        budget_counter=budget,
        train_quantum=train_quantum,
    )
    last_train, y_train_res = _split_residual_targets(x_train_use, y_train_use)
    last_val, y_val_res = _split_residual_targets(x_val_use, y_val_use)
    last_test = x_test[:, -1, :]

    n_trainable_params = 0
    if train_quantum:
        q_map.train()
        y_scaler = StandardScaler()
        y_train_res_s = y_scaler.fit_transform(y_train_res).astype(np.float32)
        y_val_res_s = y_scaler.transform(y_val_res).astype(np.float32)
        y_dim = int(y_train_res_s.shape[1])

        head = nn.Sequential(
            nn.Linear(d_phi, d_phi),
            nn.Tanh(),
            nn.Linear(d_phi, y_dim),
        )
        nn.init.xavier_uniform_(head[0].weight)
        nn.init.zeros_(head[0].bias)
        nn.init.xavier_uniform_(head[2].weight)
        nn.init.zeros_(head[2].bias)

        params = [p for p in q_map.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad]
        if not params:
            params = [p for p in head.parameters() if p.requires_grad]
        n_trainable_params = _count_trainable_params(q_map, head)

        optimizer = torch.optim.Adam(params, lr=float(qb_lr))
        loss_fn = nn.MSELoss()

        x_train_t = torch.from_numpy(u_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train_res_s)
        x_val_t = torch.from_numpy(u_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val_res_s)

        rng = np.random.default_rng(seed)
        best_val = float("inf")
        best_q_state = {k: v.detach().clone() for k, v in q_map.state_dict().items()}
        best_h_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        no_improve = 0
        step = 0
        stop = False
        eval_every = max(1, min(20, int(np.ceil(max(1, x_train_t.shape[0]) / max(1, batch_size)))))
        n_val_eval = max(1, min(x_val_t.shape[0], 2 * batch_size))

        while (step < int(max(1, qb_max_steps))) and (not stop):
            perm = rng.permutation(x_train_t.shape[0])
            for s in range(0, x_train_t.shape[0], batch_size):
                if budget.train_qevals * effective_mult >= train_qeval_budget:
                    stop = True
                    break

                idx = perm[s : s + batch_size]
                xb = x_train_t[idx]
                yb = y_train_t[idx]

                optimizer.zero_grad(set_to_none=True)
                phi = q_map(xb, phase="train")
                pred = head(phi)
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    stop = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                step += 1

                if (step % eval_every == 0) or (step >= int(max(1, qb_max_steps))):
                    with torch.no_grad():
                        preds_val = []
                        for vs in range(0, n_val_eval, batch_size):
                            xvb = x_val_t[vs : vs + batch_size]
                            preds_val.append(head(q_map(xvb, phase="train")))
                        pred_val_t = torch.cat(preds_val, dim=0)
                        val_loss = torch.mean(torch.abs(pred_val_t - y_val_t[:n_val_eval])).item()
                    if val_loss + 1e-8 < best_val:
                        best_val = val_loss
                        best_q_state = {k: v.detach().clone() for k, v in q_map.state_dict().items()}
                        best_h_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= int(max(1, qb_patience)):
                        stop = True
                        break
                    if budget.train_qevals * effective_mult >= train_qeval_budget:
                        stop = True
                        break

        q_map.load_state_dict(best_q_state)
        head.load_state_dict(best_h_state)
        q_map.eval()
        head.eval()

        with torch.no_grad():
            pred_val_chunks = []
            for i in range(0, u_val.shape[0], batch_size):
                xb = torch.from_numpy(u_val[i : i + batch_size].astype(np.float32))
                pred_val_chunks.append(head(q_map(xb, phase="train")).cpu().numpy())
            pred_val_res = y_scaler.inverse_transform(np.vstack(pred_val_chunks)).reshape(last_val.shape[0], horizon, -1)

            pred_test_chunks = []
            for i in range(0, u_test.shape[0], batch_size):
                xb = torch.from_numpy(u_test[i : i + batch_size].astype(np.float32))
                pred_test_chunks.append(head(q_map(xb, phase="infer")).cpu().numpy())
            pred_test_res = y_scaler.inverse_transform(np.vstack(pred_test_chunks)).reshape(
                last_test.shape[0], horizon, -1
            )
        phi_train = _quantum_features(q_map, u_train, phase="train", batch_size=batch_size)
        phi_val = _quantum_features(q_map, u_val, phase="train", batch_size=batch_size)
        phi_test = _quantum_features(q_map, u_test, phase="infer", batch_size=batch_size)
    else:
        q_map.eval()
        phi_train = _quantum_features(q_map, u_train, phase="train", batch_size=batch_size)
        phi_val = _quantum_features(q_map, u_val, phase="train", batch_size=batch_size)
        phi_test = _quantum_features(q_map, u_test, phase="infer", batch_size=batch_size)

        y_scaler = StandardScaler()
        y_train_res_s = y_scaler.fit_transform(y_train_res)
        y_val_res_s = y_scaler.transform(y_val_res)

        phi_scaler = StandardScaler()
        phi_train_s = phi_scaler.fit_transform(phi_train)
        phi_val_s = phi_scaler.transform(phi_val)
        phi_test_s = phi_scaler.transform(phi_test)

        model = _fit_ridge_with_val(phi_train_s, y_train_res_s, phi_val_s, y_val_res_s, seed=seed)

        pred_val_res = y_scaler.inverse_transform(_ridge_predict(model, phi_val_s)).reshape(last_val.shape[0], horizon, -1)
        pred_test_res = y_scaler.inverse_transform(_ridge_predict(model, phi_test_s)).reshape(last_test.shape[0], horizon, -1)

    _assert_non_degenerate_quantum_features(phi_train, model_name="ported_q_bottleneck")
    _assert_non_degenerate_quantum_features(phi_val, model_name="ported_q_bottleneck")
    _assert_non_degenerate_quantum_features(phi_test, model_name="ported_q_bottleneck")

    z_val_model = last_val[:, None, :] + pred_val_res
    val_mae = float(np.mean(np.abs(z_val_model - y_val_use)))
    z_test_model = last_test[:, None, :] + pred_test_res

    return ModelForecast(
        model="q_bottleneck",
        z_pred_test=z_test_model,
        best_blend_w=1.0,
        budget=budget,
        algo_id="q_bottleneck",
        q_features=phi_test,
        q_feature_summary=_summary_stats(phi_test),
        config={
            "q_dim": int(d_phi),
            "shots": int(shots),
            "featuremap_seed": int(seed),
            "input_dim": int(u_train.shape[1]),
        },
        extra={
            "qrc_steps_per_sample": 0,
            "n_trainable_params": int(n_trainable_params),
            "val_mae": float(val_mae),
            "ported_source": PORTED_ALGO_SOURCES["q_bottleneck"],
        },
    )


# COPIED-AND-ADAPTED FROM src/qml/models.py::run_ctrl_rff
def run_ctrl_rff(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    horizon: int,
    seed: int,
    d_q: int,
    d_phi: int,
    use_mask_features: bool,
    mask_train_vec: np.ndarray | None,
    mask_val_vec: np.ndarray | None,
    mask_test_vec: np.ndarray | None,
    d_mask: int = 8,
) -> ModelForecast:
    np.random.seed(seed)

    u_train, u_val, u_test = _prepare_summary_inputs(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        seed=seed,
        d_q=d_q,
        use_mask_features=use_mask_features,
        mask_train=mask_train_vec,
        mask_val=mask_val_vec,
        mask_test=mask_test_vec,
        d_mask=d_mask,
    )

    budget = BudgetCounter(shots=0)
    budget.add_feat_evals(u_train.shape[0] + u_val.shape[0], phase="train")
    budget.add_feat_evals(u_test.shape[0], phase="infer")

    rff = RandomFourierFeatures(input_dim=u_train.shape[1], output_dim=d_phi, seed=seed + 11)
    phi_train = rff.transform(u_train)
    phi_val = rff.transform(u_val)
    phi_test = rff.transform(u_test)

    _, y_train_res = _split_residual_targets(x_train, y_train)
    last_val, y_val_res = _split_residual_targets(x_val, y_val)
    last_test = x_test[:, -1, :]

    y_scaler = StandardScaler()
    y_train_res_s = y_scaler.fit_transform(y_train_res)
    y_val_res_s = y_scaler.transform(y_val_res)

    phi_scaler = StandardScaler()
    phi_train_s = phi_scaler.fit_transform(phi_train)
    phi_val_s = phi_scaler.transform(phi_val)
    phi_test_s = phi_scaler.transform(phi_test)

    model = _fit_ridge_with_val(phi_train_s, y_train_res_s, phi_val_s, y_val_res_s, seed=seed)

    pred_val_res = y_scaler.inverse_transform(_ridge_predict(model, phi_val_s)).reshape(last_val.shape[0], horizon, -1)
    pred_test_res = y_scaler.inverse_transform(_ridge_predict(model, phi_test_s)).reshape(last_test.shape[0], horizon, -1)

    z_test_model = last_test[:, None, :] + pred_test_res

    return ModelForecast(
        model="ctrl_rff",
        z_pred_test=z_test_model,
        best_blend_w=1.0,
        budget=budget,
        algo_id="ctrl_rff",
        config={
            "rff_dim": int(d_phi),
            "omega_seed": int(seed + 11),
            "gamma": 1.0,
        },
        extra={
            "qrc_steps_per_sample": 0,
            "n_trainable_params": 0,
            "ported_source": PORTED_ALGO_SOURCES["ctrl_rff"],
        },
    )


# COPIED-AND-ADAPTED FROM src/qml/models.py::run_ctrl_learned_featuremap
def run_ctrl_learned_featuremap(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    horizon: int,
    seed: int,
    d_q: int,
    d_phi: int,
    use_mask_features: bool,
    mask_train_vec: np.ndarray | None,
    mask_val_vec: np.ndarray | None,
    mask_test_vec: np.ndarray | None,
    lr: float = 3e-3,
    max_steps: int = 200,
    patience: int = 20,
    batch_size: int = 32,
    d_mask: int = 8,
) -> ModelForecast:
    np.random.seed(seed)
    torch.manual_seed(seed)

    u_train, u_val, u_test = _prepare_summary_inputs(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        seed=seed,
        d_q=d_q,
        use_mask_features=use_mask_features,
        mask_train=mask_train_vec,
        mask_val=mask_val_vec,
        mask_test=mask_test_vec,
        d_mask=d_mask,
    )

    _, y_train_res = _split_residual_targets(x_train, y_train)
    last_val, y_val_res = _split_residual_targets(x_val, y_val)
    last_test = x_test[:, -1, :]

    x_scaler = StandardScaler()
    u_train_s = x_scaler.fit_transform(u_train).astype(np.float32)
    u_val_s = x_scaler.transform(u_val).astype(np.float32)
    u_test_s = x_scaler.transform(u_test).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_res_s = y_scaler.fit_transform(y_train_res).astype(np.float32)
    y_val_res_s = y_scaler.transform(y_val_res).astype(np.float32)

    budget = BudgetCounter(shots=0)
    model = TinyLearnedFeatureMap(
        input_dim=u_train_s.shape[1],
        d_q=d_q,
        d_phi=d_phi,
        output_dim=y_train_res_s.shape[1],
    )
    n_trainable_params = _count_trainable_params(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    x_train_t = torch.from_numpy(u_train_s)
    y_train_t = torch.from_numpy(y_train_res_s)
    x_val_t = torch.from_numpy(u_val_s)
    y_val_t = torch.from_numpy(y_val_res_s)

    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    step = 0
    eval_every = max(1, min(20, int(np.ceil(max(1, x_train_t.shape[0]) / max(1, batch_size)))))
    n_val_eval = max(1, min(x_val_t.shape[0], 2 * batch_size))

    while step < int(max(1, max_steps)):
        perm = rng.permutation(x_train_t.shape[0])
        stop = False
        for s in range(0, x_train_t.shape[0], batch_size):
            idx = perm[s : s + batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]
            budget.add_feat_evals(xb.shape[0], phase="train")

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                stop = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1

            if (step % eval_every == 0) or (step >= int(max(1, max_steps))):
                with torch.no_grad():
                    preds_val = []
                    for vs in range(0, n_val_eval, batch_size):
                        xvb = x_val_t[vs : vs + batch_size]
                        budget.add_feat_evals(xvb.shape[0], phase="train")
                        preds_val.append(model(xvb))
                    pred_val_t = torch.cat(preds_val, dim=0)
                    val_loss = torch.mean(torch.abs(pred_val_t - y_val_t[:n_val_eval])).item()
                if val_loss + 1e-8 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= int(max(1, patience)):
                    stop = True
                    break
            if step >= int(max(1, max_steps)):
                stop = True
                break
        if stop:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_val_parts = []
        for i in range(0, u_val_s.shape[0], batch_size):
            xb = torch.from_numpy(u_val_s[i : i + batch_size])
            budget.add_feat_evals(xb.shape[0], phase="train")
            pred_val_parts.append(model(xb).cpu().numpy())
        pred_val_res = y_scaler.inverse_transform(np.vstack(pred_val_parts)).reshape(last_val.shape[0], horizon, -1)

        pred_test_parts = []
        for i in range(0, u_test_s.shape[0], batch_size):
            xb = torch.from_numpy(u_test_s[i : i + batch_size])
            budget.add_feat_evals(xb.shape[0], phase="infer")
            pred_test_parts.append(model(xb).cpu().numpy())
        pred_test_res = y_scaler.inverse_transform(np.vstack(pred_test_parts)).reshape(last_test.shape[0], horizon, -1)

    z_test_model = last_test[:, None, :] + pred_test_res

    return ModelForecast(
        model="ctrl_learned_featuremap",
        z_pred_test=z_test_model,
        best_blend_w=1.0,
        budget=budget,
        algo_id="ctrl_learned_featuremap",
        config={
            "num_layers": 3,
            "trainable_params_count": int(n_trainable_params),
        },
        extra={
            "qrc_steps_per_sample": 0,
            "n_trainable_params": int(n_trainable_params),
            "ported_source": PORTED_ALGO_SOURCES["ctrl_learned_featuremap"],
        },
    )


# COPIED-AND-ADAPTED FROM src/qml/models.py::run_qrc
def run_qrc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    horizon: int,
    shots: int,
    train_qeval_budget: int,
    batch_size: int,
    seed: int,
    d_phi: int,
    train_quantum: bool,
    grad_qeval_multiplier: float,
    qrc_stride: int,
    qrc_pool: str,
    reservoir_alpha: float,
    reservoir_rho: float,
    reservoir_ridge_lambda: float | None,
) -> ModelForecast:
    _require_quantum_backend()
    shots = _sanitize_quantum_shots(shots, model_name="ported_qrc")
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train_seq, steps_before_stride = _build_qrc_sequence(x_train, qrc_pool)
    x_val_seq, _ = _build_qrc_sequence(x_val, qrc_pool)
    x_test_seq, _ = _build_qrc_sequence(x_test, qrc_pool)

    stride = 1 if qrc_pool == "summary" else max(1, int(qrc_stride))
    steps_per_sample = int((steps_before_stride + stride - 1) // stride)

    effective_mult = max(1.0, float(grad_qeval_multiplier))
    effective_cost_per_sample = max(1.0, effective_mult * steps_per_sample)
    sample_budget = max(1, int(train_qeval_budget // effective_cost_per_sample))
    n_train_use, n_val_use = _select_subset_counts(x_train_seq.shape[0], x_val_seq.shape[0], sample_budget)

    x_train_use = x_train_seq[:n_train_use]
    y_train_use = y_train[:n_train_use]
    x_val_use = x_val_seq[:n_val_use] if n_val_use > 0 else x_val_seq
    y_val_use = y_val[:n_val_use] if n_val_use > 0 else y_val

    scaler = StandardScaler()
    x_train_use_s = scaler.fit_transform(x_train_use.reshape(-1, x_train_use.shape[-1])).reshape(x_train_use.shape)
    x_val_use_s = scaler.transform(x_val_use.reshape(-1, x_val_use.shape[-1])).reshape(x_val_use.shape)
    x_test_s = scaler.transform(x_test_seq.reshape(-1, x_test_seq.shape[-1])).reshape(x_test_seq.shape)

    budget = BudgetCounter(shots=shots)
    q_res = QuantumReservoir(  # type: ignore[operator]
        input_dim=x_train.shape[-1],
        feature_dim=d_phi,
        shots=shots,
        budget_counter=budget,
        alpha=float(reservoir_alpha),
        rho=float(reservoir_rho),
        stride=stride,
        train_quantum=train_quantum,
        seed=seed,
    )
    q_res.eval()
    n_trainable_params = _count_trainable_params(q_res)

    state_train = _quantum_states(q_res, x_train_use_s, phase="train", batch_size=batch_size)
    state_val = _quantum_states(q_res, x_val_use_s, phase="train", batch_size=batch_size)
    state_test = _quantum_states(q_res, x_test_s, phase="infer", batch_size=batch_size)
    _assert_non_degenerate_quantum_features(state_train, model_name="ported_qrc")
    _assert_non_degenerate_quantum_features(state_val, model_name="ported_qrc")
    _assert_non_degenerate_quantum_features(state_test, model_name="ported_qrc")

    last_train, y_train_res = _split_residual_targets(x_train_use, y_train_use)
    last_val, y_val_res = _split_residual_targets(x_val[:n_val_use] if n_val_use > 0 else x_val, y_val_use)
    last_test = x_test[:, -1, :]

    y_scaler = StandardScaler()
    y_train_res_s = y_scaler.fit_transform(y_train_res)
    y_val_res_s = y_scaler.transform(y_val_res)

    state_scaler = StandardScaler()
    state_train_s = state_scaler.fit_transform(state_train)
    state_val_s = state_scaler.transform(state_val)
    state_test_s = state_scaler.transform(state_test)

    model = _fit_ridge_with_val(
        state_train_s,
        y_train_res_s,
        state_val_s,
        y_val_res_s,
        seed=seed,
        fixed_alpha=reservoir_ridge_lambda,
    )

    pred_val_res = y_scaler.inverse_transform(_ridge_predict(model, state_val_s)).reshape(last_val.shape[0], horizon, -1)
    pred_test_res = y_scaler.inverse_transform(_ridge_predict(model, state_test_s)).reshape(last_test.shape[0], horizon, -1)

    z_test_model = last_test[:, None, :] + pred_test_res

    return ModelForecast(
        model="qrc",
        z_pred_test=z_test_model,
        best_blend_w=1.0,
        budget=budget,
        algo_id="qrc",
        q_features=state_test,
        q_feature_summary=_summary_stats(state_test),
        config={
            "reservoir_stride": int(stride),
            "q_dim": int(d_phi),
            "shots": int(shots),
            "featuremap_seed": int(seed),
        },
        extra={
            "qrc_steps_per_sample": int(steps_per_sample),
            "n_trainable_params": int(n_trainable_params),
            "ported_source": PORTED_ALGO_SOURCES["qrc"],
        },
    )


# COPIED-AND-ADAPTED FROM src/qml/models.py::run_ctrl_classical_reservoir
def run_ctrl_classical_reservoir(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    horizon: int,
    seed: int,
    d_phi: int,
    qrc_stride: int,
    qrc_pool: str,
    reservoir_alpha: float,
    reservoir_rho: float,
    reservoir_ridge_lambda: float | None,
) -> ModelForecast:
    x_train_seq, steps_before_stride = _build_qrc_sequence(x_train, qrc_pool)
    x_val_seq, _ = _build_qrc_sequence(x_val, qrc_pool)
    x_test_seq, _ = _build_qrc_sequence(x_test, qrc_pool)

    stride = 1 if qrc_pool == "summary" else max(1, int(qrc_stride))
    steps_per_sample = int((steps_before_stride + stride - 1) // stride)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_seq.reshape(-1, x_train_seq.shape[-1])).reshape(x_train_seq.shape)
    x_val_s = scaler.transform(x_val_seq.reshape(-1, x_val_seq.shape[-1])).reshape(x_val_seq.shape)
    x_test_s = scaler.transform(x_test_seq.reshape(-1, x_test_seq.shape[-1])).reshape(x_test_seq.shape)

    budget = BudgetCounter(shots=0)
    reservoir = ClassicalRandomReservoir(
        input_dim=x_train.shape[-1],
        feature_dim=d_phi,
        alpha=float(reservoir_alpha),
        rho=float(reservoir_rho),
        stride=stride,
        seed=seed,
    )

    state_train = reservoir.transform(x_train_s, budget=budget, phase="train")
    state_val = reservoir.transform(x_val_s, budget=budget, phase="train")
    state_test = reservoir.transform(x_test_s, budget=budget, phase="infer")

    _, y_train_res = _split_residual_targets(x_train, y_train)
    last_val, y_val_res = _split_residual_targets(x_val, y_val)
    last_test = x_test[:, -1, :]

    y_scaler = StandardScaler()
    y_train_res_s = y_scaler.fit_transform(y_train_res)
    y_val_res_s = y_scaler.transform(y_val_res)

    state_scaler = StandardScaler()
    state_train_s = state_scaler.fit_transform(state_train)
    state_val_s = state_scaler.transform(state_val)
    state_test_s = state_scaler.transform(state_test)

    model = _fit_ridge_with_val(
        state_train_s,
        y_train_res_s,
        state_val_s,
        y_val_res_s,
        seed=seed,
        fixed_alpha=reservoir_ridge_lambda,
    )

    pred_val_res = y_scaler.inverse_transform(_ridge_predict(model, state_val_s)).reshape(last_val.shape[0], horizon, -1)
    pred_test_res = y_scaler.inverse_transform(_ridge_predict(model, state_test_s)).reshape(last_test.shape[0], horizon, -1)

    z_test_model = last_test[:, None, :] + pred_test_res

    return ModelForecast(
        model="ctrl_classical_reservoir",
        z_pred_test=z_test_model,
        best_blend_w=1.0,
        budget=budget,
        algo_id="ctrl_classical_reservoir",
        config={
            "spectral_radius": float(reservoir_rho),
            "leak_rate": float(reservoir_alpha),
            "reservoir_seed": int(seed),
        },
        extra={
            "qrc_steps_per_sample": int(steps_per_sample),
            "n_trainable_params": 0,
            "ported_source": PORTED_ALGO_SOURCES["ctrl_classical_reservoir"],
        },
    )


PortedModelName = Literal[
    "ported_ctrl_rff",
    "ported_ctrl_learned_featuremap",
    "ported_ctrl_classical_reservoir",
    "ported_q_bottleneck",
    "ported_qrc",
]


PORTED_MODEL_BUILDERS: dict[str, tuple[str, Callable[..., ModelForecast]]] = {
    "ported_ctrl_rff": ("ctrl_rff", run_ctrl_rff),
    "ported_ctrl_learned_featuremap": ("ctrl_learned_featuremap", run_ctrl_learned_featuremap),
    "ported_ctrl_classical_reservoir": ("ctrl_classical_reservoir", run_ctrl_classical_reservoir),
    "ported_q_bottleneck": ("q_bottleneck", run_q_bottleneck),
    "ported_qrc": ("qrc", run_qrc),
}


def run_ported_model(model_name: PortedModelName | str, **kwargs: Any) -> ModelForecast:
    name = str(model_name).strip()
    if name not in PORTED_MODEL_BUILDERS:
        known = ", ".join(sorted(PORTED_MODEL_BUILDERS.keys()))
        raise ValueError(f"Unknown ported model '{name}'. Expected one of: {known}")
    expected_algo, builder = PORTED_MODEL_BUILDERS[name]
    forecast = builder(**kwargs)
    actual_algo = str(forecast.algo_id or forecast.model).strip()
    if actual_algo != expected_algo:
        raise RuntimeError(
            f"Ported dispatch mismatch for {name}: expected algo_id={expected_algo}, got {actual_algo}. "
            "Refusing to continue."
        )
    if ("qrc" in name or "q_bottleneck" in name) and actual_algo not in {"qrc", "q_bottleneck"}:
        raise RuntimeError(f"Quantum dispatch mismatch for {name}: got {actual_algo}")
    if name.startswith("ported_ctrl_") and actual_algo.startswith("q"):
        raise RuntimeError(f"Control dispatch mismatch for {name}: got quantum algo {actual_algo}")
    forecast.extra.setdefault("dispatch_model_name", name)
    forecast.extra.setdefault("dispatch_expected_algo", expected_algo)
    forecast.extra.setdefault("dispatch_actual_algo", actual_algo)
    return forecast
