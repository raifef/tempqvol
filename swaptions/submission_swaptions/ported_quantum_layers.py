from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .ported_budget import BudgetCounter

# COPIED-AND-ADAPTED FROM:
# https://github.com/Quandela/HybridAIQuantum-Challenge
# Source path: src/qml/quantum_layers.py (mirrored locally in aaquandela/qml/quantum_layers.py)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))


def _load_merlin_with_quantumlayer():
    try:
        ml = importlib.import_module("merlin")
    except Exception as exc:
        raise RuntimeError("MerLin import failed. Install merlinquantum/perceval-quandela and rerun.") from exc

    quantum_layer = getattr(ml, "QuantumLayer", None)
    if quantum_layer is None:
        try:
            from merlin import QuantumLayer as imported_quantum_layer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Imported 'merlin' but 'QuantumLayer' is missing. This usually means a different package named "
                "'merlin' is installed in this environment. Install merlinquantum/perceval-quandela and ensure "
                "that `import merlin; merlin.QuantumLayer` works."
            ) from exc
        setattr(ml, "QuantumLayer", imported_quantum_layer)
    return ml


ML = _load_merlin_with_quantumlayer()


def _freeze_module_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class QuantumFeatureMap(nn.Module):
    """MerLin-backed quantum feature map for vector inputs."""

    # COPIED-AND-ADAPTED FROM qml/quantum_layers.py::QuantumFeatureMap
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        shots: int,
        budget_counter: BudgetCounter,
        train_quantum: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if input_dim > 20:
            raise ValueError("MerLin QuantumLayer.simple supports input_dim <= 20.")

        self.shots = int(max(1, shots))
        self.budget_counter = budget_counter
        self.quantum = ML.QuantumLayer.simple(input_size=int(input_dim))
        if not train_quantum:
            _freeze_module_parameters(self.quantum)

        raw_dim = int(getattr(self.quantum, "output_size"))
        self.project = nn.Linear(raw_dim, int(feature_dim))
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor, phase: str = "train") -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"QuantumFeatureMap expects [B, D], got {tuple(x.shape)}")
        batch = int(x.shape[0])
        self.budget_counter.add_qevals(batch, phase=phase)
        q_out = self.quantum(x, shots=self.shots)
        return torch.tanh(self.project(q_out))


class QuantumReservoir(nn.Module):
    """Fixed quantum reservoir state update using MerLin features."""

    # COPIED-AND-ADAPTED FROM qml/quantum_layers.py::QuantumReservoir
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        shots: int,
        budget_counter: BudgetCounter,
        alpha: float = 0.70,
        rho: float = 0.9,
        stride: int = 1,
        train_quantum: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if input_dim > 20:
            raise ValueError("MerLin QuantumLayer.simple supports input_dim <= 20.")

        self.shots = int(max(1, shots))
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.stride = max(1, int(stride))
        self.budget_counter = budget_counter

        self.quantum = ML.QuantumLayer.simple(input_size=int(input_dim))
        if not train_quantum:
            _freeze_module_parameters(self.quantum)

        raw_dim = int(getattr(self.quantum, "output_size"))
        rng = np.random.default_rng(seed)
        w = rng.normal(loc=0.0, scale=1.0 / np.sqrt(max(1, raw_dim)), size=(raw_dim, feature_dim))
        b = rng.normal(loc=0.0, scale=0.2, size=(feature_dim,))

        self.register_buffer("proj_w", torch.tensor(w, dtype=torch.float32))
        self.register_buffer("proj_b", torch.tensor(b, dtype=torch.float32))

    def steps_for_length(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        return int((seq_len + self.stride - 1) // self.stride)

    def forward_step(self, x_t: torch.Tensor, phase: str) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"QuantumReservoir.forward_step expects [B, D], got {tuple(x_t.shape)}")
        batch = int(x_t.shape[0])
        self.budget_counter.add_qevals(batch, phase=phase)
        q_out = self.quantum(x_t, shots=self.shots)
        return torch.tanh(self.rho * (q_out @ self.proj_w + self.proj_b))

    def forward_sequence(self, x_seq: torch.Tensor, phase: str = "train") -> torch.Tensor:
        if x_seq.ndim != 3:
            raise ValueError(f"QuantumReservoir.forward_sequence expects [B, L, D], got {tuple(x_seq.shape)}")
        b, l, _ = x_seq.shape
        state = torch.zeros((b, self.proj_b.numel()), dtype=torch.float32, device=x_seq.device)
        for t in range(0, l, self.stride):
            phi_t = self.forward_step(x_seq[:, t, :], phase=phase)
            state = self.alpha * state + (1.0 - self.alpha) * phi_t
        return state
