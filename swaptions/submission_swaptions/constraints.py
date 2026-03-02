from __future__ import annotations

from typing import Any

SIM_INPUT_DIM_MAX = 20
QPU_INPUT_DIM_MAX = 24


def _warning_banner(message: str) -> None:
    border = "=" * 72
    print(border)
    print("WARNING")
    print(message)
    print(border)


def inspect_merlin_layer(layer: Any) -> dict[str, Any]:
    """Best-effort inspection of MerLin-like layer metadata."""
    attrs = ("input_size", "output_size", "n_modes", "n_photons", "circuit")
    info: dict[str, Any] = {}

    print("MerLin layer inspection")
    print("-" * 24)
    for attr in attrs:
        try:
            value = getattr(layer, attr)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"{attr}: <unavailable: {exc.__class__.__name__}>")
            continue
        info[attr] = value
        print(f"{attr}: {value!r}")

    if "n_photons" not in info or info.get("n_photons") is None:
        _warning_banner(
            "Photon count could not be verified from this layer object. "
            "Proceeding without hard photon-count validation."
        )

    return info


def validate_constraints(input_dim: int, backend: str) -> None:
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"sim", "qpu"}:
        raise ValueError(f"Unknown backend '{backend}'. Expected one of: sim, qpu.")

    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}.")

    limit = SIM_INPUT_DIM_MAX if backend_norm == "sim" else QPU_INPUT_DIM_MAX
    if input_dim > limit:
        raise ValueError(
            f"Constraint violation: input_dim={input_dim} exceeds backend='{backend_norm}' limit ({limit}). "
            f"Reduce encoded dimension or add a compression stage."
        )
