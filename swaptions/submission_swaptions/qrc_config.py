from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class QRCConfig:
    # modeling modes
    qrc_mode: str = "residual"
    qrc_target: str = "delta"
    qrc_baseline: str = "persistence"
    target_transform: str = "log"
    y_floor_mode: str = "train_p001"
    y_floor_value: float | None = None
    qrc_feat_norm: str = "standard"
    qrc_gate_tau: float = 0.06184302083987248
    qrc_resid_clip: float | None = None
    qrc_residvar_penalty: float = 0.1
    qrc_sweep_objective: str = "surface_mape"
    # photonic knobs
    pqrc_modes: int = 8
    pqrc_gain: float = 0.7565800181198181
    pqrc_input_scale: float = 0.6689556127535666
    pqrc_ridge: float = 0.0051830988569115735
    pqrc_shots: int = 32
    pqrc_budget: int = 32
    pqrc_nph: int = 2
    pqrc_in_pca: int = 4
    pqrc_feature: str = "clickprob"
    pqrc_pseudocount: float = 0.25
    pqrc_higher_order: int = 0
    pqrc_budget_alloc: str = "uniform"
    # run metadata
    seed: int = 0
    level: int = 1
    lookback: int = 20
    horizons: int = 6

    @staticmethod
    def from_args(args: Any) -> "QRCConfig":
        modes = int(getattr(args, "pqrc_modes", 0) or 0)
        if modes <= 0:
            modes = int(getattr(args, "pqrc_M", 8))
        tau = getattr(args, "qrc_gate_tau", None)
        if tau is None:
            tau = getattr(args, "tau", 0.06184302083987248)
        tau_f = float(tau if tau is not None else 0.06184302083987248)
        qrc_target = str(getattr(args, "qrc_target", "delta")).strip().lower()
        if qrc_target == "norm_resid" and abs(tau_f - 0.05) < 1e-12:
            tau_f = 0.5
        if qrc_target == "log_return" and abs(tau_f - 0.05) < 1e-12:
            tau_f = 0.02
        qrc_baseline = str(getattr(args, "qrc_baseline", "persistence")).strip().lower()
        if qrc_baseline == "persistence_naive":
            qrc_baseline = "persistence"

        return QRCConfig(
            qrc_mode=str(getattr(args, "qrc_mode", "residual")),
            qrc_target=qrc_target,
            qrc_baseline=qrc_baseline,
            target_transform=str(getattr(args, "target_transform", "log")),
            y_floor_mode=str(getattr(args, "y_floor_mode", "train_p001")),
            y_floor_value=(
                None
                if getattr(args, "y_floor_value", None) is None
                else float(getattr(args, "y_floor_value"))
            ),
            qrc_feat_norm=str(getattr(args, "qrc_feat_norm", "standard")),
            qrc_gate_tau=tau_f,
            qrc_resid_clip=(
                None
                if getattr(args, "qrc_resid_clip", None) is None
                else float(getattr(args, "qrc_resid_clip"))
            ),
            qrc_residvar_penalty=float(
                getattr(
                    args,
                    "qrc_residvar_penalty",
                    getattr(args, "qrc_residual_penalty", 0.1),
                )
            ),
            qrc_sweep_objective=str(getattr(args, "qrc_sweep_objective", "surface_mape")),
            pqrc_modes=int(modes),
            pqrc_gain=float(getattr(args, "pqrc_gain", 0.7565800181198181)),
            pqrc_input_scale=float(getattr(args, "pqrc_input_scale", 0.6689556127535666)),
            pqrc_ridge=float(getattr(args, "pqrc_ridge", 0.0051830988569115735)),
            pqrc_shots=int(getattr(args, "pqrc_shots", 32)),
            pqrc_budget=int(getattr(args, "pqrc_budget", 32)),
            pqrc_nph=int(getattr(args, "pqrc_Nph", 2)),
            pqrc_in_pca=int(getattr(args, "pqrc_in_pca", 4)),
            pqrc_feature=str(getattr(args, "pqrc_feature", "clickprob")),
            pqrc_pseudocount=float(getattr(args, "pqrc_pseudocount", 0.25)),
            pqrc_higher_order=int(getattr(args, "pqrc_higher_order", 0)),
            pqrc_budget_alloc=str(getattr(args, "pqrc_budget_alloc", "uniform")),
            seed=int(getattr(args, "seed", 0)),
            level=int(getattr(args, "level", 1)),
            lookback=int(getattr(args, "lookback", 20) or 20),
            horizons=int(getattr(args, "forecast_horizons", 6) or 6),
        )

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "QRCConfig":
        p = dict(payload)
        qrc_baseline = str(p.get("qrc_baseline", "persistence")).strip().lower()
        if qrc_baseline == "persistence_naive":
            qrc_baseline = "persistence"
        return QRCConfig(
            qrc_mode=str(p.get("qrc_mode", "residual")),
            qrc_target=str(p.get("qrc_target", "delta")),
            qrc_baseline=qrc_baseline,
            target_transform=str(p.get("target_transform", "log")),
            y_floor_mode=str(p.get("y_floor_mode", "train_p001")),
            y_floor_value=(None if p.get("y_floor_value", None) is None else float(p.get("y_floor_value"))),
            qrc_feat_norm=str(p.get("qrc_feat_norm", "standard")),
            qrc_gate_tau=float(p.get("qrc_gate_tau", p.get("tau", 0.06184302083987248))),
            qrc_resid_clip=(None if p.get("qrc_resid_clip", None) is None else float(p.get("qrc_resid_clip"))),
            qrc_residvar_penalty=float(p.get("qrc_residvar_penalty", p.get("qrc_residual_penalty", 0.1))),
            qrc_sweep_objective=str(p.get("qrc_sweep_objective", "surface_mape")),
            pqrc_modes=int(p.get("pqrc_modes", p.get("pqrc_M", 8))),
            pqrc_gain=float(p.get("pqrc_gain", 0.7565800181198181)),
            pqrc_input_scale=float(p.get("pqrc_input_scale", 0.6689556127535666)),
            pqrc_ridge=float(p.get("pqrc_ridge", 0.0051830988569115735)),
            pqrc_shots=int(p.get("pqrc_shots", 32)),
            pqrc_budget=int(p.get("pqrc_budget", 32)),
            pqrc_nph=int(p.get("pqrc_nph", p.get("pqrc_Nph", 2))),
            pqrc_in_pca=int(p.get("pqrc_in_pca", 4)),
            pqrc_feature=str(p.get("pqrc_feature", "clickprob")),
            pqrc_pseudocount=float(p.get("pqrc_pseudocount", 0.25)),
            pqrc_higher_order=int(p.get("pqrc_higher_order", 0)),
            pqrc_budget_alloc=str(p.get("pqrc_budget_alloc", "uniform")),
            seed=int(p.get("seed", 0)),
            level=int(p.get("level", 1)),
            lookback=int(p.get("lookback", 20)),
            horizons=int(p.get("horizons", p.get("forecast_horizons", 6))),
        )

    def validate(self) -> "QRCConfig":
        if str(self.qrc_baseline).strip().lower() not in {"persistence", "seasonal13"}:
            raise ValueError(f"qrc_baseline must be one of: persistence, seasonal13 (got {self.qrc_baseline}).")
        if str(self.target_transform).strip().lower() not in {"none", "log"}:
            raise ValueError(f"target_transform must be one of: none, log (got {self.target_transform}).")
        if str(self.y_floor_mode).strip().lower() not in {"train_p01", "train_p001", "fixed"}:
            raise ValueError(f"y_floor_mode must be one of: train_p01, train_p001, fixed (got {self.y_floor_mode}).")
        if str(self.qrc_sweep_objective).strip().lower() not in {
            "surface_mape",
            "surface_wape",
            "surface_mae",
            "surface_rmse",
            "mape",
            "mc_then_mape",
        }:
            raise ValueError(
                "qrc_sweep_objective must be one of: surface_mape, surface_wape, "
                "surface_mae, surface_rmse, mape, mc_then_mape."
            )
        if str(self.y_floor_mode).strip().lower() == "fixed":
            if self.y_floor_value is None or not np.isfinite(float(self.y_floor_value)):
                raise ValueError("y_floor_value must be finite when y_floor_mode=fixed.")
        if not (1 <= int(self.pqrc_modes) <= 20):
            raise ValueError(f"pqrc_modes must be in [1,20], got {self.pqrc_modes}.")
        if int(self.pqrc_shots) <= 0:
            raise ValueError(f"pqrc_shots must be >0, got {self.pqrc_shots}.")
        numeric = {
            "pqrc_gain": float(self.pqrc_gain),
            "pqrc_input_scale": float(self.pqrc_input_scale),
            "pqrc_ridge": float(self.pqrc_ridge),
            "qrc_gate_tau": float(self.qrc_gate_tau),
            "qrc_residvar_penalty": float(self.qrc_residvar_penalty),
            "pqrc_pseudocount": float(self.pqrc_pseudocount),
        }
        if self.qrc_resid_clip is not None:
            numeric["qrc_resid_clip"] = float(self.qrc_resid_clip)
        for k, v in numeric.items():
            if not np.isfinite(v):
                raise ValueError(f"{k} must be finite, got {v}.")
        if float(self.pqrc_pseudocount) < 0.0:
            raise ValueError(f"pqrc_pseudocount must be >= 0, got {self.pqrc_pseudocount}.")
        if str(self.pqrc_feature).strip().lower() not in {"coincidence", "clickprob", "pseudo_coincidence"}:
            raise ValueError(
                f"pqrc_feature must be one of: coincidence, clickprob, pseudo_coincidence (got {self.pqrc_feature})."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_to_hash(config: QRCConfig | dict[str, Any]) -> str:
    payload = asdict(config) if isinstance(config, QRCConfig) else dict(config)
    payload.pop("config_hash", None)
    blob = _canonical_json(payload).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def save_config(config: QRCConfig | dict[str, Any], path: str | Path) -> Path:
    cfg_dict = config.to_dict() if isinstance(config, QRCConfig) else dict(config)
    cfg_dict["config_hash"] = config_to_hash(cfg_dict)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_canonical_json(cfg_dict) + "\n", encoding="utf-8")
    return p


def load_config(path: str | Path) -> QRCConfig:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    cfg = QRCConfig.from_dict(payload).validate()
    return cfg
