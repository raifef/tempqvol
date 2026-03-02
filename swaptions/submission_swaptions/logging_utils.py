from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def stable_hash_array(arr: np.ndarray, *, name: str) -> str:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=-9999.0, posinf=1e308, neginf=-1e308)
    h = hashlib.sha256()
    h.update(str(name).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()[:12]


def stable_hash_config(config: dict[str, Any], *, name: str = "config") -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    h = hashlib.sha256()
    h.update(str(name).encode("utf-8"))
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:12]


def summarize_matrix(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "l2norm": float(np.linalg.norm(a)),
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str))
        f.write("\n")
