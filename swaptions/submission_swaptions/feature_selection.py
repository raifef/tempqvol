from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - explicit runtime dependency
    raise RuntimeError("scikit-learn is required for submission_swaptions.feature_selection.") from exc


ScoreFn = Callable[[np.ndarray, np.ndarray], float]


def ridge_validation_mae_score(
    features: np.ndarray,
    target: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> float:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got {features.shape}.")
    features = np.clip(np.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
    if target.ndim == 1:
        target_2d = target[:, None]
    else:
        target_2d = target.reshape(target.shape[0], -1)
    target_2d = np.clip(np.nan_to_num(target_2d, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
    if features.shape[0] != target_2d.shape[0]:
        raise ValueError("features and target must share first dimension.")
    n = features.shape[0]
    if n < 5:
        return float("inf")

    split = max(1, min(n - 1, int(round((1.0 - float(val_fraction)) * n))))
    x_train, x_val = features[:split], features[split:]
    y_train, y_val = target_2d[:split], target_2d[split:]
    if x_val.shape[0] == 0:
        return float("inf")

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    x_val_s = x_scaler.transform(x_val)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    best = float("inf")
    for alpha in alpha_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = Ridge(alpha=float(alpha), random_state=seed)
            model.fit(x_train_s, y_train_s)
            pred_s = model.predict(x_val_s)
        mae = float(np.mean(np.abs(pred_s - y_val_s)))
        if mae < best:
            best = mae
    return float(best)


def wrapper_forward_selection(
    features: np.ndarray,
    target: np.ndarray,
    max_k: int,
    score_fn: ScoreFn,
) -> list[int]:
    """
    Forward selection wrapper.
    Selects up to max_k feature indices using score_fn (lower is better).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got {features.shape}.")
    if int(max_k) < 1:
        return []

    n_features = int(features.shape[1])
    k_lim = min(int(max_k), n_features)
    selected: list[int] = []
    remaining = list(range(n_features))
    best_score_so_far = float("inf")

    while remaining and len(selected) < k_lim:
        best_idx = None
        best_score = float("inf")
        for idx in remaining:
            trial = selected + [idx]
            score = float(score_fn(features[:, trial], target))
            if score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_score_so_far = min(best_score_so_far, best_score)
        print(
            f"[feature_selection] step={len(selected)} selected_idx={best_idx} "
            f"score={best_score:.6f}"
        )

    print(
        f"[feature_selection] final_selected={selected} "
        f"final_dim={len(selected)} max_k={k_lim} best_score={best_score_so_far:.6f}"
    )
    return selected
