from __future__ import annotations

import numpy as np
import pandas as pd

from submission_swaptions.models_classical import train_classical_forecaster
from submission_swaptions.plot_model_comparison import (
    PERSISTENCE_SURFACE_NAME,
    _compute_collapse_diagnostics,
    _horizon_metric,
    _select_row_by_hash,
    _write_horizon_metrics,
)
from submission_swaptions.positive_transform import PositiveLogTransform


def test_positive_log_transform_roundtrip() -> None:
    rng = np.random.default_rng(0)
    y = rng.lognormal(mean=-1.0, sigma=0.4, size=(64, 3))
    tfm = PositiveLogTransform(floor_mode="train_p001").fit(y)
    z = tfm.forward(y)
    y_rec = tfm.inverse(z)
    err = float(np.mean(np.abs(y_rec - y)))
    assert err < 1e-3
    assert np.all(y_rec >= tfm.clamp(np.zeros_like(y)) - 1e-12)


def test_objective_selection_prefers_surface_objective() -> None:
    df = pd.DataFrame(
        [
            {"model_hash": "a", "collapsed": False, "objective_value": 10.0, "val_mape": 1.0, "val_mae": 1.0},
            {"model_hash": "b", "collapsed": False, "objective_value": 2.0, "val_mape": 9.0, "val_mae": 9.0},
        ]
    )
    row = _select_row_by_hash(df, model_hash=None)
    assert str(row["model_hash"]) == "b"


def test_persist_qrc_weak_smoke() -> None:
    rng = np.random.default_rng(1)
    n_train, n_val, l, h, d = 48, 16, 14, 2, 3
    x_train = rng.normal(0.0, 0.5, size=(n_train, l, d + 6))
    x_val = rng.normal(0.0, 0.5, size=(n_val, l, d + 6))
    base_train = x_train[:, -1, :d]
    base_val = x_val[:, -1, :d]
    # mild predictable dynamics around persistence
    y_train = np.repeat(base_train[:, None, :], h, axis=1) + 0.05 * rng.normal(size=(n_train, h, d))
    y_val = np.repeat(base_val[:, None, :], h, axis=1) + 0.05 * rng.normal(size=(n_val, h, d))

    model = train_classical_forecaster(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        kind="persist_qrc_weak",
        seed=0,
        pqrc_modes=8,
        pqrc_shots=8,
        pqrc_budget=8,
        pqrc_gain=0.2,
        persist_qrc_alpha=0.25,
        persist_qrc_rclip=0.05,
        persist_qrc_drive_scale=0.25,
    )
    pred = np.asarray(model.predict(x_val), dtype=float)
    assert pred.shape == y_val.shape
    assert np.isfinite(pred).all()
    base = np.repeat(x_val[:, -1, :d][:, None, :], h, axis=1)
    shift = max(0.0, -float(np.min(np.concatenate([base, pred], axis=1))) + 1e-8)
    delta_log = np.abs(np.log(np.maximum(pred + shift, 1e-8)) - np.log(np.maximum(base + shift, 1e-8)))
    assert float(np.median(delta_log)) <= (0.25 * 0.05 + 1e-2)


def test_photonic_memory_feedback_non_inert() -> None:
    rng = np.random.default_rng(7)
    n_train, n_val, l, h, d = 64, 24, 16, 2, 3
    x_train = rng.normal(0.0, 0.6, size=(n_train, l, d + 4))
    x_val = rng.normal(0.0, 0.6, size=(n_val, l, d + 4))
    base_train = np.repeat(x_train[:, -1, :d][:, None, :], h, axis=1)
    base_val = np.repeat(x_val[:, -1, :d][:, None, :], h, axis=1)
    y_train = base_train + 0.08 * rng.normal(size=(n_train, h, d))
    y_val = base_val + 0.08 * rng.normal(size=(n_val, h, d))

    shared_kwargs = dict(
        pqrc_modes=10,
        pqrc_budget=8,
        pqrc_shots=16,
        pqrc_gain=0.8,
        pqrc_feature="coincidence",
        pqrc_in_pca=3,
        qrc_mode="auto",
        qrc_target="delta",
        qrc_baseline="persistence",
        target_transform="none",
        qrc_feat_norm="standard",
    )
    model_fb = train_classical_forecaster(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        kind="photonic_memory",
        seed=3,
        **shared_kwargs,
    )
    model_nf = train_classical_forecaster(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        kind="photonic_memory_no_feedback",
        seed=3,
        **shared_kwargs,
    )
    pred_fb = np.asarray(model_fb.predict(x_val), dtype=float)
    pred_nf = np.asarray(model_nf.predict(x_val), dtype=float)
    assert pred_fb.shape == y_val.shape
    assert pred_nf.shape == y_val.shape
    assert np.isfinite(pred_fb).all()
    assert np.isfinite(pred_nf).all()
    delta = float(np.mean(np.abs(pred_fb - pred_nf)))
    assert delta > 1e-8
    fb_sanity = getattr(model_fb, "feedback_sanity", {}) or {}
    assert bool(fb_sanity.get("feedback_inert", True)) is False


def test_photonic_qrc_feature_modes_smoke() -> None:
    rng = np.random.default_rng(13)
    n_train, n_val, l, h, d = 48, 18, 12, 2, 2
    x_train = np.abs(rng.normal(0.8, 0.3, size=(n_train, l, d + 3))) + 1e-3
    x_val = np.abs(rng.normal(0.8, 0.3, size=(n_val, l, d + 3))) + 1e-3
    base_train = np.repeat(x_train[:, -1, :d][:, None, :], h, axis=1)
    base_val = np.repeat(x_val[:, -1, :d][:, None, :], h, axis=1)
    y_train = np.clip(base_train + 0.03 * rng.normal(size=(n_train, h, d)), 1e-4, 50.0)
    y_val = np.clip(base_val + 0.03 * rng.normal(size=(n_val, h, d)), 1e-4, 50.0)

    for feature in ["coincidence", "clickprob", "pseudo_coincidence"]:
        model = train_classical_forecaster(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            kind="photonic_qrc_feedback",
            seed=5,
            pqrc_modes=8,
            pqrc_budget=8,
            pqrc_shots=16,
            pqrc_gain=0.6,
            pqrc_feature=feature,
            pqrc_pseudocount=0.5,
            pqrc_in_pca=2,
            pqrc_higher_order=(0 if feature == "clickprob" else 1),
            qrc_mode="auto",
            qrc_target="delta",
            qrc_baseline="persistence",
            target_transform="log",
            y_floor_mode="train_p001",
            qrc_feat_norm="standard",
        )
        pred = np.asarray(model.predict(x_val), dtype=float)
        assert pred.shape == y_val.shape
        assert np.isfinite(pred).all()
        assert float(np.min(pred)) > 0.0
        fs = getattr(model, "feature_stats", {}) or {}
        feat_std = np.asarray(fs.get("feat_std_train", []), dtype=float)
        assert feat_std.size > 0
        assert bool(np.any(feat_std > 1e-10))


def test_best_model_selection_matches_min_mean_surface_mape_csv(tmp_path) -> None:
    eval_rows = [
        {
            "model": PERSISTENCE_SURFACE_NAME,
            "collapse_failed": False,
            "horizon_surface_mape": np.array([4.0, 4.2, 4.4], dtype=float),
            "horizon_surface_mae": np.array([0.4, 0.42, 0.44], dtype=float),
            "horizon_surface_rmse": np.array([0.5, 0.52, 0.54], dtype=float),
            "horizon_surface_smape": np.array([3.5, 3.6, 3.7], dtype=float),
            "horizon_surface_wape": np.array([3.9, 4.0, 4.1], dtype=float),
        },
        {
            "model": "photonic_qrc_feedback",
            "collapse_failed": False,
            "horizon_surface_mape": np.array([2.2, 2.4, 2.6], dtype=float),
            "horizon_surface_mae": np.array([0.22, 0.24, 0.26], dtype=float),
            "horizon_surface_rmse": np.array([0.30, 0.32, 0.34], dtype=float),
            "horizon_surface_smape": np.array([2.0, 2.1, 2.2], dtype=float),
            "horizon_surface_wape": np.array([2.1, 2.3, 2.5], dtype=float),
        },
        {
            "model": "mlp",
            "collapse_failed": False,
            "horizon_surface_mape": np.array([2.8, 3.0, 3.2], dtype=float),
            "horizon_surface_mae": np.array([0.28, 0.30, 0.32], dtype=float),
            "horizon_surface_rmse": np.array([0.36, 0.38, 0.40], dtype=float),
            "horizon_surface_smape": np.array([2.6, 2.7, 2.8], dtype=float),
            "horizon_surface_wape": np.array([2.7, 2.9, 3.1], dtype=float),
        },
    ]
    for row in eval_rows:
        row["surface_mape"] = float(np.mean(np.asarray(row["horizon_surface_mape"], dtype=float)))
        row["surface_mae"] = float(np.mean(np.asarray(row["horizon_surface_mae"], dtype=float)))
        row["surface_rmse"] = float(np.mean(np.asarray(row["horizon_surface_rmse"], dtype=float)))
        row["surface_smape"] = float(np.mean(np.asarray(row["horizon_surface_smape"], dtype=float)))
        row["surface_wape"] = float(np.mean(np.asarray(row["horizon_surface_wape"], dtype=float)))
        row["model_hash"] = f"h_{row['model']}"
        row["qrc_config_hash"] = "cfg"
        row["dataset_signature"] = "sig"

    metrics_df = pd.DataFrame(
        {
            "model": [r["model"] for r in eval_rows],
            "collapse_failed": [bool(r.get("collapse_failed", False)) for r in eval_rows],
            "no_gain": [False] * len(eval_rows),
            "surface_metric": [float(r["surface_mape"]) for r in eval_rows],
        }
    ).sort_values(["collapse_failed", "no_gain", "surface_metric"], ascending=[True, True, True], ignore_index=True)

    out_csv = tmp_path / "mape_vs_horizon.csv"
    _write_horizon_metrics(
        eval_rows=eval_rows,
        out_csv=out_csv,
        out_png_mape=tmp_path / "mape.png",
        out_png_mae=tmp_path / "mae.png",
        out_png_rmse=tmp_path / "rmse.png",
        topk_lines=8,
    )
    horizon_df = pd.read_csv(out_csv)
    mean_by_model = horizon_df.groupby("model", as_index=False)["mape"].mean()
    best_from_csv = str(mean_by_model.sort_values("mape", ascending=True, ignore_index=True).iloc[0]["model"])
    best_from_table = str(metrics_df.iloc[0]["model"])
    assert best_from_table == best_from_csv


def test_collapse_diagnostic_surface_delta_matches_horizon_delta() -> None:
    y_true = np.array(
        [
            [[1.0, 1.2], [1.1, 1.3], [1.2, 1.4]],
            [[0.9, 1.1], [1.0, 1.2], [1.1, 1.3]],
        ],
        dtype=float,
    )
    y_persist = np.array(
        [
            [[1.0, 1.2], [1.0, 1.2], [1.0, 1.2]],
            [[0.9, 1.1], [0.9, 1.1], [0.9, 1.1]],
        ],
        dtype=float,
    )
    y_model = np.array(
        [
            [[1.02, 1.18], [1.08, 1.33], [1.15, 1.44]],
            [[0.88, 1.09], [1.03, 1.19], [1.09, 1.28]],
        ],
        dtype=float,
    )
    h_persist = _horizon_metric(y_true, y_persist, metric="mape", eps=1e-8)
    h_model = _horizon_metric(y_true, y_model, metric="mape", eps=1e-8)
    expected_delta = np.asarray(h_model - h_persist, dtype=float)

    rows = [
        {
            "model": PERSISTENCE_SURFACE_NAME,
            "model_hash": "h_persist",
            "qrc_config_hash": "cfg",
            "dataset_signature": "sig",
            "pred_test_surface": y_persist,
            "pred_test_norm": np.zeros((2, 3, 2), dtype=float),
            "horizon_surface_mape": h_persist,
        },
        {
            "model": "photonic_qrc_feedback",
            "model_hash": "h_model",
            "qrc_config_hash": "cfg",
            "dataset_signature": "sig",
            "pred_test_surface": y_model,
            "pred_test_norm": np.zeros((2, 3, 2), dtype=float),
            "horizon_surface_mape": h_model,
        },
    ]
    diag = _compute_collapse_diagnostics(rows, y_true_test_surface=y_true, eps=1e-8)
    qrc_diag = [r for r in diag if str(r.get("model", "")) == "photonic_qrc_feedback"]
    qrc_diag = sorted(qrc_diag, key=lambda r: int(r["horizon"]))
    assert len(qrc_diag) == int(y_true.shape[1])
    got_delta = np.array([float(r["mape_vs_persistence_delta_h"]) for r in qrc_diag], dtype=float)
    assert np.allclose(got_delta, expected_delta, atol=1e-12, rtol=1e-12)
