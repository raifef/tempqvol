# Q-ANNOT: Quantum-Annotated Option Surface Forecasting under Photonic Constraints

## Hackathon Narrative
We recast the original hybrid forecasters to the Quandela Swaptions challenge so the same classical+quantum stack operates on swaption volatility surfaces (L1 future prediction and L2 missing-data prediction).

## Why We Do Not Use Amplitude Encoding / State Injection
In MerLin photonic workflows, reliable operation is centered on low-dimensional parameterized circuits under strict mode/photon budgets.  
Amplitude encoding/state injection is not the robust path for this setup, so this package uses phase/parameter encoding into MerLin `QuantumLayer.simple(...)` with explicit input-dimension caps.

## Method: Quantum as Annotator + Reservoir-Style Training
- Train a classical base forecaster in factor space (`factor_ar`, `mlp`, `gru`, `lstm`).
- Use MerLin as a feature annotator (`q_bottleneck` or `qrc`) and fit a ridge residual readout.
- Blend `base + w * quantum_residual`, with optional regime gating from missingness/volatility.

## Photonic QRC Feedback Baseline (arXiv:2602.17440, arXiv:2512.02928)
- Implementation path: `submission_swaptions/models/photonic_qrc.py`
- Model names in the harness:
  - `photonic_qrc_feedback`
  - `photonic_qrc_no_feedback` (ablation, gain forced to 0)
  - `persist_qrc_weak` (conservative persistence + weak photonic correction)
- Software analogue of the papers:
  - fixed random linear-optical mixing unitary `U0` + programmable phase layer,
  - coarse-grained click/coincidence feature extraction (optionally shot sampled),
  - measurement-conditioned phase updates on a budgeted phase subset,
  - only linear ridge readout is trained.
- Input mapping uses low-rank PCA of factor trajectories (`--pqrc_in_pca`) to feed scalar/vector inputs to the photonic reservoir block.
- Optional positive/log target transform:
  - `--target_transform {none,log}` (default `log`)
  - `--y_floor_mode {train_p01,train_p001,fixed}`
  - `--y_floor_value` (used when `fixed`)

## Ported HybridAIQuantum Algorithms (Clearly Marked)
Ported implementations live in:
- `submission_swaptions/models_ported_hybridai.py`
- `submission_swaptions/ported_quantum_layers.py`
- `submission_swaptions/ported_budget.py`

Each copied implementation block is explicitly tagged with `COPIED-AND-ADAPTED FROM` and source paths from:
`https://github.com/Quandela/HybridAIQuantum-Challenge`.
Included ported algorithms:
- `ctrl_rff`
- `ctrl_learned_featuremap`
- `ctrl_classical_reservoir`
- `q_bottleneck`
- `qrc`

## Constraints
- `sim`: `input_dim <= 20`
- `qpu`: `input_dim <= 24`
- `inspect_merlin_layer(...)` is best-effort and warns if photon metadata cannot be verified.

## How To Run
### Simulation
```bash
python -m submission_swaptions.solution \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --backend sim \
  --seed 0 \
  --out_csv results/swaptions_level1.csv
```

### Optional QPU
```bash
python -m submission_swaptions.solution \
  --data_dir Quandela/Challenge_Swaptions \
  --level 2 \
  --backend qpu \
  --seed 0 \
  --out_csv results/swaptions_level2_qpu.csv
```

### Smoke Test
```bash
bash submission_swaptions/run_smoke.sh
```

### Compare Ported Algorithms On Swaptions
```bash
python -m submission_swaptions.compare_ported_qml \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --backend sim \
  --seed 0 \
  --out_dir results/ported_hybridai_level1
```

### One-File Full Comparison (All Classical + Native Quantum + Ported Quantum)
```bash
python -m submission_swaptions.plot_model_comparison \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --backend sim \
  --seed 0 \
  --out_dir results/model_compare_level1_all
```
Use `--fast` for a quicker all-model smoke benchmark.

### Photonic QRC Focused Run
```bash
python -m submission_swaptions.plot_model_comparison \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --model photonic_qrc_feedback \
  --skip_quantum \
  --skip_ported \
  --lookback 26 \
  --forecast_horizons 6 \
  --pqrc_M 16 \
  --pqrc_Nph 2 \
  --pqrc_budget 32 \
  --pqrc_gain 0.5 \
  --pqrc_feature coincidence \
  --pqrc_shots 32 \
  --pqrc_ridge 1e-3 \
  --seed 0 \
  --out_dir results/photonic_qrc_level1
```

### Photonic QRC Sweep + Collapse Diagnostics
```bash
python -m submission_swaptions.pqrc_sweep \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --backend sim \
  --seed 0 \
  --lookback 26 \
  --forecast_horizons 6 \
  --qrc_mode auto \
  --qrc_sweep_objective surface_mape \
  --qrc_baseline persistence \
  --target_transform log \
  --tau 1e-3 \
  --pqrc_sweep_subset 256 \
  --pqrc_sweep_topk 3 \
  --out_dir results/pqrc_sweep_level1
```
Interpretation of diagnostics:
- `level1_qrc_collapse_diag*.csv`: if `mean_abs_diff_h` is near zero across horizons, the model is collapsing to persistence.
- `corr_h` compares predicted residual vs true residual; values near zero indicate weak residual learning.
- `level1_qrc_feature_stats*.csv`: if median feature std is tiny (or many near-zero std features), the photonic feature map is under-informative.
- Rows marked `collapse_failed=True` are kept in reports but not ranked as best models.

## Reproducibility
- Set `--seed`.
- Deterministic torch flags are enabled where available.
- For stable runs set:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
- `solution.py` re-execs into `.venv310_arm` when present for consistent local behavior.
