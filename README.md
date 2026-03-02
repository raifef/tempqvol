# tempqvol
use these args for decent performance \\
PYTHONPATH=. python -m submission_swaptions.plot_model_comparison \
  --data_dir Quandela/Challenge_Swaptions \
  --level 1 \
  --forecast_horizons 6 \
  --lookback 14 \
  --run_quantum \
  --pqrc_modes 8 \
  --pqrc_nphotons 2 \
  --pqrc_shots 32 \
  --pqrc_feature clickprob \
  --pqrc_pseudocount 0.25 \
  --pqrc_higher_order 0 \
  --pqrc_in_pca 4 \
  --pqrc_factor_cap 0 \
  --pqrc_gain 0.7565800181198181 \
  --pqrc_input_scale 0.6689556127535666 \
  --pqrc_ridge 0.0051830988569115735 \
  --qrc_mode residual \
  --qrc_target delta \
  --qrc_baseline persistence \
  --target_transform log \
  --y_floor_mode train_p001 \
  --qrc_gate_tau 0.06184302083987248 \
  --seed 0 \
  --out_dir results/all_models_l1_h6_refined_qrc
