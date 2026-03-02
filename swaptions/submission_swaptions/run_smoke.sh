#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv310_arm/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python}"
fi

DATA_DIR="${1:-Quandela/Challenge_Swaptions}"
OUT_DIR="${2:-results/swaptions_smoke}"
mkdir -p "$OUT_DIR"

OUT_L1="$OUT_DIR/submission_swaptions_level1_smoke.csv"
OUT_L2="$OUT_DIR/submission_swaptions_level2_smoke.csv"

export SUBMISSION_SMOKE_FAST=1

echo "[smoke] python: $PYTHON_BIN"
echo "[smoke] data_dir: $DATA_DIR"
echo "[smoke] out_dir: $OUT_DIR"

"$PYTHON_BIN" -m submission_swaptions.solution \
  --data_dir "$DATA_DIR" \
  --level 1 \
  --lookback 26 \
  --backend sim \
  --seed 0 \
  --out_csv "$OUT_L1"

"$PYTHON_BIN" -m submission_swaptions.solution \
  --data_dir "$DATA_DIR" \
  --level 2 \
  --lookback 26 \
  --backend sim \
  --seed 0 \
  --out_csv "$OUT_L2"

"$PYTHON_BIN" - "$OUT_L1" "$OUT_L2" <<'PY'
import sys
import numpy as np
import pandas as pd

for p in sys.argv[1:]:
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise SystemExit(f"[smoke] missing Date column: {p}")
    surf = df.drop(columns=["Date"])
    if surf.isna().any().any():
        raise SystemExit(f"[smoke] NaNs found: {p}")
    vals = surf.to_numpy(dtype=float)
    if np.isinf(vals).any():
        raise SystemExit(f"[smoke] Infs found: {p}")
    if (vals < 0.0).any():
        raise SystemExit(f"[smoke] negative values found: {p}")
    print(f"[smoke] validated {p} shape={df.shape}")
PY

echo "[smoke] done: $OUT_L1 $OUT_L2"
