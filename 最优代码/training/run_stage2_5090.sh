#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PBASE:?请将 PBASE 指向组委会官方连续 ERA5 根目录（其下应有 data/年份/*.h5）}"
PSTATS="${PSTATS:-$ROOT/../onedatasets/ERA5_test}"
PMETA="${PMETA:-$ROOT/training/assets/metadata.json}"
PCKPT="${PCKPT:-$ROOT/data/checkpoints/model_bak.pth}"
PYTHON="${PYTHON:-python}"
RESUME="${RESUME:-$ROOT/training_outputs/student_ratio2_mom99.pth}"
OUT="${OUT:-$ROOT/training_outputs/student_truth_ft50.pth}"

test -d "$PBASE/data" || { echo "PBASE 无 data/ 子目录: $PBASE" >&2; exit 2; }
test -f "$PSTATS/stats/global_means.npy" || { echo "缺少官方统计量: $PSTATS" >&2; exit 2; }
test -f "$PMETA" || { echo "缺少官方变量元数据: $PMETA" >&2; exit 2; }
test -f "$PCKPT" || { echo "缺少组委会官方教师权重: $PCKPT" >&2; exit 2; }
test -f "$RESUME" || { echo "缺少第一阶段学生权重: $RESUME" >&2; exit 2; }
mkdir -p "$(dirname -- "$OUT")"

export PBASE PSTATS PMETA PCKPT
exec "$PYTHON" "$ROOT/distill_truth.py" \
  --years 1980 --epochs 35 --max-cache 300 --val-frac 0.1 \
  --truth-alpha 0.5 --full-temp 1 --phys 0 --amp bf16 \
  --lr 6e-5 --warmup 2 --dtp 0 --hf-spec 0 --truth-windq 0 --truth-full15 0 \
  --global-mode grid --mlp-ratio 2.0 \
  --resume "$RESUME" --save "$OUT"
