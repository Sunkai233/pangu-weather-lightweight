#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PBASE:?请将 PBASE 指向组委会官方 ERA5 根目录（其下应有 data/年份/*.h5）}"
PSTATS="${PSTATS:-$ROOT/../onedatasets/ERA5_test}"
PMETA="${PMETA:-$ROOT/training/assets/metadata.json}"
PCKPT="${PCKPT:-$ROOT/data/checkpoints/model_bak.pth}"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-$ROOT/training_outputs/student_ratio2_mom99.pth}"

test -d "$PBASE/data" || { echo "PBASE 无 data/ 子目录: $PBASE" >&2; exit 2; }
test -f "$PSTATS/stats/global_means.npy" || { echo "缺少官方统计量: $PSTATS" >&2; exit 2; }
test -f "$PMETA" || { echo "缺少官方变量元数据: $PMETA" >&2; exit 2; }
test -f "$PCKPT" || { echo "缺少组委会官方教师权重: $PCKPT" >&2; exit 2; }
mkdir -p "$(dirname -- "$OUT")"

export PBASE PSTATS PMETA PCKPT
exec "$PYTHON" "$ROOT/distill_cache.py" \
  --opt muon --muon-lr 0.02 --muon-momentum 0.99 \
  --global-mode grid --depths 2,4,2 --heads 6,12,6 \
  --embed 96 --patch 2,16,16 --mlp-ratio 2.0 \
  --train-years 1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996 \
  --val-years 1997 --epochs 80 --max-cache 300 --amp bf16 --aug 1 \
  --residual 0 --ivw 0 --dtp 0 --spec 0 --freeze-backbone 0 \
  --save "$OUT"
