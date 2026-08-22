#!/bin/bash
# Capacity sweep: train ~40k / ~80k / ~100k parameter models for the
# v1 / v2 / v3 families.
#
#   scripts/scale_sweep.sh v1v2   # construction + teacher labels (1M frames)
#   scripts/scale_sweep.sh v3     # scaled 10.7M-frame set (lazy windows)
#
# Hidden sizes (input 400, two hidden layers, 1 out):
#   88/48   → 39,609 params   (~40k)
#   164/88  → 80,373 params   (~80k)
#   200/96  → 99,593 params   (~100k)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

SIZES="88:48:40k 164:88:80k 200:96:100k"

if [ "$1" = "v1v2" ]; then
  for spec in $SIZES; do
    IFS=: read h1 h2 tag <<< "$spec"
    echo "=== v1-$tag (hidden $h1/$h2) ==="
    $PY scripts/train.py --hidden "$h1" "$h2" --out "models/teensy-v1-$tag.npz"
    echo "=== v2-$tag (hidden $h1/$h2) ==="
    $PY scripts/train.py --data-suffix .distill --ycol y \
        --hidden "$h1" "$h2" --out "models/teensy-v2-$tag.npz"
  done
elif [ "$1" = "v3" ]; then
  for spec in $SIZES; do
    IFS=: read h1 h2 tag <<< "$spec"
    echo "=== v3-$tag (hidden $h1/$h2) ==="
    $PY scripts/train_v3.py --stage float --hidden "$h1" "$h2" \
        --out "models/teensy-v3-$tag.npz"
  done
else
  echo "usage: $0 v1v2|v3" >&2; exit 1
fi
echo "SWEEP_$1_DONE"
