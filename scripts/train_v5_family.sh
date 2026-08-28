#!/bin/bash
# v5 family trainer — mirrors qat_sweep_v4.sh, MUSAN-trained data.
# Sizes: 20k (48/24), 40k (88/48), 80k (164/88), 100k (200/96).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

run () {
  local H1=$1 H2=$2 TAG=$3
  if [ ! -f "models/teensy-v5-${TAG}.npz" ]; then
    echo "=== float teensy-v5-${TAG} (hidden ${H1}/${H2}) ==="
    $PY -u scripts/train_v3.py --stage float --data data/prepared_v5 \
        --hidden "$H1" "$H2" --out "models/teensy-v5-${TAG}.npz" \
        > "logs/float_v5_${TAG}.log" 2>&1
    tail -1 "logs/float_v5_${TAG}.log"
  fi
  if [ ! -f "models/teensy-v5-${TAG}-qat.npz" ]; then
    echo "=== qat teensy-v5-${TAG}-qat ==="
    $PY -u scripts/train_v3.py --stage qat --data data/prepared_v5 \
        --out "models/teensy-v5-${TAG}.npz" --out-qat "models/teensy-v5-${TAG}-qat.npz" \
        --qat-epochs 10 > "logs/qat_v5_${TAG}.log" 2>&1
    tail -1 "logs/qat_v5_${TAG}.log"
  fi
}

run 48 24 20k
run 88 48 40k
run 164 88 80k
run 200 96 100k
echo V5_FAMILY_DONE
