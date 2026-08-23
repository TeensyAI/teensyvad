#!/bin/bash
# QAT sweep for the v4 family: 40k / 60k / 80k / 100k (20k already done).
# Resumable: skips any artifact that already exists. Waits for any
# in-progress trainer to finish before starting (single-trainer policy).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

# sizes: hidden1 hidden2 tag [float_model]
run () {  # $1 $2 = hidden, $3 = tag
  local H1=$1 H2=$2 TAG=$3
  # float first (60k has no float yet)
  if [ ! -f "models/teensy-v4-${TAG}.npz" ]; then
    echo "=== float teensy-v4-${TAG} (hidden ${H1}/${H2}) ==="
    $PY -u scripts/train_v3.py --stage float --data data/prepared_v4 \
        --hidden "$H1" "$H2" --out "models/teensy-v4-${TAG}.npz" \
        > "logs/float_v4_${TAG}.log" 2>&1
    tail -1 "logs/float_v4_${TAG}.log"
  fi
  # then QAT (warm-start from the float file)
  if [ ! -f "models/teensy-v4-${TAG}-qat.npz" ]; then
    echo "=== qat teensy-v4-${TAG}-qat (hidden ${H1}/${H2}) ==="
    $PY -u scripts/train_v3.py --stage qat --data data/prepared_v4 \
        --out "models/teensy-v4-${TAG}.npz" --out-qat "models/teensy-v4-${TAG}-qat.npz" \
        --qat-epochs 10 > "logs/qat_v4_${TAG}.log" 2>&1
    tail -1 "logs/qat_v4_${TAG}.log"
  fi
}

# wait for any running trainer to finish (single-trainer policy)
while pgrep -f "train_v3.py --stage" > /dev/null; do
  echo "waiting for running trainer ($(pgrep -f 'train_v3.py --stage' | head -1))..."
  sleep 60
done

run 88 48 40k      # float exists -> qat only
run 200 96 100k    # float exists -> qat only
# 80k: qat handled by the earlier launch; only if it somehow didn't finish
if [ ! -f models/teensy-v4-80k-qat.npz ]; then
  run 164 88 80k
fi

echo "QAT_V4_SWEEP_DONE"
ls -la models/teensy-v4*qat*.npz | awk '{print $NF}'
