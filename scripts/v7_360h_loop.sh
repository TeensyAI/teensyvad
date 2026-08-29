#!/bin/bash
# Auto-resume wrapper: relaunches the 360h GRU training until it completes.
# Each attempt resumes from the 500-batch checkpoint, so machine memory
# pressure only costs time, never progress.
cd "$(dirname "$0")/.."
while ! grep -q "TRAINING_COMPLETE" logs/train_v7_gru96_360h.log 2>/dev/null; do
  echo "=== attempt $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/train_rnn.py --data data/prepared_v7 --hidden 96 \
    --epochs 10 --batches-per-epoch 5000 --batch 128 \
    --out models/teensy-v7-gru96-360h.npz >> logs/train_v7_gru96_360h.log 2>&1
  ec=$?
  echo "attempt exited $ec at $(date +%H:%M:%S)"
  grep -q "TRAINING_COMPLETE" logs/train_v7_gru96_360h.log && break
  sleep 20
done
echo V7_360H_TRAINING_COMPLETE
