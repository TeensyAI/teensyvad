#!/bin/bash
# v8 campaign: ~960 h prior-balanced data → three 500k-class models (float + QAT)
# GRU-192x3 (584k) | causal transformer d128x4 (~525k) | MLP a3 ctx25 448/224 (~549k)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

echo "=== [1/6] prep 960h prior-balanced mixtures ==="
if [ ! -f data/prepared_v8/yteach.npy ]; then
  $PY -u scripts/prepare_data_v3.py --npy --utts 285000 \
      --musan data/raw/musan --libri100 data/raw/LibriSpeech-960 \
      --out data/prepared_v8 > logs/prep_v8.log 2>&1
  tail -2 logs/prep_v8.log
fi

echo "=== [2/6] Silero distillation ==="
if [ ! -f data/prepared_v8/yteach.npy ]; then
  $PY -u scripts/distill_label.py --data data/prepared_v8 --splits train \
      > logs/distill_v8.log 2>&1
  tail -2 logs/distill_v8.log
fi

echo "=== [3/6] GRU-192 x3 layers (584k) float + QAT ==="
$PY -u scripts/train_rnn.py --data data/prepared_v8 --hidden 192 --layers 3 \
    --epochs 8 --batches-per-epoch 12000 --batch 96 --qat-epochs 2 \
    --out models/teensy-v8-gru192.npz >> logs/train_v8_gru192.log 2>&1
tail -2 logs/train_v8_gru192.log
$PY scripts/eval_gru.py --model models/teensy-v8-gru192.npz --calibrate >> logs/train_v8_gru192.log 2>&1
$PY scripts/eval_gru.py --model models/teensy-v8-gru192.npz --eval \
    --out models/comparison_v8_gru192.json >> logs/train_v8_gru192.log 2>&1
tail -2 logs/train_v8_gru192.log

echo "=== [4/6] causal transformer d128 x4 (~525k) float + QAT ==="
$PY -u scripts/train_transformer.py --data data/prepared_v8 --d-model 128 \
    --layers 4 --heads 8 --epochs 6 --batches-per-epoch 12000 --batch 96 \
    --qat-epochs 2 --out models/teensy-v8-tt.npz >> logs/train_v8_tt.log 2>&1
tail -2 logs/train_v8_tt.log
$PY scripts/eval_transformer.py --model models/teensy-v8-tt.npz --calibrate >> logs/train_v8_tt.log 2>&1
$PY scripts/eval_transformer.py --model models/teensy-v8-tt.npz --eval \
    --out models/comparison_v8_tt.json >> logs/train_v8_tt.log 2>&1
tail -2 logs/train_v8_tt.log

echo "=== [5/6] MLP a3 ctx25 448/224 (~549k) float + QAT ==="
$PY -u scripts/train_v3.py --stage all --data data/prepared_v8 \
    --context 25 --hidden 448 224 --out models/teensy-v8-a3.npz \
    --out-qat models/teensy-v8-a3-qat.npz >> logs/train_v8_a3.log 2>&1
tail -2 logs/train_v8_a3.log

echo "=== [6/6] calibrate + benchmark a3 ==="
$PY scripts/calibrate_realworld.py --model models/teensy-v8-a3.npz >> logs/train_v8_a3.log 2>&1
$PY scripts/calibrate_realworld.py --model models/teensy-v8-a3-qat.npz >> logs/train_v8_a3.log 2>&1
$PY scripts/compare_all.py --models models/teensy-v8-a3.npz models/teensy-v8-a3-qat.npz \
    -o models/comparison_v8_a3.json >> logs/train_v8_a3.log 2>&1

echo V8_PIPELINE_COMPLETE
