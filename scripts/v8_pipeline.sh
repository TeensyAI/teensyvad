#!/bin/bash
# v8 campaign (Mac-sized): ~660 h prior-balanced data -> three ~500k models,
# each float-trained then QAT-fine-tuned. Every stage is guarded so the whole
# script is safe to re-run after an OOM kill: completed stages are skipped,
# training resumes from 500-batch checkpoints, conversion cache is reused.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
UTTS=192000

stage_done() { grep -q "$1" "$2" 2>/dev/null; }

# A stage passes only when its command exits 0 AND its artifact exists.
# Failures abort the pipeline with a nonzero exit so the auto-rerun loop
# reports the failure instead of silently retrying a stage that crashes
# at startup (the GRU-192 hidden-size bug burned 12 loop attempts that way).
run_stage() { # run_stage <log> <marker> <artifact> <cmd...>
  local log="$1" marker="$2" artifact="$3"; shift 3
  if stage_done "$marker" "$log" && [ -f "$artifact" ]; then
    echo "skip"
    return 0
  fi
  "$@" >> "$log" 2>&1
  local ec=$?
  if [ $ec -ne 0 ]; then
    echo "FATAL: stage failed (exit $ec): $* — see $log" >&2
    exit 1
  fi
  if [ ! -f "$artifact" ]; then
    echo "FATAL: stage finished but artifact $artifact is missing — see $log" >&2
    exit 1
  fi
}

echo "=== [1/6] prep ~660h prior-balanced mixtures ==="
if [ ! -f data/prepared_v8/F.npy ]; then
  $PY -u scripts/prepare_data_v3.py --npy --utts $UTTS \
      --musan data/raw/musan --libri100 data/raw/LibriSpeech \
      --out data/prepared_v8 >> logs/prep_v8.log 2>&1
  tail -2 logs/prep_v8.log
else
  echo "skip (features exist)"
fi

echo "=== [2/6] Silero distillation ==="
if [ ! -f data/prepared_v8/.labels_done ]; then
  $PY -u scripts/distill_label.py --data data/prepared_v8 --splits train \
      >> logs/distill_v8.log 2>&1
  tail -2 logs/distill_v8.log
else
  echo "skip (labels complete)"
fi

echo "=== [3/6] GRU-192 x3 (584k) float + QAT ==="
run_stage logs/train_v8_gru192.log TRAINING_COMPLETE models/teensy-v8-gru192.npz \
  $PY -u scripts/train_rnn.py --data data/prepared_v8 --hidden 192 --layers 3 \
      --epochs 6 --batches-per-epoch 3500 --batch 96 --qat-epochs 2 \
      --out models/teensy-v8-gru192.npz
tail -2 logs/train_v8_gru192.log
if ! grep -q "GRU_CAL_DONE" logs/train_v8_gru192.log; then
  $PY scripts/eval_gru.py --model models/teensy-v8-gru192.npz --calibrate \
      >> logs/train_v8_gru192.log 2>&1 && echo "GRU_CAL_DONE" >> logs/train_v8_gru192.log
fi

echo "=== [4/6] causal transformer d128 x4 (~525k) float + QAT ==="
run_stage logs/train_v8_tt.log TRAINING_COMPLETE models/teensy-v8-tt.npz \
  $PY -u scripts/train_transformer.py --data data/prepared_v8 --d-model 128 \
      --layers 4 --heads 8 --epochs 5 --batches-per-epoch 2500 --batch 64 \
      --qat-epochs 2 --out models/teensy-v8-tt.npz
tail -2 logs/train_v8_tt.log
if ! grep -q "TT_CAL_DONE" logs/train_v8_tt.log; then
  $PY scripts/eval_transformer.py --model models/teensy-v8-tt.npz --calibrate \
      >> logs/train_v8_tt.log 2>&1 && echo "TT_CAL_DONE" >> logs/train_v8_tt.log
fi

echo "=== [5/6] MLP a3 ctx25 448/224 (~549k) float + QAT ==="
# batch 1024: the default 2048 was OOM-killed (macOS memory pressure) on the
# 282M-frame v8 set; the run must survive, not battle the machine.
run_stage logs/train_v8_a3.log TRAINING_COMPLETE models/teensy-v8-a3.npz \
  $PY -u scripts/train_v3.py --stage all --data data/prepared_v8 \
      --context 25 --hidden 448 224 --batch-size 1024 \
      --out models/teensy-v8-a3.npz \
      --out-qat models/teensy-v8-a3-qat.npz
tail -2 logs/train_v8_a3.log

echo "=== [6/6] calibrate + benchmark a3 ==="
if [ -f models/teensy-v8-a3.npz ] && ! grep -q "A3_BENCH_DONE" logs/train_v8_a3.log; then
  $PY scripts/calibrate_realworld.py --model models/teensy-v8-a3.npz >> logs/train_v8_a3.log 2>&1 || true
  $PY scripts/calibrate_realworld.py --model models/teensy-v8-a3-qat.npz >> logs/train_v8_a3.log 2>&1 || true
  $PY scripts/compare_all.py --models models/teensy-v8-a3.npz models/teensy-v8-a3-qat.npz \
      -o models/comparison_v8_a3.json >> logs/train_v8_a3.log 2>&1 || true
  echo "A3_BENCH_DONE" >> logs/train_v8_a3.log
fi
if [ -f models/teensy-v8-gru192.npz ] && [ -f models/teensy-v8-tt.npz ] && \
   [ -f models/teensy-v8-a3.npz ] && [ -f models/teensy-v8-a3-qat.npz ] && \
   [ -f models/comparison_v8_a3.json ]; then
  echo V8_PIPELINE_COMPLETE
else
  echo V8_PIPELINE_INCOMPLETE
fi
