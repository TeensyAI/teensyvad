#!/bin/bash
# v8 campaign loop wrapper: re-runs the guarded pipeline until it completes.
# Every stage is resumable (conversion cache, 500-batch training checkpoints,
# label files), so OOM kills only cost time.
cd /Users/pankajdoharey/Development/Projects/ML/teensyvad
attempts=0
max_attempts=10
while ! grep -q "V8_PIPELINE_COMPLETE" logs/v8_pipeline_state.log 2>/dev/null; do
  attempts=$((attempts + 1))
  if [ "$attempts" -gt "$max_attempts" ]; then
    echo "V8_FAILED: $max_attempts attempts without completion — inspect logs/, do not blind-retry"
    exit 1
  fi
  echo "=== pipeline attempt $attempts $(date +%H:%M:%S) ==="
  ./scripts/v8_pipeline.sh >> logs/v8_pipeline_state.log 2>&1
  grep -q "V8_PIPELINE_COMPLETE" logs/v8_pipeline_state.log && break
  echo "attempt died — retrying in 30s"
  sleep 30
done
echo V8_ALL_DONE
