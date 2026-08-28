"""V6 experiment queue: context widening + hard-example mining.

Runs AFTER the v5 parameter-scaling search finishes (single-trainer policy),
then trains each variant from data/prepared_v5, calibrates on AMI dev, and
benchmarks on the clean TEN + AMI protocol. All comparison JSONs and a
summary state file are kept for the ablation table.

    .venv/bin/python scripts/v6_queue.py        # logs → logs/v6_queue.log

Variants (float only; size kept tiny on purpose — speed is a requirement):
  a1-k25-24k : context 25 (250 ms), hidden 24/16  (~24k params)
  a2-k25-49k : context 25 (250 ms), hidden 48/24  (~49k params)
  c-20k      : context 10, hard-example oversampling (20k params)

Selection rule (recorded, applied by the parent):
  BEATS_SILERO_TEN = ten_auc > 0.9519  (Silero's only remaining win)
  Among winners, prefer the fastest (µs/20ms) then smallest file.
  AMI F1 already beats Silero for every v5/v6 variant.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
LOG = ROOT / "logs" / "v6_queue.log"
STATE = ROOT / "models" / "v6_results.json"
SILERO_TEN_AUC = 0.9519

VARIANTS = [
    dict(tag="a1-k25-24k", extra=["--context", "25", "--hidden", "24", "16"]),
    dict(tag="a2-k25-49k", extra=["--context", "25", "--hidden", "48", "24"]),
    dict(tag="c-20k", extra=["--oversample-disagreement",
                             "--hidden", "48", "24"]),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_for_scaling_search() -> None:
    log("waiting for the v5 scaling search to finish (single-trainer policy)…")
    while subprocess.run(["pgrep", "-f", "v5_scaling_search"],
                         capture_output=True).returncode == 0:
        time.sleep(60)
    log("scaling search finished — starting v6 queue")


def record(tag: str, res: dict) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state[tag] = res
    state[tag]["beats_silero_ten"] = bool(
        (res.get("ten_auc") or 0.0) > SILERO_TEN_AUC)
    STATE.write_text(json.dumps(state, indent=2))
    log(f"{tag}: {state[tag]}")


def run_variant(v: dict) -> None:
    tag = v["tag"]
    model = ROOT / "models" / f"teensy-v6-{tag}.npz"
    if not model.exists():
        log(f"{tag}: training")
        sh([str(PY), "-u", "scripts/train_v3.py", "--stage", "float",
            "--data", "data/prepared_v5", "--out", str(model),
            "--batch-size", "2048", *v["extra"]])
    log(f"{tag}: calibrating")
    sh([str(PY), "scripts/calibrate_realworld.py", "--model", str(model)])
    log(f"{tag}: benchmarking")
    cmp_out = ROOT / "models" / f"comparison_v6_{tag}.json"
    sh([str(PY), "scripts/compare_all.py", "--models", str(model),
        "-o", str(cmp_out)])
    rows = json.loads(cmp_out.read_text())["rows"]
    row = next(r for r in rows if r["model"].startswith("v6"))
    record(tag, {k: row.get(k) for k in
                 ("params", "kb", "ten_f1", "ten_auc", "ami_f1", "ami_auc",
                  "us_per_20ms")})


def main() -> None:
    LOG.touch()
    wait_for_scaling_search()
    for v in VARIANTS:
        try:
            run_variant(v)
        except subprocess.CalledProcessError as e:
            log(f"{v['tag']}: FAILED ({e}) — continuing with next variant")
    log("=== v6 queue complete ===")
    if STATE.exists():
        state = json.loads(STATE.read_text())
        winners = [(t, r) for t, r in state.items() if r.get("beats_silero_ten")]
        if winners:
            best = min(winners, key=lambda tr_: (tr_[1]["us_per_20ms"] or 9e9,
                                                 tr_[1]["kb"] or 9e9))
            log(f"smallest/fastest TEN-Silero-beater: {best[0]} → {best[1]}")
        else:
            log("no variant beat Silero on TEN AUC; best-by-F1 per benchmark "
                "recorded in state file")


if __name__ == "__main__":
    main()
