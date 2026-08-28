"""Adaptive parameter-scaling search above 100k, chasing Silero accuracy.

    .venv/bin/python scripts/v5_scaling_search.py        # logs + JSON state

Rule (per Pankaj):
  1. train ~150k (hidden 280/130). Calibrate on AMI dev, benchmark TEN+AMI.
  2. If it BEATS Silero on TEN AUC (> 0.9519): walk DOWN (130k, 120k, …)
     to find the smallest winner.
  3. If it improves on v5-100k (0.8865) but does NOT beat Silero: increase
     toward ~200k (hidden 350/165), sized by the improvement trend, then stop.
  4. If it does not even improve on 100k: still train 200k as the final
     capacity-ceiling data point.
Every decision + result is appended to logs/v5_scaling_search.log and
models/v5_scaling_results.json so the ablation is fully auditable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
LOG = ROOT / "logs" / "v5_scaling_search.log"
STATE = ROOT / "models" / "v5_scaling_results.json"
SILERO_TEN_AUC = 0.9519
V5_100K_TEN_AUC = 0.8865
SIZES = {
    "150k": (280, 130),
    "200k": (350, 165),
    "130k": (240, 110),
    "120k": (210, 100),
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def train(tag: str) -> None:
    h1, h2 = SIZES[tag]
    out = ROOT / "models" / f"teensy-v5-{tag}.npz"
    if out.exists():
        log(f"{tag}: checkpoint exists, skipping training")
        return
    log(f"{tag}: training float (hidden {h1}/{h2})")
    sh([str(PY), "-u", "scripts/train_v3.py", "--stage", "float",
        "--data", "data/prepared_v5", "--hidden", str(h1), str(h2),
        "--out", str(out), "--batch-size", "2048"])


def calibrate(tag: str) -> None:
    log(f"{tag}: calibrating on AMI dev")
    sh([str(PY), "scripts/calibrate_realworld.py",
        "--model", str(ROOT / "models" / f"teensy-v5-{tag}.npz")])


def benchmark(tag: str) -> dict:
    log(f"{tag}: benchmarking (TEN + held-out AMI)")
    out = ROOT / "models" / f"comparison_v5_{tag}.json"
    sh([str(PY), "scripts/compare_all.py",
        "--models", str(ROOT / "models" / f"teensy-v5-{tag}.npz"),
        "-o", str(out)])
    rows = json.loads(out.read_text())["rows"]
    row = next(r for r in rows if r["model"].startswith("v5"))
    return {k: row.get(k) for k in
            ("params", "kb", "ten_f1", "ten_auc", "ami_f1", "ami_auc",
             "us_per_20ms")}


def record(tag: str, res: dict) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state[f"v5-{tag}"] = res
    STATE.write_text(json.dumps(state, indent=2))
    log(f"{tag}: {res}")


def main() -> None:
    LOG.touch()
    log("=== v5 scaling search start ===")
    results = {}

    # step 1: 150k
    train("150k")
    calibrate("150k")
    results["150k"] = benchmark("150k")
    record("150k", results["150k"])
    ten150 = results["150k"]["ten_auc"] or 0.0

    if ten150 > SILERO_TEN_AUC:
        log(f"150k BEATS Silero on TEN AUC ({ten150:.4f} > {SILERO_TEN_AUC}). "
            f"Walking down: 130k, then 120k if still winning.")
        for smaller in ("130k", "120k"):
            train(smaller)
            calibrate(smaller)
            results[smaller] = benchmark(smaller)
            record(smaller, results[smaller])
            if (results[smaller]["ten_auc"] or 0.0) <= SILERO_TEN_AUC:
                log(f"{smaller} no longer beats Silero — smallest winner found "
                    f"above this size.")
                break
    else:
        delta = ten150 - V5_100K_TEN_AUC
        log(f"150k does not beat Silero (TEN AUC {ten150:.4f} vs "
            f"{SILERO_TEN_AUC}); delta vs 100k = {delta:+.4f}. "
            f"Training 200k as the capacity-ceiling point.")
        train("200k")
        calibrate("200k")
        results["200k"] = benchmark("200k")
        record("200k", results["200k"])
        ten200 = results["200k"]["ten_auc"] or 0.0
        if ten200 > SILERO_TEN_AUC:
            log("200k beats Silero — running 130k to bracket the smallest "
                "winner.")
            train("130k")
            calibrate("130k")
            results["130k"] = benchmark("130k")
            record("130k", results["130k"])

    log("=== v5 scaling search complete ===")
    log(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
