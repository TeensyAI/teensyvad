"""Train ALL of teensyvad end to end — one command, resumable.

    .venv/bin/python scripts/train_all.py              # everything, from scratch
    .venv/bin/python scripts/train_all.py --skip download extract   # data already there
    .venv/bin/python scripts/train_all.py --only quantize export    # one stage

Stages (each skipped automatically if its output already exists,
so the script is safe to re-run / resume):

    download   fetch LibriSpeech dev/test + ESC-50          → data/downloads
    extract    untar/unzip                                   → data/raw
    prepare    flac→8kHz wav, mixtures @0–20dB SNR + wavs    → data/prepared/
    distill    Silero teacher relabels every frame (optional; needs silero-vad)
    train-v1   student on construction labels                → models/teensy-v1.npz
    train-v2   student on teacher labels (hard + soft)       → models/teensy-v2*.npz
    calibrate  event-level thresholds for v1/v2
    evaluate   frame + event metrics, teacher agreement      (evaluate_distill)
    quantize   per-layer sensitivity → int8 + selective npz  → models/*int8*.npz
    export     ONNX float32 + dynamic-int8 + speed table     → models/*.onnx
    tests      pytest quick pass

Design doctrine: 8 kHz, end to end, no resampling anywhere — PSTN,
G.711 µ-law/A-law and Asterisk `slin` are all 8 kHz; we never upsample
to a 16 kHz core the way some toolkits do.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "downloads"
RAW = ROOT / "data" / "raw"
PREP = ROOT / "data" / "prepared"
MODELS = ROOT / "models"

DOWNLOADS = [
    ("dev-clean.tar.gz", "https://www.openslr.org/resources/12/dev-clean.tar.gz"),
    ("test-clean.tar.gz", "https://www.openslr.org/resources/12/test-clean.tar.gz"),
    ("esc50.zip", "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"),
]

PY = sys.executable


def sh(cmd: list[str], **kw) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def have(p: Path) -> bool:
    return p.exists()


# ---------------------------------------------------------------- stages

def stage_download():
    DL.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS:
        out = DL / name
        if have(out) and out.stat().st_size > 1_000_000:
            print(f"skip {name} (already downloaded)")
            continue
        sh(["curl", "-L", "--retry", "3", "-o", str(out), url])


def stage_extract():
    RAW.mkdir(parents=True, exist_ok=True)
    if not have(RAW / "LibriSpeech" / "dev-clean"):
        sh(["tar", "xzf", str(DL / "dev-clean.tar.gz"), "-C", str(RAW)])
    if not have(RAW / "LibriSpeech" / "test-clean"):
        sh(["tar", "xzf", str(DL / "test-clean.tar.gz"), "-C", str(RAW)])
    if not have(RAW / "ESC-50-master"):
        sh(["unzip", "-q", str(DL / "esc50.zip"), "-d", str(RAW)])
    if shutil.which("afconvert") is None:
        print("!! macOS `afconvert` not found — prepare stage needs it "
              "(on Linux, swap in sox/ffmpeg in prepare_data.py)")


def stage_prepare():
    outs = [PREP / f"{s}.npz" for s in ("train", "val", "test")]
    if all(map(have, outs)) and have(PREP / "audio"):
        print("skip prepare (npz + audio present)")
        return
    sh([PY, "scripts/prepare_data.py", "--train", "1200", "--val", "200",
        "--test", "300", "--save-audio"])


def teacher_available() -> bool:
    try:
        import silero_vad  # noqa: F401
        return True
    except ImportError:
        return False


def stage_distill():
    if not teacher_available():
        print("skip distill (silero-vad not installed — v2 models need it;\n"
              "      .venv/bin/pip install silero-vad onnxruntime)")
        return
    if all(have(PREP / f"{s}.distill.npz") for s in ("train", "val", "test")):
        print("skip distill (labels present)")
        return
    sh([PY, "scripts/distill_label.py"])


def stage_train_v1():
    if have(MODELS / "teensy-v1.npz"):
        print("skip train-v1")
        return
    sh([PY, "scripts/train.py", "--out", str(MODELS / "teensy-v1.npz")])


def stage_train_v2():
    if not have(PREP / "train.distill.npz"):
        print("skip train-v2 (no distilled labels — run distill stage)")
        return
    if not have(MODELS / "teensy-v2.npz"):
        sh([PY, "scripts/train.py", "--data-suffix", ".distill", "--ycol", "y",
            "--out", str(MODELS / "teensy-v2.npz")])
    if not have(MODELS / "teensy-v2-soft.npz"):
        sh([PY, "scripts/train.py", "--data-suffix", ".distill", "--ycol", "ysoft",
            "--out", str(MODELS / "teensy-v2-soft.npz")])


def stage_calibrate():
    for m in ("teensy-v1.npz", "teensy-v2.npz"):
        p = MODELS / m
        if not have(p):
            continue
        meta = str(subprocess.run([PY, "-c",
                                   f"import numpy as np;print(np.load('{p}',allow_pickle=False)['meta'])"],
                                  capture_output=True, text=True, cwd=ROOT).stdout)
        if "event_f1_val" in meta:
            print(f"skip calibrate {m}")
            continue
        sh([PY, "scripts/calibrate_events.py", "--model", str(p)])


def stage_evaluate():
    m = MODELS / "teensy-v2.npz"
    if not have(m):
        m = MODELS / "teensy-v1.npz"
    if have(PREP / "test.distill.npz"):
        sh([PY, "scripts/evaluate_distill.py", "--model", str(m), "--clips", "60"])
    else:
        sh([PY, "scripts/evaluate.py"])


def stage_quantize():
    src = MODELS / "teensy-v2.npz"
    if not have(src):
        src = MODELS / "teensy-v1.npz"
        if not have(src):
            print("skip quantize (no model)")
            return
    if have(MODELS / f"{src.stem}-int8.npz"):
        print("skip quantize")
        return
    sh([PY, "scripts/quantize.py", "--model", str(src)])


def stage_qat():
    """PTQ vs QAT vs wide bake-off (also produces the -qat and wide models)."""
    if not have(PREP / "train.distill.npz"):
        print("skip qat (needs distilled labels)")
        return
    if have(MODELS / "teensy-v2-qat.npz"):
        print("skip qat")
        return
    sh([PY, "scripts/qat_bakeoff.py"])


def stage_v3():
    """Scaled training: train-clean-100 + babble + AMI ambience noise."""
    libri100 = RAW / "LibriSpeech" / "train-clean-100"
    if not libri100.exists():
        print("skip v3 (train-clean-100 not extracted — run download/extract)")
        return
    if not have(Path("data/prepared_v3/train.distill.npz")):
        sh([PY, "scripts/prepare_data_v3.py", "--utts", "8000"])
        sh([PY, "scripts/distill_label.py", "--data", "data/prepared_v3",
            "--splits", "train"])
    if not have(MODELS / "teensy-v3.npz"):
        sh([PY, "scripts/train_v3.py"])


def stage_calibrate_real():
    """Real-audio operating points (AMI dev meetings) for v2/v3 models."""
    for name in ("teensy-v3.npz", "teensy-v3-qat.npz", "teensy-v2.npz"):
        p = MODELS / name
        if not have(p):
            continue
        meta = subprocess.run([PY, "-c",
                               f"import numpy as np;print(np.load('{p}')['meta'])"],
                              capture_output=True, text=True, cwd=ROOT).stdout
        if "calibrated_on" in meta:
            print(f"skip calibrate-real {name}")
            continue
        sh([PY, "scripts/calibrate_realworld.py", "--model", str(p)])


def stage_realworld():
    """Human-labelled real recordings: TEN VAD set + AMI SDM meetings."""
    if not have(RAW / "ten-vad" / "testset"):
        sh(["git", "clone", "-q", "--depth", "1",
            "https://github.com/TEN-framework/ten-vad.git", str(RAW / "ten-vad")])
    sh([PY, "scripts/eval_realworld.py", "--ami", "--models",
        str(MODELS / "teensy-v1.npz"), str(MODELS / "teensy-v2.npz"),
        str(MODELS / "teensy-v2-qat.npz"), str(MODELS / "teensy-v3.npz"),
        str(MODELS / "teensy-v3-qat.npz")])


def stage_export():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        print("skip export (pip install onnx onnxruntime)")
        return
    src = MODELS / "teensy-v2.npz"
    if not have(src):
        print("skip export (no v2 model)")
        return
    if have(MODELS / "teensy-v2-int8.onnx"):
        print("skip export")
        return
    sh([PY, "scripts/export_onnx.py", "--model", str(src)])


def stage_tests():
    sh([PY, "-m", "pytest", "tests/", "-q"])


STAGES = {n.replace("stage_", "").replace("_", "-"): f for n, f in
          sorted(globals().items()) if n.startswith("stage_")}
ORDER = ["download", "extract", "prepare", "distill", "train-v1", "train-v2",
         "calibrate", "evaluate", "quantize", "qat", "v3", "calibrate-real",
         "realworld", "export", "tests"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=ORDER, help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=ORDER, default=[], help="skip these stages")
    args = ap.parse_args()

    todo = args.only or [s for s in ORDER if s not in args.skip]
    print("teensyvad end-to-end")
    print(f"stages: {' → '.join(todo)}\n")
    timings = {}
    for s in todo:
        t0 = time.time()
        print(f"===== [{s}] " + "=" * (50 - len(s)))
        STAGES[s]()
        timings[s] = time.time() - t0

    print("\n" + "=" * 62)
    print("END-TO-END SUMMARY")
    print("=" * 62)
    for s, t in timings.items():
        print(f"  {s:<12} {t:7.1f}s")
    print(f"  {'TOTAL':<12} {sum(timings.values()):7.1f}s")
    print("\nartifacts:")
    for p in sorted(MODELS.glob("*")):
        print(f"  models/{p.name:<28} {p.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
