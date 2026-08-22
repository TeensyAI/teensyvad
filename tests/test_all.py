"""teensyvad test suite — also a guided tour of the moving parts.

Run:  .venv/bin/python -m pytest tests/ -v
"""

import numpy as np
import pytest

from teensyvad.audio import (dbfs, float_to_pcm16, mulaw_decode,
                             mulaw_encode, pcm16_to_float, read_wav,
                             resample_fft, rms, telephony_roundtrip,
                             write_wav)
from teensyvad.features import LogMel, StreamingLogMel, mel_filterbank
from teensyvad.model import MLP, Adam, auc, bce_with_logits, prf, train
from teensyvad.streaming import Hysteresis, StreamingVAD, hysteresis_events
from teensyvad.energy_vad import EnergyVAD

RNG = np.random.default_rng(42)


# ==========================================================================
# audio.py
# ==========================================================================

def test_pcm16_roundtrip():
    x = RNG.uniform(-0.99, 0.99, 5000).astype(np.float32)
    y = pcm16_to_float(float_to_pcm16(x))
    assert np.abs(x - y).max() < 1 / 16000          # ≤ half a quantisation step


def test_mulaw_roundtrip_error_small():
    x = RNG.uniform(-0.99, 0.99, 20000).astype(np.float32)
    y = telephony_roundtrip(x)
    err = np.abs(x - y)
    assert err.mean() < 0.02                        # ~1.5% mean error is fine
    assert err.max() < 0.12                         # worst case near zero-crossings
    # µ-law's whole point: relative error is roughly level-independent.
    # (denominator floored so samples hovering at 0 don't blow up the ratio)
    rel = err / np.maximum(np.abs(x), 0.02)
    loud = np.abs(x) > 0.5
    quiet = np.abs(x) < 0.05
    assert rel[loud].mean() < 0.05
    assert rel[quiet].mean() < 0.10


def test_mulaw_extreme_values():
    x = np.array([0.0, -1.0, 0.999, -0.999], dtype=np.float32)
    y = telephony_roundtrip(x)
    assert np.all(np.abs(y) <= 1.0)
    assert abs(y[0]) < 0.01                         # silence stays silent


def test_resample_preserves_tone():
    sr, f = 16000, 440.0
    t = np.arange(sr * 2) / sr                      # 2 s of A4
    x = np.sin(2 * np.pi * f * t).astype(np.float32)
    y = resample_fft(x, sr, 8000)
    assert len(y) == 16000
    # dominant frequency must survive
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak = np.fft.rfftfreq(len(y), 1 / 8000)[np.argmax(spec)]
    assert abs(peak - 440.0) < 10.0


def test_wav_io_roundtrip(tmp_path):
    x = RNG.uniform(-0.9, 0.9, 8000).astype(np.float32)
    p = tmp_path / "t.wav"
    write_wav(p, x, 8000)
    y, sr = read_wav(p)
    assert sr == 8000 and np.abs(x - y).max() < 1 / 16000


def test_rms_dbfs_silence():
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0
    assert dbfs(np.zeros(1000, dtype=np.float32)) == -100.0
    full = np.ones(1000, dtype=np.float32)
    assert abs(dbfs(full)) < 1e-6


# ==========================================================================
# features.py
# ==========================================================================

def test_mel_filterbank_shapes_and_norm():
    W, centers = mel_filterbank(8000, 256, n_mels=20)
    assert W.shape == (20, 129)
    assert centers.shape == (20,)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-5)   # rows sum to 1
    assert np.all(W >= 0)
    assert np.all(np.diff(centers) > 0)                 # bands are sorted
    assert centers[0] >= 80.0                           # fmin honoured
    assert centers[-1] <= 3900.0                       # below Nyquist


def test_logmel_shape_and_gain_invariance():
    lm = LogMel(sr=8000)
    x = RNG.uniform(-0.5, 0.5, 16000).astype(np.float32)
    F1 = lm(x)
    assert F1.shape[1] == lm.dim == 40                 # 20 mels + 20 deltas
    assert F1.shape[0] == (16000 - 200) // 80 + 1      # 25ms/10ms framing
    # gain invariance: 10x louder → same features (band-mean subtraction)
    F2 = lm(x * 10.0)
    assert np.abs(F1[:, :20] - F2[:, :20]).max() < 0.5


def test_streaming_equals_offline_exactly():
    """The single most important property for a streaming VAD: feeding
    chunks must produce bit-identical features to offline processing."""
    lm_off = LogMel(sr=8000)
    lm_str = StreamingLogMel(sr=8000)
    x = RNG.uniform(-0.5, 0.5, 8000 * 3).astype(np.float32)
    offline = lm_off(x)

    chunks, i = [], 0
    while i < len(x):
        n = int(RNG.integers(40, 400))                # ragged chunk sizes!
        chunks.append(x[i:i + n])
        i += n
    streamed = np.concatenate([lm_str.feed(c) for c in chunks])
    assert streamed.shape == offline.shape
    # Not bitwise: batched vs single-row float32 matmuls reduce in
    # different orders (≈1e-6).  Alignment itself must be perfect.
    np.testing.assert_allclose(streamed, offline, rtol=0, atol=1e-5)


def test_streaming_reset():
    lm = StreamingLogMel(sr=8000)
    x = RNG.uniform(-0.5, 0.5, 4000).astype(np.float32)
    a = lm.feed(x)
    lm.reset()
    b = lm.feed(x)
    assert np.array_equal(a, b)


# ==========================================================================
# model.py — including a numerical gradient check
# ==========================================================================

def tiny_model():
    return MLP(sizes=(10, 8, 6, 1), seed=7)


def test_forward_shapes():
    m = tiny_model()
    X = RNG.normal(size=(4, 10)).astype(np.float32)
    assert m.logits(X).shape == (4, 1)
    p = m.probs(X)
    assert p.shape == (4,) and np.all((p > 0) & (p < 1))
    assert 48_000 > m.n_params() > 100


def test_backprop_matches_numerical_gradient():
    """Finite-difference check: analytic grads ≈ (L(θ+ε) − L(θ−ε)) / 2ε.

    ReLU has a kink at 0: if any pre-activation sits within ±ε of zero,
    the two-sided difference crosses the kink and the comparison is
    meaningless.  So we first shop for a batch whose pre-activations all
    sit at least 3ε away from the kink — then both sides of the
    difference live in the same linear region and must agree tightly.
    """
    m = tiny_model()
    eps = 5e-4
    X = y = None
    for seed in range(500):
        rng = np.random.default_rng(seed)
        X_c = rng.normal(size=(16, 10)).astype(np.float32)
        y_c = rng.integers(0, 2, 16).astype(np.float32)
        h1_pre = X_c @ m.p["W1"] + m.p["b1"]
        h2_pre = np.maximum(h1_pre, 0) @ m.p["W2"] + m.p["b2"]
        if min(np.abs(h1_pre).min(), np.abs(h2_pre).min()) > 3 * eps:
            X, y = X_c, y_c
            break
    assert X is not None, "no kink-free batch found"

    _, grads = m.loss_and_grads(X, y)

    def loss():
        return bce_with_logits(m.logits(X), y)

    for k in m.p:
        flat = m.p[k].ravel()
        for j in [0, flat.size // 2, flat.size - 1]:   # spot-check 3 coords
            orig = flat[j]
            flat[j] = orig + eps
            lp = loss()
            flat[j] = orig - eps
            lm_ = loss()
            flat[j] = orig
            num = (lp - lm_) / (2 * eps)
            ana = grads[k].ravel()[j]
            assert abs(num - ana) < max(1e-3, 0.05 * abs(num)), \
                f"{k}[{j}]: numerical {num:.6f} vs analytic {ana:.6f}"


def test_mlp_learns_separable_problem():
    """Sanity: the trainer can drive loss down on a solvable problem."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=40)
    X = rng.normal(size=(2000, 40)).astype(np.float32)
    y = (X @ w > 0).astype(np.float32)
    m = MLP(sizes=(40, 16, 8, 1), seed=1)
    hist = train(m, X, y, epochs=30, bs=256, verbose=False)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    acc = ((m.probs(X) > 0.5) == (y > 0.5)).mean()
    assert acc > 0.9


def test_save_load_roundtrip(tmp_path):
    m = tiny_model()
    X = RNG.normal(size=(8, 10)).astype(np.float32)
    m.in_mean = X.mean(0)
    m.in_std = np.maximum(X.std(0), 1e-3)
    p = tmp_path / "m.npz"
    m.save(p, extra_meta={"sr": 8000, "context": 7, "hello": "world"})
    m2 = MLP.load(p)
    assert np.allclose(m.probs(X), m2.probs(X))
    assert m2.meta["context"] == 7 and m2.meta["hello"] == "world"


def test_metrics():
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    perfect = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert auc(perfect, y) > 0.99
    prec, rec, f1, _ = prf(perfect, y)
    assert prec == rec == f1 == 1.0
    # anti-correlated → AUC ~0
    assert auc(1 - perfect, y) < 0.01


# ==========================================================================
# streaming.py — hysteresis + end-to-end
# ==========================================================================

def test_hysteresis_basics():
    h = Hysteresis(thr_hi=0.6, thr_lo=0.4, on_frames=3, off_frames=5)
    ev = []
    for i in range(10):
        ev += h.push(0.9)
    assert h.in_speech
    kinds = [e[0] for e in ev]
    assert kinds.count("speech_start") == 1
    assert ev[0] == ("speech_start", 0)               # onset, not 3 frames late
    # hangover: 4 quiet frames < 5 → still speech
    for _ in range(4):
        h.push(0.1)
    assert h.in_speech
    # 5th quiet frame → speech_end, stamped at first quiet frame (idx 10)
    ev2 = h.push(0.1)
    assert ev2 == [("speech_end", 10)]
    assert not h.in_speech


def test_hysteresis_debounce():
    # alternating above/below must never trigger (needs on_frames in a row)
    h = Hysteresis(thr_hi=0.6, thr_lo=0.4, on_frames=3, off_frames=3)
    for _ in range(20):
        h.push(0.9)
        h.push(0.1)
    assert not h.in_speech


def test_hysteresis_events_helper():
    scores = [0.0] * 5 + [0.9] * 5 + [0.0] * 10
    h, ev = hysteresis_events(scores, 0.6, 0.4, 3, 5)
    kinds = [(k, i) for k, i in ev]
    assert ("speech_start", 5) in kinds
    assert any(k == "speech_end" and i == 10 for k, i in kinds)


def _save_tiny_model(tmp_path, context=4):
    """Train a micro-model on synthetic 'speech vs noise' and save it.

    Feature construction mirrors StreamingVAD exactly: full log-mel+delta
    rows, a K-frame window flattened frame-major
    ([old frame dims | ... | current frame dims]).
    """
    lm = LogMel(sr=8000)
    rng = np.random.default_rng(3)
    K = context

    def synth_speech(n, sr=8000):
        out = []
        for _ in range(n):
            f0 = float(rng.uniform(100, 350))         # speech-ish: pitch + harm.
            t = np.arange(lm.frame_len) / sr
            s = sum(np.sin(2 * np.pi * f0 * h * t) * (0.8 ** h) for h in (1, 2, 3))
            out.append(s + 0.05 * rng.normal(size=lm.frame_len))
        return out

    def synth_noise(n):
        out = []
        for _ in range(n):
            x = rng.normal(size=lm.frame_len)
            k = np.exp(-np.arange(lm.frame_len) / 20)  # smooth = low-freq rumble
            out.append(np.convolve(x, k, mode="same") * 0.3)
        return out

    def feat_rows(clips):
        # full features (with deltas), like the streaming path
        return lm(np.concatenate(clips))

    fs = feat_rows(synth_speech(600))
    fn = feat_rows(synth_noise(600))

    def stack(frames, label):
        X, y = [], []
        for i in range(K - 1, len(frames)):
            X.append(frames[i - K + 1: i + 1].ravel())
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xs, ys = stack(fs, 1.0)
    Xn, yn = stack(fn, 0.0)
    X = np.concatenate([Xs, Xn])
    y = np.concatenate([ys, yn])
    m = MLP(sizes=(X.shape[1], 32, 16, 1), seed=5)
    train(m, X, y, epochs=25, bs=128, verbose=False)
    meta = dict(sr=8000, n_mels=20, win_ms=25.0, hop_ms=10.0, n_fft=256,
                fmin=80.0, fmax=3800.0, deltas=True, context=context,
                thr_hi=0.6, thr_lo=0.4, hangover_ms=100.0, arch="test")
    p = tmp_path / "tiny.npz"
    m.save(p, extra_meta=meta)
    return p, lm


def test_streaming_vad_end_to_end(tmp_path):
    model_path, lm = _save_tiny_model(tmp_path)
    vad = StreamingVAD(model_path)
    sr = vad.sr

    # Build a clip: 0.5 s noise → 1 s "speech" (harmonic buzz) → 0.5 s noise
    t = np.arange(sr * 2) / sr
    speech = sum(np.sin(2 * np.pi * 200 * h * t) * (0.7 ** h) for h in (1, 2, 3, 4))
    noise = 0.05 * RNG.normal(size=len(t))
    x = noise.copy()
    x[sr // 2: sr // 2 + sr] += speech[sr // 2: sr // 2 + sr]

    pcm = float_to_pcm16(x)
    events = []
    for i in range(0, len(pcm), 160):                # 20 ms telephony frames
        events += vad.feed(pcm[i:i + 160])
    kinds = [(e.type, round(e.t, 2)) for e in events]

    starts = [t_ for k, t_ in kinds if k == "speech_start"]
    ends = [t_ for k, t_ in kinds if k == "speech_end"]
    assert len(starts) == 1 and len(ends) == 1
    assert 0.4 <= starts[0] <= 0.9, starts           # onset near 0.5 s
    assert 1.3 <= ends[0] <= 1.9, ends               # offset near 1.5 s
    assert vad.speech_seconds > 0.5


def test_streaming_vad_reset(tmp_path):
    model_path, _ = _save_tiny_model(tmp_path)
    vad = StreamingVAD(model_path)
    t = np.arange(8000) / 8000
    buzz = float_to_pcm16(0.5 * np.sin(2 * np.pi * 180 * t))
    ev1 = vad.feed(buzz)
    vad.reset()
    ev2 = vad.feed(buzz)
    assert [(e.type) for e in ev1] == [e.type for e in ev2]


def test_energy_vad_finds_loud_speech():
    ev = EnergyVAD(sr=8000)
    t = np.arange(2 * 8000) / 8000
    speech = 0.4 * np.sin(2 * np.pi * 220 * t)
    x = 0.002 * RNG.normal(size=len(t))              # quiet noise floor
    x[8000 // 2: 8000 + 8000 // 2] += speech[8000 // 2: 8000 + 8000 // 2]
    events = []
    pcm = float_to_pcm16(x.astype(np.float32))
    for i in range(0, len(pcm), 320):
        events += ev.feed(pcm[i:i + 320])
    starts = [e.t for e in events if e.type == "speech_start"]
    assert len(starts) == 1 and 0.3 <= starts[0] <= 0.8


def test_energy_vad_robust_format_handling():
    ev = EnergyVAD(sr=8000)
    # numpy float input must work identically to bytes: noise lead, then voice
    t = np.arange(8000 * 2) / 8000
    x = 0.002 * RNG.normal(size=len(t)).astype(np.float32)
    x[8000:] += (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)[8000:]
    ev.feed(x)
    assert ev.in_speech
