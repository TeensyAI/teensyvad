# teensyvad-go

A pure-Go port of the TeensyVAD inference stack — standard library only,
single static binary, no Python/NumPy/ONNX runtime. Faithful
reimplementation of `teensyvad/{features,model,streaming}.py`: same log-mel
frontend, same MLP, same hysteresis semantics.

```go
import vad "github.com/TeensyAI/teensyvad/go"

v, _ := vad.NewStreamingVAD("teensy-v4-80k.npz")
for len(chunk) > 0 {          // PCM16LE bytes, any size (e.g. 320 B = 20 ms)
    for _, ev := range v.FeedBytes(chunk) {
        // ev.Type: "speech_start" | "speech_end", ev.T seconds
    }
}
```

## Parity (verified, not claimed)

`go test` replays a golden clip through both stacks and asserts frame-level
equality against the reference Python implementation:

```
frames=1198  max|dp|=1.91e-06  mean|dp|=2.72e-07  events identical
```

The residual is float32-vs-float64 accumulation noise. Regenerate fixtures
with `.venv/bin/python scripts/gen_golden.py`.

## Benchmark

Same machine (Apple M2 Pro), same 60 s synthetic stream, same 20 ms-chunk
pattern (`go/cmd/bench` vs `scripts/bench_python.py`):

| stack                          | µs / 10 ms hop | RTF      | notes |
|---|---|---|---|
| Python + NumPy (BLAS batched)  | ~32            | ~310×    | reference |
| **Go (this port)**             | **~40**        | **~250×**| pure stdlib, zero deps |
| ONNX Runtime (model only)      | ~8             | —        | + Python frontend cost |

Honest reading: Go is ~1.25× *slower* than NumPy here — not because of the
language but because BLAS dispatches NEON/AMX SIMD while Go's compiler emits
scalar fp32 MACs (~2.2 GMAC/s ceiling for this shape). The dense layer is
~95% of the frame budget (the radix-2 FFT costs 1.8 µs). Getting below NumPy
in pure Go would need hand-written ARM64 NEON assembly for the two GEMM
loops; getting below ONNX needs cgo, which defeats the purpose.

Why Go anyway: one static binary for telephony boxes, goroutine-native for
thousands of concurrent channels, no interpreter, no dependency supply chain.
At 250× real-time a single core absorbs ~250 simultaneous calls with 99.6%
headroom — speed stopped being the deciding factor long ago.

## Layout

- `teensyvad.go` — npz/npy reader, mel filterbank, FFT, MLP, streaming VAD
- `teensyvad_test.go` — golden parity against the Python reference
- `cmd/bench` — streaming benchmark (`go run ./cmd/bench -secs 60`)
