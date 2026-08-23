package teensyvad

import (
	"os"
	"testing"
)

func BenchmarkMLPOnly(b *testing.B) {
	v, _ := NewStreamingVAD("../models/teensy-v4-80k.npz")
	m := v.model
	for i := range v.x {
		v.x[i] = float32(i%7) * 0.01
	}
	x := make([]float32, m.InDim)
	h1 := make([]float32, m.H1)
	h2 := make([]float32, m.H2)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := 0; i < m.InDim; i++ { x[i] = v.x[i] }
		for j := 0; j < m.H1; j++ {
			var acc float32
			col := j
			for i := 0; i < m.InDim; i++ { acc += x[i]*m.W1[col]; col += m.H1 }
			if acc < 0 { acc = 0 }
			h1[j] = acc
		}
		for j := 0; j < m.H2; j++ {
			var acc float32
			col := j
			for i := 0; i < m.H1; i++ { acc += h1[i]*m.W2[col]; col += m.H2 }
			if acc < 0 { acc = 0 }
			h2[j] = acc
		}
	}
}

func BenchmarkFFTOnly(b *testing.B) {
	v, _ := NewStreamingVAD("../models/teensy-v4-80k.npz")
	wf := make([]float32, 256)
	pw := make([]float32, 129)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		v.fft.powerSpectrum(wf, pw)
	}
}

var sink float32

func BenchmarkFullFrame(b *testing.B) {
	v, err := NewStreamingVAD("../models/teensy-v4-80k.npz")
	if err != nil { b.Fatal(err) }
	raw, _ := os.ReadFile("testdata/input.raw")
	samples := make([]float32, len(raw)/2)
	frame := samples[:200]
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		sink = v.inferFrame(frame)
	}
}
