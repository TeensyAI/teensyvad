// Command bench streams a synthetic 60 s call through the Go VAD in
// 20 ms chunks (the Asterisk pattern) and reports per-hop cost, RTF,
// and detected events. Compare against scripts/bench_python.py.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	vad "github.com/TeensyAI/teensyvad/go"
)

func synthPCM16(seconds int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	sr := 8000
	n := sr * seconds
	pcm := make([]byte, n*2)
	phase := 0.0
	for i := 0; i < n; i++ {
		t := float64(i) / float64(sr)
		sample := 0.01 * rng.NormFloat64()
		// speech bursts: 1s on, 0.6s off pattern with formant-ish tones
		if burst := math.Mod(t, 1.6); burst < 1.0 {
			f0 := 130 + 30*math.Sin(2*math.Pi*1.2*t)
			phase += 2 * math.Pi * f0 / float64(sr)
			sample += 0.35 * math.Sin(phase)
			sample += 0.12 * math.Sin(3*phase)
			sample += 0.06 * math.Sin(5*phase)
		}
		bits := uint16(uint16(int16(sample * 8000)))
		binary.LittleEndian.PutUint16(pcm[i*2:], bits)
	}
	return pcm
}

func main() {
	modelPath := flag.String("model", "../models/teensy-v4-80k.npz", "npz model path")
	seconds := flag.Int("secs", 60, "seconds of audio to stream")
	flag.Parse()

	v, err := vad.NewStreamingVAD(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	pcm := synthPCM16(*seconds, 42)

	const chunk = 320 // bytes = 160 samples = 20 ms
	start := time.Now()
	events := 0
	for i := 0; i+chunk <= len(pcm); i += chunk {
		events += len(v.FeedBytes(pcm[i : i+chunk]))
	}
	events += len(v.Flush())
	elapsed := time.Since(start)

	hops := float64(len(pcm)/2) / 80 // 80-sample hop at 8 kHz
	audioDur := float64(*seconds)
	fmt.Printf("model        : %s\n", *modelPath)
	fmt.Printf("audio        : %d s streamed in 20 ms chunks\n", *seconds)
	fmt.Printf("total time   : %v\n", elapsed)
	fmt.Printf("per hop      : %.3f µs / 10 ms frame\n", float64(elapsed.Nanoseconds())/1000/hops)
	fmt.Printf("per 20 ms op : %.3f µs\n", float64(elapsed.Nanoseconds())/1000/(hops/2))
	fmt.Printf("RTF          : %.6f (%.0fx real-time on one core)\n",
		elapsed.Seconds()/audioDur, audioDur/elapsed.Seconds())
	fmt.Printf("events       : %d, speech %.2f s\n", events, v.SpeechSeconds)
}
