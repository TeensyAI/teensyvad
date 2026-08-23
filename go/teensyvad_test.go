package teensyvad

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestGoldenParity asserts the Go port reproduces the reference Python
// stack frame-for-frame on a fixed clip: max |Δp| < 1e-3 and identical
// speech events.
func TestGoldenParity(t *testing.T) {
	if _, err := os.Stat("../models/teensy-v4-80k.npz"); err != nil {
		t.Skip("run from go/ with repo models/ present; or regenerate goldens: .venv/bin/python scripts/gen_golden.py")
	}
	raw, err := os.ReadFile("testdata/input.raw")
	if err != nil {
		t.Fatalf("missing golden input (run scripts/gen_golden.py): %v", err)
	}
	wantProbs, err := readNpyFile("testdata/probs.npy")
	if err != nil {
		t.Fatalf("missing golden probs: %v", err)
	}
	wantEventsJSON, _ := os.ReadFile("testdata/events.json")
	var wantEvents []struct {
		Type string  `json:"type"`
		T    float64 `json:"t"`
	}
	if err := json.Unmarshal(wantEventsJSON, &wantEvents); err != nil {
		t.Fatalf("bad events.json: %v", err)
	}

	v, err := NewStreamingVAD("../models/teensy-v4-80k.npz")
	if err != nil {
		t.Fatal(err)
	}
	const chunk = 320 // bytes = 160 samples = 20 ms
	var events []VADEvent
	for i := 0; i+chunk <= len(raw); i += chunk {
		events = append(events, v.FeedBytes(raw[i:i+chunk])...)
	}
	events = append(events, v.Flush()...)

	if len(events) != len(wantEvents) {
		t.Fatalf("event count mismatch: got %v want %v", events, wantEvents)
	}
	for i, ev := range events {
		if ev.Type != wantEvents[i].Type || math.Abs(ev.T-wantEvents[i].T) > 1e-6 {
			t.Errorf("event %d: got (%s %.3f) want (%s %.3f)", i, ev.Type, ev.T, wantEvents[i].Type, wantEvents[i].T)
		}
	}
	maxDelta, meanDelta := 0.0, 0.0
	for i := range v.ProbHistory {
		d := math.Abs(float64(v.ProbHistory[i] - wantProbs.f32[i]))
		if d > maxDelta {
			maxDelta = d
		}
		meanDelta += d
	}
	meanDelta /= float64(len(v.ProbHistory))
	t.Logf("frames=%d max|dp|=%.2e mean|dp|=%.2e events=%d",
		len(v.ProbHistory), maxDelta, meanDelta, len(events))
	if maxDelta > 1e-3 {
		t.Errorf("probability divergence too high: max|dp|=%g", maxDelta)
	}
}

func readNpyFile(path string) (*array, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return readNpy(data)
}

var _ = binary.LittleEndian // keep import if unused in future edits
