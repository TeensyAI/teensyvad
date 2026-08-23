// Package teensyvad is a pure-Go port of the TeensyVAD inference stack:
// npz model loading, the log-mel(+delta) frontend, the MLP, and the
// streaming VAD state machine. It is a faithful reimplementation of
// teensyvad/{features,model,streaming}.py — same math, same event
// semantics, zero dependencies beyond the Go standard library.
package teensyvad

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// .npz / .npy reading (zip of .npy arrays)
// ---------------------------------------------------------------------------

type array struct {
	shape []int
	f32   []float32 // C-order data for numeric dtypes
	str   string    // payload for unicode string arrays (meta)
}

func (a *array) rows() int { return a.shape[0] }
func (a *array) cols() int {
	if len(a.shape) == 1 {
		return 1
	}
	return a.shape[1]
}

// readNpy parses one .npy payload: magic + version + header + raw data.
func readNpy(data []byte) (*array, error) {
	if len(data) < 10 || !bytes.HasPrefix(data, []byte("\x93NUMPY")) {
		return nil, fmt.Errorf("not an npy file")
	}
	major := data[6]
	var hlen int
	switch major {
	case 1:
		hlen = int(binary.LittleEndian.Uint16(data[8:10]))
	case 2, 3:
		hlen = int(binary.LittleEndian.Uint32(data[8:12]))
	default:
		return nil, fmt.Errorf("unsupported npy version %d", major)
	}
	off := 10
	if major >= 2 {
		off = 12
	}
	header := string(data[off : off+hlen])

	descr, err := headerValue(header, "'descr'")
	if err != nil {
		return nil, err
	}
	descr = strings.Trim(descr, "'\"")
	shapeStr, err := tupleValue(header, "'shape'")
	if err != nil {
		return nil, err
	}
	var shape []int
	for _, p := range strings.Split(strings.Trim(shapeStr, "() "), ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("bad shape %q: %w", shapeStr, err)
		}
		shape = append(shape, n)
	}
	body := data[off+hlen:]

	// Unicode string array (the meta blob): dtype like '<U1234'.
	if strings.HasPrefix(descr, "<U") || strings.HasPrefix(descr, "|U") {
		var sb strings.Builder
		for i := 0; i+3 < len(body); i += 4 {
			c := binary.LittleEndian.Uint32(body[i : i+4])
			if c == 0 {
				break
			}
			sb.WriteRune(rune(c))
		}
		return &array{shape: shape, str: sb.String()}, nil
	}
	if descr != "<f4" && descr != "<f8" {
		return nil, fmt.Errorf("unsupported dtype %q", descr)
	}
	width := 4
	scale := float32(1)
	if descr == "<f8" {
		width = 8
	}
	n := len(body) / width
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		if width == 4 {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4 : i*4+4]))
		} else {
			bits := binary.LittleEndian.Uint64(body[i*8 : i*8+8])
			out[i] = float32(math.Float64frombits(bits)) * scale
		}
	}
	return &array{shape: shape, f32: out}, nil
}

// tupleValue extracts a parenthesized tuple like "'shape': (400, 164),"
func tupleValue(header, key string) (string, error) {
	i := strings.Index(header, key)
	if i < 0 {
		return "", fmt.Errorf("key %s not in npy header", key)
	}
	rest := header[i+len(key):]
	open := strings.Index(rest, "(")
	close := strings.Index(rest, ")")
	if open < 0 || close < open {
		return "", fmt.Errorf("malformed tuple for %s", key)
	}
	return rest[open : close+1], nil
}

// headerValue extracts a python-dict-literal value (quoted string or
// tuple) for key, e.g. "'descr': '<f4'," or "'shape': (400, 164),".
func headerValue(header, key string) (string, error) {
	i := strings.Index(header, key)
	if i < 0 {
		return "", fmt.Errorf("key %s not in npy header", key)
	}
	rest := header[i+len(key):]
	j := strings.Index(rest, ":")
	if j < 0 {
		return "", fmt.Errorf("malformed npy header around %s", key)
	}
	rest = strings.TrimSpace(rest[j+1:])
	end := len(rest)
	if k := strings.Index(rest, ","); k >= 0 {
		end = k
	}
	return strings.TrimSpace(rest[:end]), nil
}

// Model holds the loaded MLP weights and feature configuration.
type Model struct {
	InDim, H1, H2          int
	W1                     []float32 // InDim x H1
	B1                     []float32
	W2                     []float32 // H1 x H2
	B2                     []float32
	W3                     []float32 // H2 x 1
	B3                     float32
	InMean, InStd          []float32
	Meta                   map[string]any
	nMels                  int
	frameLen, hopLen, nFFT int
	sr                     int
	winMs, hopMs           float64
	fmin, fmax             float64
	deltas                 bool
	window                 []float32
	filters                []float32 // nMels x (nFFT/2+1), row-major
	nBins                  int
}

// LoadModel reads a TeensyVAD .npz (float32; quantized npz files carry
// int8 weights and are not supported by the Go reader).
func LoadModel(path string) (*Model, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	arrs := map[string]*array{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(f.Name, ".npy")
		a, err := readNpy(data)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		arrs[name] = a
	}
	metaArr, ok := arrs["meta"]
	if !ok {
		return nil, fmt.Errorf("model has no meta array")
	}
	var meta map[string]any
	if err := json.Unmarshal([]byte(metaArr.str), &meta); err != nil {
		return nil, fmt.Errorf("meta json: %w", err)
	}
	w1 := arrs["p/W1"]
	m := &Model{
		InDim: w1.rows(), H1: w1.cols(),
		H2:     arrs["p/W2"].cols(),
		W1:     w1.f32, B1: arrs["p/b1"].f32,
		W2: arrs["p/W2"].f32, B2: arrs["p/b2"].f32,
		W3: arrs["p/W3"].f32,
		B3: arrs["p/b3"].f32[0],
	}
	if v, ok := arrs["in_mean"]; ok {
		m.InMean = v.f32
	} else {
		m.InMean = make([]float32, m.InDim)
	}
	if v, ok := arrs["in_std"]; ok {
		m.InStd = v.f32
	} else {
		m.InStd = ones(m.InDim)
	}
	m.Meta = meta
	get := func(k string, def float64) float64 {
		if v, ok := meta[k]; ok {
			if f, ok := v.(float64); ok {
				return f
			}
		}
		return def
	}
	m.sr = int(get("sr", 8000))
	m.nMels = int(get("n_mels", 20))
	m.winMs = get("win_ms", 25.0)
	m.hopMs = get("hop_ms", 10.0)
	m.nFFT = int(get("n_fft", 256))
	m.fmin = get("fmin", 80.0)
	m.fmax = get("fmax", 3800.0)
	if m.fmax == 0 {
		m.fmax = math.Min(float64(m.sr)/2-200, 8000)
	}
	m.deltas = true
	if v, ok := meta["deltas"]; ok {
		if b, ok := v.(bool); ok {
			m.deltas = b
		}
	}
	m.frameLen = int(math.Round(m.winMs / 1000 * float64(m.sr)))
	m.hopLen = int(math.Round(m.hopMs / 1000 * float64(m.sr)))
	m.nBins = m.nFFT/2 + 1
	m.buildWindow()
	if err := m.buildFilterbank(); err != nil {
		return nil, err
	}
	return m, nil
}

func ones(n int) []float32 {
	o := make([]float32, n)
	for i := range o {
		o[i] = 1
	}
	return o
}

func (m *Model) buildWindow() {
	m.window = make([]float32, m.frameLen)
	for n := range m.window {
		m.window[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(m.frameLen-1)))
	}
}

// buildFilterbank mirrors features.mel_filterbank exactly.
func (m *Model) buildFilterbank() error {
	nMels := m.nMels
	hzToMel := func(f float64) float64 { return 2595.0 * math.Log10(1.0+f/700.0) }
	melToHz := func(mm float64) float64 { return 700.0 * (math.Pow(10, mm/2595.0) - 1.0) }

	m.filters = make([]float32, nMels*m.nBins)
	melLo, melHi := hzToMel(m.fmin), hzToMel(m.fmax)
	hzPts := make([]float64, nMels+2)
	for i := range hzPts {
		hzPts[i] = melToHz(melLo + (melHi-melLo)*float64(i)/float64(nMels+1))
	}
	sums := make([]float64, nMels)
	for i := 0; i < nMels; i++ {
		left, cen, right := hzPts[i], hzPts[i+1], hzPts[i+2]
		upStep := math.Max(cen-left, 1e-9)
		downStep := math.Max(right-cen, 1e-9)
		for k := 0; k < m.nBins; k++ {
			f := float64(k) * float64(m.sr) / float64(m.nFFT)
			w := math.Min((f-left)/upStep, (right-f)/downStep)
			if w > 0 {
				m.filters[i*m.nBins+k] = float32(w)
				sums[i] += w
			}
		}
	}
	for i := 0; i < nMels; i++ {
		inv := 1.0 / math.Max(sums[i], 1e-12)
		for k := 0; k < m.nBins; k++ {
			m.filters[i*m.nBins+k] *= float32(inv)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// FFT (iterative radix-2, size must be a power of two)
// ---------------------------------------------------------------------------

type fftPlan struct {
	n     int
	rev   []int
	cos   []float32
	sin   []float32
	re, im []float32
}

func newFFT(n int) *fftPlan {
	p := &fftPlan{n: n, rev: make([]int, n)}
	for i := 0; i < n; i++ {
		r, v := 0, i
		for bits := 0; bits < log2(n); bits++ {
			r = r<<1 | v&1
			v >>= 1
		}
		p.rev[i] = r
	}
	half := n / 2
	p.cos = make([]float32, half)
	p.sin = make([]float32, half)
	for i := 0; i < half; i++ {
		p.cos[i] = float32(math.Cos(-2 * math.Pi * float64(i) / float64(n)))
		p.sin[i] = float32(math.Sin(-2 * math.Pi * float64(i) / float64(n)))
	}
	p.re = make([]float32, n)
	p.im = make([]float32, n)
	return p
}

func log2(n int) int {
	l := 0
	for n > 1 {
		n >>= 1
		l++
	}
	return l
}

// powerSpectrum writes |X(k)|^2 for k in [0,n/2] into out.
func (p *fftPlan) powerSpectrum(in []float32, out []float32) {
	re, im := p.re[:p.n], p.im[:p.n]
	for i, r := range p.rev {
		re[i], im[i] = in[r], 0
	}
	for length := 2; length <= p.n; length <<= 1 {
		step := p.n / length
		half := length / 2
		for start := 0; start < p.n; start += length {
			for j := 0; j < half; j++ {
				wIdx := j * step
				wr, wi := p.cos[wIdx], p.sin[wIdx]
				a, b := start+j, start+j+half
				tr := wr*re[b] - wi*im[b]
				ti := wr*im[b] + wi*re[b]
				re[b] = re[a] - tr
				im[b] = im[a] - ti
				re[a] += tr
				im[a] += ti
			}
		}
	}
	for k := 0; k <= p.n/2; k++ {
		out[k] = re[k]*re[k] + im[k]*im[k]
	}
}

// ---------------------------------------------------------------------------
// Streaming VAD
// ---------------------------------------------------------------------------

// VADEvent mirrors streaming.VADEvent: "speech_start" or "speech_end".
type VADEvent struct {
	Type string
	T    float64 // seconds since stream start
}

// StreamingVAD processes PCM16LE bytes (or float32 samples) chunk by chunk.
type StreamingVAD struct {
	model    *Model
	fft      *fftPlan
	ThrLo, ThrHi float64
	onFrames, offFrames int

	buf        []float32 // pending samples
	prevV      []float32 // previous mel frame (for delta); valid when hasPrev
	hasPrev    bool
	hist       []float32 // K frames x featDim context history
	K, featDim int

	inSpeech      bool
	onStreak, offStreak int
	onStartFrame, offStartFrame int
	frameCount    int
	SpeechSeconds float64
	ProbHistory   []float32
	LastProb      float32

	// scratch
	winFrame []float32
	power    []float32
	melV     []float32
	x        []float32
	h1, h2   []float32
	acc1     []float32 // dense-layer accumulators (H1, H2)
	acc2     []float32
}

// NewStreamingVAD loads path and returns a ready VAD. Thresholds come from
// the model metadata unless overridden.
func NewStreamingVAD(path string) (*StreamingVAD, error) {
	m, err := LoadModel(path)
	if err != nil {
		return nil, err
	}
	metaF := func(k string, def float64) float64 {
		if v, ok := m.Meta[k]; ok {
			if f, ok := v.(float64); ok {
				return f
			}
		}
		return def
	}
	v := &StreamingVAD{
		model: m,
		fft:   newFFT(m.nFFT),
		ThrHi: metaF("thr_hi", 0.60),
		ThrLo: metaF("thr_lo", 0.40),
		onFrames:  3,
		offFrames: int(math.Round(metaF("hangover_ms", 250.0) / 1000 / (metaF("hop_ms", 10.0)/1000))),
		K:         int(metaF("context", 10)),
		featDim:   m.nMels * 2,
		buf:       make([]float32, 0, m.frameLen*4),
	}
	v.power = make([]float32, m.nBins)
	v.melV = make([]float32, m.nMels)
	v.winFrame = make([]float32, m.nFFT)
	v.x = make([]float32, v.K*v.featDim)
	v.h1 = make([]float32, m.H1)
	v.h2 = make([]float32, m.H2)
	v.acc1 = make([]float32, m.H1)
	v.acc2 = make([]float32, m.H2)
	v.prevV = make([]float32, m.nMels)
	v.hist = make([]float32, v.K*v.featDim) // zeros = silence before start
	return v, nil
}

// Reset starts a new stream (new call).
func (v *StreamingVAD) Reset() {
	v.buf = v.buf[:0]
	v.hasPrev = false
	clear(v.hist)
	v.inSpeech = false
	v.onStreak, v.offStreak = 0, 0
	v.frameCount = 0
	v.SpeechSeconds = 0
	v.ProbHistory = nil
	v.LastProb = 0
}

// FeedBytes feeds one chunk of PCM16 little-endian audio.
func (v *StreamingVAD) FeedBytes(chunk []byte) []VADEvent {
	n := len(chunk) / 2
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(chunk[i*2:]))) / 32768
	}
	return v.Feed(samples)
}

// Feed feeds float samples in [-1, 1) and returns events triggered here.
func (v *StreamingVAD) Feed(samples []float32) []VADEvent {
	m := v.model
	v.buf = append(v.buf, samples...)
	var events []VADEvent
	for len(v.buf) >= m.frameLen {
		p := v.inferFrame(v.buf[:m.frameLen])
		v.ProbHistory = append(v.ProbHistory, p)
		v.LastProb = p
		for _, ev := range v.pushHysteresis(float64(p)) {
			events = append(events, ev)
		}
		if v.inSpeech {
			v.SpeechSeconds += float64(m.hopLen) / float64(m.sr)
		}
		v.buf = v.buf[m.hopLen:]
	}
	return events
}

// Flush closes any open speech segment at the current position.
func (v *StreamingVAD) Flush() []VADEvent {
	if !v.inSpeech {
		return nil
	}
	t := float64(v.frameCount) * float64(v.model.hopLen) / float64(v.model.sr)
	v.inSpeech = false
	v.offStreak = 0
	return []VADEvent{{Type: "speech_end", T: t}}
}

func (v *StreamingVAD) InSpeech() bool { return v.inSpeech }

// pushHysteresis mirrors streaming.Hysteresis.push.
func (v *StreamingVAD) pushHysteresis(score float64) []VADEvent {
	var events []VADEvent
	i := v.frameCount
	if !v.inSpeech {
		if score >= v.ThrHi {
			if v.onStreak == 0 {
				v.onStartFrame = i
			}
			v.onStreak++
			if v.onStreak >= v.onFrames {
				v.inSpeech = true
				v.offStreak = 0
				events = append(events, VADEvent{"speech_start", float64(v.onStartFrame) * v.hopSeconds()})
			}
		} else {
			v.onStreak = 0
		}
	} else {
		if score < v.ThrLo {
			if v.offStreak == 0 {
				v.offStartFrame = i
			}
			v.offStreak++
			if v.offStreak >= v.offFrames {
				v.inSpeech = false
				v.onStreak = 0
				events = append(events, VADEvent{"speech_end", float64(v.offStartFrame) * v.hopSeconds()})
			}
		} else {
			v.offStreak = 0
		}
	}
	v.frameCount++
	return events
}

func (v *StreamingVAD) hopSeconds() float64 {
	return float64(v.model.hopLen) / float64(v.model.sr)
}

// inferFrame runs one full frame pipeline: window → FFT → log-mel → delta →
// context stack → normalize → MLP → sigmoid.
func (v *StreamingVAD) inferFrame(frame []float32) float32 {
	m := v.model

	// window into the padded fft buffer (zero-padded to nFFT)
	wf := v.winFrame
	for i := range wf {
		wf[i] = 0
	}
	for i, s := range frame {
		wf[i] = s * m.window[i]
	}
	v.fft.powerSpectrum(wf, v.power)

	// mel band energies → log → per-frame mean subtraction (in that order:
	// the gain-invariance shift is a constant in LOG space)
	for b := 0; b < m.nMels; b++ {
		row := m.filters[b*m.nBins:]
		var acc float32
		for k, pw := range v.power {
			acc += row[k] * pw
		}
		v.melV[b] = float32(math.Log(float64(acc) + 1e-10))
	}
	var sum float32
	for _, lv := range v.melV {
		sum += lv
	}
	mean := sum / float32(m.nMels)
	for b := range v.melV {
		v.melV[b] -= mean
	}

	// shift context history by one frame, append [v | delta]
	copy(v.hist, v.hist[v.featDim:])
	off := (v.K - 1) * v.featDim
	copy(v.hist[off:off+m.nMels], v.melV)
	if !v.hasPrev {
		clear(v.hist[off+m.nMels : off+v.featDim])
	} else {
		for b := 0; b < m.nMels; b++ {
			v.hist[off+m.nMels+b] = v.melV[b] - v.prevV[b]
		}
	}
	copy(v.prevV, v.melV)
	v.hasPrev = true

	// input vector is the whole context history, frame-major
	copy(v.x, v.hist)

	// normalize + MLP forward
	//
	// Dense layers run as row-sequential accumulations: for each input
	// dim, scale its weight ROW and add into the accumulator. This keeps
	// both the weights and the accumulator streaming through cache in
	// order (the transposed dot-product form strides H1 floats between
	// MACs and runs several times slower in Go).
	h1, h2, x := v.h1, v.h2, v.x
	for i := 0; i < m.InDim; i++ {
		x[i] = (x[i] - m.InMean[i]) / m.InStd[i]
	}
	clear(v.acc1)
	clear(v.acc2)
	base := 0
	for i := 0; i < m.InDim; i++ {
		xi := x[i]
		if xi != 0 {
			row := m.W1[base : base+m.H1]
			j := 0
			for ; j+4 <= m.H1; j += 4 {
				v.acc1[j] += xi * row[j]
				v.acc1[j+1] += xi * row[j+1]
				v.acc1[j+2] += xi * row[j+2]
				v.acc1[j+3] += xi * row[j+3]
			}
			for ; j < m.H1; j++ {
				v.acc1[j] += xi * row[j]
			}
		}
		base += m.H1
	}
	for j := 0; j < m.H1; j++ {
		a := v.acc1[j] + m.B1[j]
		if a < 0 {
			a = 0
		}
		h1[j] = a
	}
	base = 0
	for i := 0; i < m.H1; i++ {
		xi := h1[i]
		if xi != 0 {
			row := m.W2[base : base+m.H2]
			j := 0
			for ; j+4 <= m.H2; j += 4 {
				v.acc2[j] += xi * row[j]
				v.acc2[j+1] += xi * row[j+1]
				v.acc2[j+2] += xi * row[j+2]
				v.acc2[j+3] += xi * row[j+3]
			}
			for ; j < m.H2; j++ {
				v.acc2[j] += xi * row[j]
			}
		}
		base += m.H2
	}
	var z float32
	for j := 0; j < m.H2; j++ {
		a := v.acc2[j] + m.B2[j]
		if a < 0 {
			a = 0
		}
		h2[j] = a
		z += a * m.W3[j]
	}
	z += m.B3
	return sigmoid(z)
}

func sigmoid(z float32) float32 {
	if z >= 0 {
		return 1 / (1 + float32(math.Exp(float64(-z))))
	}
	ez := float32(math.Exp(float64(z)))
	return ez / (1 + ez)
}
