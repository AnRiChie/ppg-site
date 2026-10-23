// Streaming port of full_refine.py: POS -> filter_pipeline(fft=False) -> SNR windows v2 -> HR (fine DFT, as CZT).
// Pure functions + PpgMeter; no DOM, so it can be checked on recorded sessions (see check.html).

const FS = 30;            // resampling rate, Hz (camera rate in the project)
const WIN_SEC = 3.0;      // SNR window, as compute_windows_v2
const STEP_SEC = 0.5;     // SNR window step
const SNR_THR = 0.6;      // quality threshold, as full_refine
const MIN_RUN_SEC = 30;   // continuous good signal needed for HR (calculation interval)
const MAX_GAP_SEC = 1.5;  // bad gap allowed inside a run
const BUF_SEC = 10;       // history the filters see per call; also the plot length
const BLOCK_SEC = 30;     // clipping bounds are refreshed from the last 30 s every 30 s
const FAIL_SEC = 30;      // give up after this long without any good signal

function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

function std(a) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / a.length);
}

// numpy.percentile with linear interpolation
function percentile(a, p) {
  const s = Float64Array.from(a).sort();
  const pos = (s.length - 1) * p / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// full_refine.pos_projection
function posProjection(r, g, b) {
  const n = g.length;
  let win = Math.floor(FS * 1.5), slide = Math.floor(FS * 0.5);
  if (win >= n) { win = Math.max(1, n - 1); slide = 1; }
  const out = new Float64Array(n), cnt = new Float64Array(n);
  const starts = [];
  for (let i = 0; i + win <= n; i += Math.max(1, slide)) starts.push(i);
  // extra window at the end, otherwise the newest samples stay uncovered (zero) in a live stream
  if (starts.length && starts[starts.length - 1] + win < n) starts.push(n - win);
  for (const i of starts) {
    const mr = mean(r.subarray(i, i + win)) + 1e-8;
    const mg = mean(g.subarray(i, i + win)) + 1e-8;
    const mb = mean(b.subarray(i, i + win)) + 1e-8;
    const x = new Float64Array(win), y = new Float64Array(win);
    for (let k = 0; k < win; k++) {
      const rn = r[i + k] / mr, gn = g[i + k] / mg, bn = b[i + k] / mb;
      x[k] = gn - bn;
      y[k] = -2 * rn + gn + bn;
    }
    const alpha = std(x) / (std(y) + 1e-8);
    const s = x.map((v, k) => v + alpha * y[k]);
    const ms = mean(s);
    for (let k = 0; k < win; k++) { out[i + k] += s[k] - ms; cnt[i + k] += 1; }
  }
  for (let i = 0; i < n; i++) out[i] /= cnt[i] || 1;
  return out;
}

// preprocess_data.filter_channel is a plain clip to percentile bounds: the "longest stable segment"
// it spares lies inside the bounds by definition. Split into bounds + clip so bounds can be kept for 30 s.
function bounds(x, low, up) {
  return [percentile(x, low), percentile(x, up)];
}

function clip(x, [lo, hi]) {
  return x.map(v => Math.min(hi, Math.max(lo, v)));
}

// preprocess_data.detrend_filter (window 1 s, slide 0.5 s)
function detrend(x) {
  const win = FS, slide = FS / 2;
  const out = new Float64Array(x.length), cnt = new Float64Array(x.length);
  for (let s = 0; s < x.length; s += slide) {
    const e = Math.min(x.length, s + win);
    const m = mean(x.subarray(s, e));
    for (let i = s; i < e; i++) { out[i] += x[i] - m; cnt[i] += 1; }
  }
  for (let i = 0; i < x.length; i++) out[i] /= cnt[i] || 1;
  return out;
}

// DFT amplitude of x (mean m removed) at frequency f, Hz
function ampAt(x, m, f) {
  const w = 2 * Math.PI * f / FS;
  let re = 0, im = 0;
  for (let i = 0; i < x.length; i++) { re += (x[i] - m) * Math.cos(w * i); im -= (x[i] - m) * Math.sin(w * i); }
  return Math.hypot(re, im);
}

// preprocess_data.calc_snr_and_freq: spectral peak vs the rest of the bins in 0.5-3.5 Hz.
// The 3 s bin grid is 20 BPM wide, so there a clean pulse scores 2.5 at 60/80/100 BPM but 0.4 at 70/90 BPM.
// Here the grid is shifted to pass through the true peak (found on a 0.01 Hz grid): same ratio, no HR bias.
function snrAndFreq(chunk) {
  const n = chunk.length;
  if (n < FS) return [0, 1];
  const m = mean(chunk);
  if (Math.max(...chunk.map(v => Math.abs(v - m))) < 1e-6) return [0, 1];
  let fp = 0.5, ap = -1;
  for (let f = 0.5; f <= 3.5; f += 0.01) {
    const a = ampAt(chunk, m, f);
    if (a > ap) { ap = a; fp = f; }
  }
  const df = FS / n;
  let others = 0;
  for (let f = fp - df * Math.floor((fp - 0.5) / df); f <= 3.5 + 1e-9; f += df) {
    if (Math.abs(f - fp) > 1e-9) others += ampAt(chunk, m, f);
  }
  return [ap / (others + 1e-8), fp];
}

// preprocess_data.compute_hr_czt: power spectrum on a fine grid 0.75-3.5 Hz (10x points), peak -> BPM
function hrCzt(x, M = x.length * 10) {
  const n = x.length, m = mean(x), f0 = 0.75, f1 = 3.5;
  let bestP = -1, bestF = f0;
  for (let k = 0; k < M; k++) {
    const f = f0 + (f1 - f0) * k / M;
    const w = 2 * Math.PI * f / FS;
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) { re += (x[i] - m) * Math.cos(w * i); im -= (x[i] - m) * Math.sin(w * i); }
    const p = re * re + im * im;
    if (p > bestP) { bestP = p; bestF = f; }
  }
  return bestF * 60;
}

// full_refine step 5: Gaussian with sigma = pulse period / 6, here at 30 Hz
// (full_refine first repeats each sample 15x, which only makes the curve denser), edges held
function gaussSmooth(x, sigma) {
  const r = Math.ceil(3 * sigma), k = [];
  for (let j = -r; j <= r; j++) k.push(Math.exp(-j * j / (2 * sigma * sigma)));
  const ks = k.reduce((a, v) => a + v, 0);
  return x.map((_, i) => {
    let s = 0;
    for (let j = -r; j <= r; j++) s += k[j + r] * x[Math.min(x.length - 1, Math.max(0, i + j))];
    return s / ks;
  });
}

// full_refine step 5 peaks: Gaussian (sigma = period / 6), find_peaks(distance = 0.6 period, prominence = 0.25 std).
// Peak times are refined with a parabola through 3 samples instead of full_refine's 15x upsampling.
function findPeaks(x, hr) {
  const period = FS * 60 / hr, sm = gaussSmooth(x, period / 6), minProm = 0.25 * std(sm), dist = 0.6 * period;
  const peaks = [];
  for (let i = 1; i < sm.length - 1; i++) {
    if (!(sm[i] > sm[i - 1] && sm[i] >= sm[i + 1])) continue;
    // prominence over one period on each side: height above the higher of the two side minima
    let lo = Infinity, hi = Infinity;
    for (let k = Math.max(0, Math.round(i - period)); k < i; k++) lo = Math.min(lo, sm[k]);
    for (let k = i + 1; k <= Math.min(sm.length - 1, Math.round(i + period)); k++) hi = Math.min(hi, sm[k]);
    if (sm[i] - Math.max(lo, hi) < minProm) continue;
    const den = sm[i - 1] - 2 * sm[i] + sm[i + 1];
    const t = i + (den ? 0.5 * (sm[i - 1] - sm[i + 1]) / den : 0);
    const last = peaks[peaks.length - 1];
    if (last && t - last.t < dist) { if (sm[i] > last.v) peaks[peaks.length - 1] = { t, v: sm[i] }; }
    else peaks.push({ t, v: sm[i] });
  }
  return peaks.map(p => p.t);
}

// analysis.compute_hrv_metrics: SDNN (ms, ddof=1), CV (%), RMSSD (ms), AMo (share of intervals in the modal 50 ms bin)
function hrv(x, hr) {
  const peaks = findPeaks(x, hr);
  if (peaks.length < 3) return null;
  const rr = peaks.slice(1).map((t, k) => (t - peaks[k]) / FS * 1000);
  const m = mean(rr), sdnn = Math.sqrt(rr.reduce((a, v) => a + (v - m) ** 2, 0) / (rr.length - 1));
  const lo = Math.min(...rr) - 25, counts = {};
  for (const v of rr) { const b = Math.floor((v - lo) / 50); counts[b] = (counts[b] || 0) + 1; }
  const rmssd = Math.sqrt(mean(rr.slice(1).map((v, k) => (v - rr[k]) ** 2)));
  return { sdnn, cv: sdnn / m * 100, rmssd, amo: Math.max(...Object.values(counts)) / rr.length, beats: peaks.length };
}

class PpgMeter {
  constructor() {
    // room for a run that is still green at the timeout to reach MIN_RUN_SEC
    // there is no time limit any more, so the arrays grow (see grow) from 60 s
    const max = 60 * FS;
    this.r = new Float64Array(max); this.g = new Float64Array(max); this.b = new Float64Array(max);
    this.q = new Float64Array(max).fill(NaN);  // SNR per sample, NaN until its 0.5 s step is computed
    // merge_v2_snr_profile: each sample's SNR is the Hann-weighted mean of all 3 s windows covering it
    this.num = new Float64Array(max); this.den = new Float64Array(max);
    this.runStart = -1; // first sample of the current green run, -1 when there is none
    this.boundsAt = 0;  // n at the last clip-bounds refresh
    this.hann = new Float64Array(WIN_SEC * FS).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 1) / (WIN_SEC * FS + 1)));
    this.n = 0;
    this.prev = null;   // last raw frame [t, r, g, b] for resampling
    this.t0 = null;
    this.b1 = null; this.b2 = null;  // clip bounds: p20-p80 of POS, p3-p97 after detrend
    this.freq = null;   // pulse frequency of the last good window, Hz (for the Gaussian on the plot)
    this.thr = SNR_THR; this.minRun = MIN_RUN_SEC;  // fields so check.html can sweep them
    this.run = 0;       // green time in the current run (gaps not counted), s
    this.shown = 0;     // run, but never going back while the run lasts (provisional SNR gets revised)
    this.green = false; // the newest 0.5 s step is good
    this.liveHr = null; // HR over the current, still growing run (expanding window, as compute_expanding_hr_curve)
    this.hr = null;
    this.hrv = null;    // { sdnn, cv, rmssd, amo, beats } over the final run, null if too few beats
    this.badSec = 0;    // how long there has been no good signal, s
    this.failed = false;
  }

  // Raw frame at time t (s); frames come at an uneven rate, so resample to FS by linear interpolation.
  push(t, r, g, b) {
    if (this.hr !== null || this.failed) return;
    if (this.prev === null) { this.prev = [t, r, g, b]; this.t0 = t; return; }
    const [tp, rp, gp, bp] = this.prev;
    while (!this.failed && this.hr === null) {
      const ts = this.t0 + this.n / FS;
      if (ts > t) break;
      const a = t > tp ? Math.max(0, (ts - tp) / (t - tp)) : 1;
      this.addSample(rp + a * (r - rp), gp + a * (g - gp), bp + a * (b - bp));
    }
    this.prev = [t, r, g, b];
  }

  // doubles every per-sample array, keeping the data
  grow() {
    for (const k of ['r', 'g', 'b', 'q', 'num', 'den']) {
      const a = new Float64Array(this[k].length * 2);
      if (k === 'q') a.fill(NaN);
      a.set(this[k]);
      this[k] = a;
    }
  }

  addSample(r, g, b) {
    if (this.n === this.r.length) this.grow();
    const i = this.n++;
    this.r[i] = r; this.g[i] = g; this.b[i] = b;
    const win = WIN_SEC * FS, step = STEP_SEC * FS, block = BLOCK_SEC * FS;
    if (this.n < win || this.n % step !== 0) return;
    // Once a green run is going, everything before it is ignored: bounds and filters see only the run
    // (rounded up to the 0.5 s grid; at least the 3 s SNR window). Without a run: the recent history.
    const from = this.runStart < 0 ? 0 : Math.min(Math.ceil(this.runStart / step) * step, this.n - win);
    // Bounds follow the data every step until 30 s are collected, then once per 30 s
    if (this.n - from < block || this.n - this.boundsAt >= block) this.refreshBounds(from);
    // Every 0.5 s: SNR of the last 3 s, merged into all its samples as in full_refine (single windows flicker)
    const base = this.filtered(Math.max(from, this.n - BUF_SEC * FS), this.n);
    const [snr, freq] = snrAndFreq(base.subarray(base.length - win));
    for (let k = 0, i0 = this.n - win; k < win; k++) {
      this.num[i0 + k] += snr * this.hann[k];
      this.den[i0 + k] += this.hann[k];
      this.q[i0 + k] = this.num[i0 + k] / this.den[i0 + k];
    }
    if (snr >= this.thr || this.freq === null) this.freq = freq;
    this.updateRun();
  }

  refreshBounds(from) {
    this.boundsAt = this.n;
    const s = Math.max(from, this.n - BLOCK_SEC * FS);
    const pos = posProjection(this.r.subarray(s, this.n), this.g.subarray(s, this.n), this.b.subarray(s, this.n));
    this.b1 = bounds(pos, 20, 80);
    this.b2 = bounds(detrend(clip(pos, this.b1)), 3, 97);
  }

  // POS + filter_pipeline(fft=False) over samples [s, e) with the current 30 s bounds
  filtered(s, e) {
    s = Math.max(0, s);
    s -= s % (STEP_SEC * FS);  // keep POS/detrend window grids fixed so the drawn history doesn't wobble
    return clip(detrend(clip(posProjection(this.r.subarray(s, e), this.g.subarray(s, e), this.b.subarray(s, e)), this.b1)), this.b2);
  }

  quality(i) {
    return this.q[i];
  }

  // Current run of good samples where bad gaps <= MAX_GAP_SEC; HR once its green time reaches MIN_RUN_SEC
  updateRun() {
    let start = -1, lastGood = -1, good = 0;
    for (let i = 0; i < this.n; i++) {
      if (this.q[i] >= this.thr) {
        if (start < 0 || i - lastGood - 1 > MAX_GAP_SEC * FS) { start = i; good = 0; }
        lastGood = i;
        good++;
      }
    }
    this.badSec = (this.n - 1 - lastGood) / FS;  // time since the last good sample (or since the start)
    const alive = lastGood >= 0 && this.n - 1 - lastGood <= MAX_GAP_SEC * FS;
    this.run = alive ? good / FS : 0;
    this.runStart = alive ? start : -1;
    if (this.badSec >= FAIL_SEC) this.failed = true;
    // a reset needs a gap > 1.5 s, i.e. several steps with run = 0, so a new run always starts from 0
    this.shown = this.run === 0 ? 0 : Math.max(this.shown, this.run);
    this.green = this.q[this.n - 1] >= this.thr;
    if (!alive || (lastGood + 1 - start) / FS < WIN_SEC) { this.liveHr = null; return; }
    const seg = this.filtered(start, lastGood + 1);
    if (this.shown >= this.minRun) { this.hr = hrCzt(seg); this.hrv = hrv(seg, this.hr); }
    else this.liveHr = hrCzt(seg, 550);  // 0.005 Hz grid: cheap enough for every 0.5 s on a phone
  }

  // Processed signal of the last BUF_SEC and its Gaussian-smoothed version scaled to the same spread (as full_refine)
  plotSignals() {
    const sig = this.filtered(this.n - BUF_SEC * FS, this.n);
    const hz = this.hr !== null ? this.hr / 60 : (this.freq || 1.25);
    const sm = gaussSmooth(sig, FS / hz / 6);
    const k = std(sig) / (std(sm) || 1);
    return [sig, sm.map(v => v * k)];
  }
}

if (typeof module !== 'undefined') module.exports = { PpgMeter, FS, SNR_THR, MIN_RUN_SEC, FAIL_SEC, BUF_SEC };
