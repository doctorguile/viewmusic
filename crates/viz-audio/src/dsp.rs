//! Pure DSP analyzer — no Core Audio dependency.
//!
//! [`Analyzer`] turns a stream of mono f32 samples (at [`viz_core::SAMPLE_RATE`])
//! into [`FeatureFrame`]s on every hop. It is deliberately hardware-free so the
//! synthetic-signal tests in `tests/synthetic.rs` exercise the exact same code
//! the audio thread runs (so the analyzer is fully deterministic and testable).
//!
//! Pipeline per hop (10.67 ms at 48 kHz):
//! * sliding window N = 1024, Hann-windowed, FFT via `realfft` with all buffers
//!   pre-allocated and `process_with_scratch` (zero per-hop heap allocation);
//! * 513 power bins folded into 48 log-spaced bands
//!   (`f_k = 30 * (16000/30)^(k/47)`), with per-band AGC + asymmetric smoothing
//!   into `bands[48] ∈ 0..1`;
//! * low/mid/high aggregates from the same powers;
//! * RMS-over-hop fed through an EWMA into `energy ∈ 0..1`;
//! * SuperFlux onset detection ([`crate::onset`]) → [`BeatEvent`]s + the
//!   decaying `beat` pulse;
//! * a monotonic sample counter drives the audio clock `t = samples / 48 000`
//!   and a sustained-silence flag.

use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;
use viz_core::{BeatEvent, FeatureFrame, BAND_COUNT, SAMPLE_RATE, WAVEFORM_LEN};

use crate::onset::Onset;

/// FFT / analysis window length.
pub const FFT_SIZE: usize = 1024;
/// Hop size between consecutive windows.
pub const HOP_SIZE: usize = 512;
/// Number of magnitude bins produced by a real FFT of [`FFT_SIZE`].
pub const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1; // 513

/// Lowest band edge (Hz).
const BAND_LO_HZ: f64 = 30.0;
/// Highest band edge (Hz).
const BAND_HI_HZ: f64 = 16_000.0;

/// Aggregate cutoffs (Hz).
const LOW_HI_HZ: f64 = 250.0;
const MID_HI_HZ: f64 = 4_000.0;

// --- AGC / smoothing constants ---------------------------------------------

/// Per-band AGC peak decay applied every hop. Chosen so the running peak halves
/// in ~2.5 s: `decay^(2.5 / hop_secs) = 0.5` with hop_secs ≈ 0.01067 →
/// `decay = 0.5^(hop_secs / 2.5) ≈ 0.9970`.
const AGC_PEAK_DECAY: f32 = 0.9970;
/// Smallest peak the AGC will divide by — clamps the noise-floor gain so silence
/// is not amplified into visible bands.
const AGC_PEAK_FLOOR: f32 = 1.0e-4;
/// Fast attack coefficient for band smoothing (rises quickly toward new value).
const BAND_ATTACK: f32 = 0.35;
/// Slow release coefficient for band smoothing (falls slowly).
const BAND_RELEASE: f32 = 0.87;

/// EWMA coefficient for the energy aggregate.
const ENERGY_EWMA: f32 = 0.80;
/// Scale mapping raw RMS to the 0..1 energy range. 0 dBFS sine has RMS ≈ 0.707;
/// this maps that to ~1.0 (then clamped).
const ENERGY_SCALE: f32 = 1.414;

/// `beat` pulse exponential decay per hop (~0.85).
const BEAT_DECAY: f32 = 0.85;

// --- Silence detection ------------------------------------------------------

/// RMS below this for the whole silence window ⇒ silence.
const SILENCE_RMS: f32 = 1.0e-4;
/// Sustained-silence duration before the flag latches (seconds).
const SILENCE_SECS: f64 = 0.5;

/// Per-hop output bundle returned by [`Analyzer`] internals (test convenience).
struct HopOutput {
    frame: FeatureFrame,
    beat: Option<BeatEvent>,
}

/// Precomputed lower frequency edges of all `BAND_COUNT` log-spaced bands
/// (plus the upper Nyquist edge at index `BAND_COUNT`).
fn band_edges() -> [f64; BAND_COUNT + 1] {
    let ratio = BAND_HI_HZ / BAND_LO_HZ;
    // edges[k] = lower frequency of band k; band k spans [edges[k], edges[k+1]).
    let mut edges = [0.0f64; BAND_COUNT + 1];
    for (k, slot) in edges.iter_mut().enumerate() {
        *slot = if k == BAND_COUNT {
            // upper edge of the last band -> Nyquist so high content lands in band 47.
            (SAMPLE_RATE as f64 / 2.0).max(BAND_HI_HZ)
        } else {
            BAND_LO_HZ * ratio.powf(k as f64 / (BAND_COUNT as f64 - 1.0))
        };
    }
    edges
}

/// Precomputed mapping of each FFT bin to a band index (or `usize::MAX` if the
/// bin is outside the 30 Hz–16 kHz range and ignored).
fn build_bin_to_band() -> Vec<usize> {
    let bin_hz = SAMPLE_RATE as f64 / FFT_SIZE as f64; // 46.875 Hz
    let edges = band_edges();

    let mut map = vec![usize::MAX; SPECTRUM_BINS];
    for (bin, slot) in map.iter_mut().enumerate() {
        let f = bin as f64 * bin_hz;
        if f < edges[0] || f >= edges[BAND_COUNT] {
            continue;
        }
        // Linear scan (48 bands) — only runs once at construction.
        for k in 0..BAND_COUNT {
            if f >= edges[k] && f < edges[k + 1] {
                *slot = k;
                break;
            }
        }
    }
    map
}

/// Precomputes linear-interpolation bin indices `(lo, hi, frac)` for any band
/// that contains zero FFT bins (at 48 kHz / 1024-pt FFT, narrow low-frequency
/// log bands fall between 46.875 Hz bin spacings). Clamped to `1..SPECTRUM_BINS-1`
/// so DC bin `0` is never sampled.
fn build_empty_band_interp(
    band_count_per: &[u32; BAND_COUNT],
) -> [Option<(usize, usize, f32)>; BAND_COUNT] {
    let bin_hz = SAMPLE_RATE as f64 / FFT_SIZE as f64;
    let edges = band_edges();
    let mut table = [None; BAND_COUNT];
    for k in 0..BAND_COUNT {
        if band_count_per[k] == 0 {
            let f_center = (edges[k] * edges[k + 1]).sqrt();
            let pos = (f_center / bin_hz).clamp(1.0, (SPECTRUM_BINS - 1) as f64);
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = (pos - lo as f64) as f32;
            table[k] = Some((lo, hi, frac));
        }
    }
    table
}

/// The pure analyzer. Construct once, then drive with [`Analyzer::push_samples`].
pub struct Analyzer {
    // FFT machinery (all pre-allocated).
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,        // Hann window, length FFT_SIZE
    ring: Vec<f32>,          // sliding sample ring, length FFT_SIZE
    ring_filled: usize,      // samples written before the ring first fills
    write_pos: usize,        // next write index into `ring`
    since_hop: usize,        // samples accumulated since the last hop
    fft_in: Vec<f32>,        // windowed frame (FFT input scratch)
    fft_out: Vec<Complex32>, // spectrum (length SPECTRUM_BINS)
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>, // per-bin power, length SPECTRUM_BINS
    mag: Vec<f32>,   // per-bin magnitude for the onset detector

    // Band mapping & state.
    /// Contiguous `[start..end)` FFT bin range per band (empty if 0-bin band).
    band_ranges: [(usize, usize); BAND_COUNT],
    /// Precomputed `1.0 / count` for non-empty bands (`0.0` for 0-bin bands).
    inv_band_count: [f32; BAND_COUNT],
    band_interp: [Option<(usize, usize, f32)>; BAND_COUNT], // fallback for 0-bin bands
    band_peak: [f32; BAND_COUNT],                           // AGC running peak per band
    band_smooth: [f32; BAND_COUNT],                         // smoothed output per band

    // Aggregate cutoffs in bins.
    low_hi_bin: usize,
    mid_hi_bin: usize,

    // Energy / loudness.
    energy_ewma: f32,

    // Onset.
    onset: Onset,
    beat_pulse: f32,
    beat_count: u32,

    // Clock & silence.
    samples_processed: u64,
    silence_run: u64, // consecutive samples whose hop RMS was sub-silence
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    /// Builds a fully pre-allocated analyzer. No allocation happens afterwards.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let fft_in = fft.make_input_vec();
        let fft_out = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();
        debug_assert_eq!(fft_out.len(), SPECTRUM_BINS);

        // Periodic Hann window (matches typical STFT analysis).
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                let x = std::f32::consts::TAU * n as f32 / FFT_SIZE as f32;
                0.5 - 0.5 * x.cos()
            })
            .collect();

        let bin_to_band = build_bin_to_band();
        let mut band_count_per = [0u32; BAND_COUNT];
        let mut band_ranges = [(0usize, 0usize); BAND_COUNT];
        for (bin, &b) in bin_to_band.iter().enumerate() {
            if b != usize::MAX {
                if band_count_per[b] == 0 {
                    band_ranges[b] = (bin, bin + 1);
                } else {
                    band_ranges[b].1 = bin + 1;
                }
                band_count_per[b] += 1;
            }
        }
        let mut inv_band_count = [0.0f32; BAND_COUNT];
        for k in 0..BAND_COUNT {
            if band_count_per[k] > 0 {
                inv_band_count[k] = 1.0 / band_count_per[k] as f32;
            }
        }
        let band_interp = build_empty_band_interp(&band_count_per);

        let bin_hz = SAMPLE_RATE as f64 / FFT_SIZE as f64;
        let low_hi_bin = (LOW_HI_HZ / bin_hz).round() as usize;
        let mid_hi_bin = (MID_HI_HZ / bin_hz).round() as usize;

        let hop_secs = HOP_SIZE as f64 / SAMPLE_RATE as f64;

        Self {
            fft,
            window,
            ring: vec![0.0; FFT_SIZE],
            ring_filled: 0,
            write_pos: 0,
            since_hop: 0,
            fft_in,
            fft_out,
            fft_scratch,
            power: vec![0.0; SPECTRUM_BINS],
            mag: vec![0.0; SPECTRUM_BINS],
            band_ranges,
            inv_band_count,
            band_interp,
            band_peak: [AGC_PEAK_FLOOR; BAND_COUNT],
            band_smooth: [0.0; BAND_COUNT],
            low_hi_bin,
            mid_hi_bin,
            energy_ewma: 0.0,
            onset: Onset::new(SPECTRUM_BINS, hop_secs),
            beat_pulse: 0.0,
            beat_count: 0,
            samples_processed: 0,
            silence_run: 0,
        }
    }

    /// Resets all running state (clock, AGC, onset history) to start fresh.
    pub fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.ring_filled = 0;
        self.write_pos = 0;
        self.since_hop = 0;
        self.band_peak = [AGC_PEAK_FLOOR; BAND_COUNT];
        self.band_smooth = [0.0; BAND_COUNT];
        self.energy_ewma = 0.0;
        self.onset.reset();
        self.beat_pulse = 0.0;
        self.beat_count = 0;
        self.samples_processed = 0;
        self.silence_run = 0;
    }

    /// Current audio-clock time in seconds (`samples_processed / SAMPLE_RATE`).
    pub fn clock_secs(&self) -> f64 {
        self.samples_processed as f64 / SAMPLE_RATE as f64
    }

    /// Feeds a block of mono samples. For every completed hop, `on_hop` is called
    /// with the freshly computed [`FeatureFrame`] and the (0 or 1) beat events
    /// produced on that hop.
    ///
    /// Allocation-free in the steady state: all buffers are pre-sized in
    /// [`Analyzer::new`]. The sample counter advances for *every* pushed sample,
    /// including zero-fill, so the clock stays aligned through dropouts (keeping
    /// the audio clock deterministic).
    pub fn push_samples(
        &mut self,
        mono: &[f32],
        mut on_hop: impl FnMut(&FeatureFrame, &[BeatEvent]),
    ) {
        let mut offset = 0;
        while offset < mono.len() {
            let until_hop = HOP_SIZE.saturating_sub(self.since_hop).max(1);
            let until_wrap = FFT_SIZE - self.write_pos;
            let chunk = (mono.len() - offset).min(until_hop).min(until_wrap);

            self.ring[self.write_pos..self.write_pos + chunk]
                .copy_from_slice(&mono[offset..offset + chunk]);
            self.write_pos += chunk;
            if self.write_pos == FFT_SIZE {
                self.write_pos = 0;
            }
            self.ring_filled = (self.ring_filled + chunk).min(FFT_SIZE);
            self.samples_processed += chunk as u64;
            self.since_hop += chunk;
            offset += chunk;

            if self.since_hop >= HOP_SIZE && self.ring_filled >= FFT_SIZE {
                self.since_hop = 0;
                let out = self.analyze_hop();
                match out.beat {
                    Some(ev) => on_hop(&out.frame, std::slice::from_ref(&ev)),
                    None => on_hop(&out.frame, &[]),
                }
            } else if self.since_hop >= HOP_SIZE {
                // Ring not yet full (very first window) — still consume the hop
                // boundary so timing does not drift, but emit nothing.
                self.since_hop = 0;
            }
        }
    }

    /// Runs the analysis on the current window. Allocation-free.
    fn analyze_hop(&mut self) -> HopOutput {
        let mut frame = FeatureFrame::default();
        let t = self.clock_secs();

        // 1. Compute hop RMS energy across the two contiguous slices of the ring
        //    buffer (auto-vectorizable). The oldest sample is at `write_pos`.
        let wp = self.write_pos;
        let tail = &self.ring[wp..];
        let head = &self.ring[..wp];
        let mut rms_acc = 0.0f64;
        for &raw in tail {
            rms_acc += (raw as f64) * (raw as f64);
        }
        for &raw in head {
            rms_acc += (raw as f64) * (raw as f64);
        }

        // Fast path for exact digital silence: skip Hann windowing, 1024-pt FFT,
        // 513 sqrts, 48-band folding, and SuperFlux spectral loops (>95% DSP CPU
        // saved during silence) while decaying AGC/band/energy/beat/onset state
        // identically to a zero-spectrum hop.
        if rms_acc == 0.0 {
            for k in 0..BAND_COUNT {
                self.band_peak[k] = (self.band_peak[k] * AGC_PEAK_DECAY).max(AGC_PEAK_FLOOR);
                let smoothed = self.band_smooth[k] * BAND_RELEASE;
                self.band_smooth[k] = smoothed;
                frame.bands[k] = sanitize(smoothed).clamp(0.0, 1.0);
            }
            self.energy_ewma *= ENERGY_EWMA;
            frame.energy = sanitize(self.energy_ewma).clamp(0.0, 1.0);

            let _ = self.onset.process_silence();
            self.beat_pulse *= BEAT_DECAY;
            frame.beat = sanitize(self.beat_pulse).clamp(0.0, 1.0);
            frame.beat_count = self.beat_count;

            frame.t = t;
            self.silence_run = self.silence_run.saturating_add(HOP_SIZE as u64);
            let silence_thresh = (SILENCE_SECS * SAMPLE_RATE as f64) as u64;
            frame.silence = self.silence_run >= silence_thresh;
            return HopOutput { frame, beat: None };
        }

        // 1b. Apply Hann window across the two contiguous slices into `fft_in`.
        let tail_len = tail.len();
        for (dst, (&raw, &w)) in self.fft_in[..tail_len]
            .iter_mut()
            .zip(tail.iter().zip(self.window[..tail_len].iter()))
        {
            *dst = raw * w;
        }
        for (dst, (&raw, &w)) in self.fft_in[tail_len..]
            .iter_mut()
            .zip(head.iter().zip(self.window[tail_len..].iter()))
        {
            *dst = raw * w;
        }

        // Waveform = the most recent WAVEFORM_LEN raw samples of the window.
        let wf_start = (wp + FFT_SIZE - WAVEFORM_LEN) % FFT_SIZE;
        let wf_first = (FFT_SIZE - wf_start).min(WAVEFORM_LEN);
        frame.waveform[..wf_first].copy_from_slice(&self.ring[wf_start..wf_start + wf_first]);
        if wf_first < WAVEFORM_LEN {
            frame.waveform[wf_first..].copy_from_slice(&self.ring[..WAVEFORM_LEN - wf_first]);
        }

        // 2. FFT (zero allocation).
        self.fft
            .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
            .expect("FFT lengths are fixed at construction");

        // 3. Per-bin power & magnitude.
        let inv_n2 = (1.0 / FFT_SIZE as f32) * (1.0 / FFT_SIZE as f32);
        for (k, c) in self.fft_out.iter().enumerate() {
            let p = (c.re * c.re + c.im * c.im) * inv_n2;
            self.power[k] = p;
            self.mag[k] = p.sqrt();
        }

        // 4–5. Fold contiguous bin ranges into 48 bands → AGC → asymmetric smoothing.
        for k in 0..BAND_COUNT {
            let inv = self.inv_band_count[k];
            let raw = if inv > 0.0 {
                let (lo, hi) = self.band_ranges[k];
                let sum: f32 = self.power[lo..hi].iter().sum();
                sum * inv
            } else if let Some((lo, hi, frac)) = self.band_interp[k] {
                self.power[lo] * (1.0 - frac) + self.power[hi] * frac
            } else {
                0.0
            };
            // Use amplitude (sqrt of mean power) for a perceptually flatter scale.
            let amp = raw.sqrt();

            // Decay the running peak, then raise it to the new value if larger.
            let peak = (self.band_peak[k] * AGC_PEAK_DECAY)
                .max(AGC_PEAK_FLOOR)
                .max(amp);
            self.band_peak[k] = peak;

            let target = (amp / peak).clamp(0.0, 1.0);
            let prev = self.band_smooth[k];
            let smoothed = if target > prev {
                prev + (target - prev) * BAND_ATTACK
            } else {
                prev * BAND_RELEASE + target * (1.0 - BAND_RELEASE)
            };
            self.band_smooth[k] = smoothed;
            frame.bands[k] = sanitize(smoothed).clamp(0.0, 1.0);
        }

        // 6. Aggregates low/mid/high from contiguous slices of `self.power` (skip DC bin 0).
        let low_end = self.low_hi_bin.min(SPECTRUM_BINS - 1);
        let mid_end = self.mid_hi_bin.min(SPECTRUM_BINS - 1);
        let low_slice = &self.power[1..=low_end];
        let mid_slice = &self.power[low_end + 1..=mid_end];
        let high_slice = &self.power[mid_end + 1..];
        frame.low = aggregate_level(low_slice.iter().sum(), low_slice.len() as u32);
        frame.mid = aggregate_level(mid_slice.iter().sum(), mid_slice.len() as u32);
        frame.high = aggregate_level(high_slice.iter().sum(), high_slice.len() as u32);

        // 7. Energy from hop RMS through an EWMA.
        let rms = (rms_acc / FFT_SIZE as f64).sqrt() as f32;
        let raw_energy = (rms * ENERGY_SCALE).clamp(0.0, 1.0);
        self.energy_ewma = self.energy_ewma * ENERGY_EWMA + raw_energy * (1.0 - ENERGY_EWMA);
        frame.energy = sanitize(self.energy_ewma).clamp(0.0, 1.0);

        // 8. Onset detection on the magnitude spectrum.
        let onset = self.onset.process(&self.mag, t);
        let mut beat_ev = None;
        if onset.is_onset {
            self.beat_pulse = 1.0;
            self.beat_count += 1;
            beat_ev = Some(BeatEvent {
                t,
                strength: onset.strength.max(0.0),
            });
        } else {
            self.beat_pulse *= BEAT_DECAY;
        }
        frame.beat = sanitize(self.beat_pulse).clamp(0.0, 1.0);
        frame.beat_count = self.beat_count;

        // 9. Clock + silence flag.
        frame.t = t;
        let silence_hop_samples = HOP_SIZE as u64;
        if rms < SILENCE_RMS {
            self.silence_run = self.silence_run.saturating_add(silence_hop_samples);
        } else {
            self.silence_run = 0;
        }
        let silence_thresh = (SILENCE_SECS * SAMPLE_RATE as f64) as u64;
        frame.silence = self.silence_run >= silence_thresh;

        HopOutput {
            frame,
            beat: beat_ev,
        }
    }
}

/// Maps an accumulated power sum over `n` bins to a 0..1 level via amplitude.
fn aggregate_level(power_sum: f32, n: u32) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let amp = (power_sum / n as f32).sqrt();
    // Mild fixed gain so a 0 dBFS tone in-band reads near 1.0 without an AGC.
    sanitize(amp * 2.0).clamp(0.0, 1.0)
}

/// Replaces NaN/±Inf with 0.0 (sanitizes non-finite values at the source).
#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_band_map_covers_all_bands() {
        let map = build_bin_to_band();
        let mut counts = [0u32; BAND_COUNT];
        for &b in &map {
            if b != usize::MAX {
                counts[b] += 1;
            }
        }
        // Higher bands (wider in Hz) must contain at least one bin.
        for (k, &c) in counts.iter().enumerate().skip(BAND_COUNT / 2) {
            assert!(c >= 1, "band {k} has no bins");
        }
    }

    #[test]
    fn clock_advances_with_zero_fill() {
        let mut a = Analyzer::new();
        let zeros = vec![0.0f32; HOP_SIZE * 4];
        a.push_samples(&zeros, |_, _| {});
        assert_eq!(a.samples_processed, (HOP_SIZE * 4) as u64);
        assert!((a.clock_secs() - (HOP_SIZE * 4) as f64 / SAMPLE_RATE as f64).abs() < 1e-12);
    }

    #[test]
    fn low_frequency_bands_have_no_dead_slots_and_preserve_silence() {
        // Silence must stay strictly zero across all bands (including interpolated ones).
        let mut silent = Analyzer::new();
        let zeros = vec![0.0f32; SAMPLE_RATE as usize / 2];
        let mut last_silent = FeatureFrame::default();
        silent.push_samples(&zeros, |f, _| last_silent = *f);
        for (k, &v) in last_silent.bands.iter().enumerate() {
            assert_eq!(v, 0.0, "silent band {k} must be 0.0, got {v}");
        }

        // Low-frequency / broadband sum covering 35 Hz..300 Hz must excite ALL bands 0..16.
        let mut a = Analyzer::new();
        let n = SAMPLE_RATE as usize / 2;
        let freqs = [40.0f64, 65.0, 95.0, 140.0, 200.0, 280.0];
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                let sum: f64 = freqs
                    .iter()
                    .map(|&f| (std::f64::consts::TAU * f * t).sin())
                    .sum();
                (sum / freqs.len() as f64 * 0.8) as f32
            })
            .collect();
        let mut last = FeatureFrame::default();
        a.push_samples(&signal, |f, _| last = *f);
        for k in 0..16 {
            assert!(
                last.bands[k] > 0.05,
                "low band {k} was dead/under-excited ({}), bands[0..16]={:?}",
                last.bands[k],
                &last.bands[..16]
            );
        }
    }
}
