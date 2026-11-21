//! Windowed live performance harness for ViewMusic.
//!
//! Measures what the headless `perf_harness` cannot: `CAMetalLayer` drawable
//! acquire/present, drawable memory, and the real frame pacing of
//! [`viz_app::window::App`] (`about_to_wait`, `WaitUntil`, the 60 Hz cap, the
//! occlusion pause). It runs the real `App` with a [`RenderCallback`] that owns an
//! [`ArtifactRuntime`] + [`Renderer`] (like `VizApp`) and feeds a synthetic signal
//! (the same seeded 120 BPM generator as `perf_harness`) through the real
//! [`viz_audio::Analyzer`] in real time: each frame pushes the samples that elapsed
//! on the wall clock since the previous frame. No Core Audio, no
//! `AudioPipeline::start`, so no permission prompt. The analyzer runs on the main
//! thread here (the app runs it on the capture thread); process CPU covers both.
//!
//! Each selected artifact renders live for `--warmup-secs` and then a measured
//! window of `--secs`. Artifacts switch in-process (fresh runtime, same renderer, as
//! on a dropdown switch). The window must stay visible and unoccluded, otherwise
//! `App` pauses rendering and the fps column shows it. After the last artifact the
//! event loop exits.
//!
//! Run: `scripts/perf-live.sh [flags]`, or
//! `nix develop -c cargo run --release -p viz-app --example live_harness -- [flags]`.
//!
//! Flags:
//! - `--size WxH` (default 1512x945) window inner size in points; on the Retina
//!   built-in display that is 3024x1890 px. The window opens on the primary display.
//! - `--only id,id` restrict to these built-in artifact ids (default all 17).
//! - `--secs S` (default 10) measured seconds per artifact.
//! - `--warmup-secs S` (default 2) unmeasured seconds per artifact before the window.
//! - `--overlay` attach the egui overlay (`viz_app::overlay::Overlay`, placeholder
//!   panel) and paint it every frame. The app's own panel is private and auto-hides
//!   after 5 s, so this is the "overlay visible" upper bound.
//!
//! Per artifact: process CPU % of one core (rusage delta / wall), GPU % (ioreg
//! `AGXDeviceUserClient` accumulatedGPUTime delta for this pid / wall), presented
//! frames per second, the largest gap between presents, and phys_footprint (current
//! and lifetime max). The last stdout line is a machine-parseable JSON summary.

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("live_harness is macOS-only");
    std::process::ExitCode::FAILURE
}

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    harness::main()
}

/// The harness itself. macOS-only: it reads process and GPU counters through
/// macOS APIs (proc_pid_rusage, QoS classes, ioreg AGX counters, CAMetalLayer).
#[cfg(target_os = "macos")]
mod harness {
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::process::{Command, ExitCode};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use serde_json::json;
    use viz_app::overlay::Overlay;
    use viz_app::runtime::ArtifactRuntime;
    use viz_app::window::{App, FrameContext, RenderCallback};
    use viz_audio::Analyzer;
    use viz_contract::{ArtifactLibrary, LoadedArtifact};
    use viz_core::{FeatureFrame, SAMPLE_RATE};
    use viz_render::{LayerDraw, Renderer};
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowId};

    /// Same set and order as `BUILTIN_ARTIFACTS` in `crates/viz-app/src/main.rs`.
    const BUILTIN_ARTIFACTS: &[(&str, &str)] = &[
        (
            "spectrum-bars (built-in)",
            include_str!("../../../assets/builtin-artifacts/spectrum-bars.artifact.json"),
        ),
        (
            "oscilloscope (built-in)",
            include_str!("../../../assets/builtin-artifacts/oscilloscope.artifact.json"),
        ),
        (
            "radial-pulse (built-in)",
            include_str!("../../../assets/builtin-artifacts/radial-pulse.artifact.json"),
        ),
        (
            "particle-burst (built-in)",
            include_str!("../../../assets/builtin-artifacts/particle-burst.artifact.json"),
        ),
        (
            "color-field (built-in)",
            include_str!("../../../assets/builtin-artifacts/color-field.artifact.json"),
        ),
        (
            "starfield-warp (built-in)",
            include_str!("../../../assets/builtin-artifacts/starfield-warp.artifact.json"),
        ),
        (
            "liquid-spectrum (built-in)",
            include_str!("../../../assets/builtin-artifacts/liquid-spectrum.artifact.json"),
        ),
        (
            "dna-helix (built-in)",
            include_str!("../../../assets/builtin-artifacts/dna-helix.artifact.json"),
        ),
        (
            "kaleido-petals (built-in)",
            include_str!("../../../assets/builtin-artifacts/kaleido-petals.artifact.json"),
        ),
        (
            "bass-tunnel (built-in)",
            include_str!("../../../assets/builtin-artifacts/bass-tunnel.artifact.json"),
        ),
        (
            "aurora-waves (built-in)",
            include_str!("../../../assets/builtin-artifacts/aurora-waves.artifact.json"),
        ),
        (
            "neon-gauges (built-in)",
            include_str!("../../../assets/builtin-artifacts/neon-gauges.artifact.json"),
        ),
        (
            "spectrum-galaxy (built-in)",
            include_str!("../../../assets/builtin-artifacts/spectrum-galaxy.artifact.json"),
        ),
        (
            "pixel-rain (built-in)",
            include_str!("../../../assets/builtin-artifacts/pixel-rain.artifact.json"),
        ),
        (
            "breathing-grid (built-in)",
            include_str!("../../../assets/builtin-artifacts/breathing-grid.artifact.json"),
        ),
        (
            "retro-spectrum-bars (built-in)",
            include_str!("../../../assets/builtin-artifacts/retro-spectrum-bars.artifact.json"),
        ),
        (
            "winamp-spectrum-bars (built-in)",
            include_str!("../../../assets/builtin-artifacts/winamp-spectrum-bars.artifact.json"),
        ),
    ];

    const DEFAULT_WIDTH_PT: u32 = 1512;
    const DEFAULT_HEIGHT_PT: u32 = 945;
    const DEFAULT_SECS: f64 = 10.0;
    const DEFAULT_WARMUP_SECS: f64 = 2.0;
    /// Window top-left offset from the primary display's origin, in points.
    const WINDOW_OFFSET_PT: f64 = 40.0;
    /// Same clamp as `MAX_DT` in `crates/viz-app/src/viz.rs` (`VizApp::poll_audio`).
    const MAX_DT: f64 = 0.1;
    /// Same seed as `perf_harness`.
    const SIGNAL_SEED: u64 = 0x5EED_0FA0_D10C;
    /// Largest slice pushed into the analyzer at once (100 ms); longer wall-clock gaps
    /// are pushed in several slices so no audio is skipped.
    const MAX_PUSH_SAMPLES: usize = SAMPLE_RATE as usize / 10;
    /// Extra wall time allowed per artifact before the watchdog aborts a stalled run.
    const WATCHDOG_SLACK_SECS: f64 = 20.0;

    // ---------------------------------------------------------------------------
    // CLI
    // ---------------------------------------------------------------------------

    struct Args {
        width_pt: u32,
        height_pt: u32,
        only: Option<Vec<String>>,
        secs: f64,
        warmup_secs: f64,
        overlay: bool,
    }

    const USAGE: &str =
        "usage: live_harness [--size WxH] [--only id,id] [--secs S] [--warmup-secs S] [--overlay]";

    fn parse_args() -> Result<Args, String> {
        let mut args = Args {
            width_pt: DEFAULT_WIDTH_PT,
            height_pt: DEFAULT_HEIGHT_PT,
            only: None,
            secs: DEFAULT_SECS,
            warmup_secs: DEFAULT_WARMUP_SECS,
            overlay: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
            match flag.as_str() {
                "--size" => {
                    let v = value("--size")?;
                    let (w, h) = v
                        .split_once('x')
                        .ok_or_else(|| format!("--size expects WxH, got {v}"))?;
                    args.width_pt = w.parse().map_err(|_| format!("bad width in {v}"))?;
                    args.height_pt = h.parse().map_err(|_| format!("bad height in {v}"))?;
                    if args.width_pt == 0 || args.height_pt == 0 {
                        return Err("--size must be non-zero".into());
                    }
                }
                "--only" => {
                    let v = value("--only")?;
                    args.only = Some(v.split(',').map(|s| s.trim().to_owned()).collect());
                }
                "--secs" => {
                    let v = value("--secs")?;
                    args.secs = v.parse().map_err(|_| format!("bad --secs {v}"))?;
                    if !args.secs.is_finite() || args.secs <= 0.0 {
                        return Err("--secs must be a finite number > 0".into());
                    }
                }
                "--warmup-secs" => {
                    let v = value("--warmup-secs")?;
                    args.warmup_secs = v.parse().map_err(|_| format!("bad --warmup-secs {v}"))?;
                    if !args.warmup_secs.is_finite() || args.warmup_secs < 0.0 {
                        return Err("--warmup-secs must be a finite number >= 0".into());
                    }
                }
                "--overlay" => args.overlay = true,
                "-h" | "--help" => return Err(String::new()),
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(args)
    }

    // ---------------------------------------------------------------------------
    // Synthetic music-like signal (streaming form of `perf_harness::synth_signal`)
    // ---------------------------------------------------------------------------

    /// The deterministic 120 BPM mono "track" of `perf_harness`: a pitch-swept kick on
    /// every beat, a 3-harmonic bass line (one note per beat), a sustained chord pad
    /// changing every bar, and high-passed noise hi-hats on every 8th note. Same math
    /// and seed, generated incrementally so an arbitrarily long run needs no buffer.
    struct SynthSignal {
        /// Index of the next sample.
        n: usize,
        rng: u64,
        prev_white: f64,
    }

    impl SynthSignal {
        fn new(seed: u64) -> Self {
            Self {
                n: 0,
                rng: seed | 1,
                prev_white: 0.0,
            }
        }

        fn fill(&mut self, out: &mut [f32]) {
            use std::f64::consts::TAU;
            const BEAT_SECS: f64 = 0.5; // 120 BPM
            const EIGHTH_SECS: f64 = 0.25;
            const BASS_HZ: [f64; 8] = [55.0, 55.0, 65.41, 55.0, 43.65, 43.65, 49.0, 49.0];
            const CHORDS_HZ: [[f64; 3]; 4] = [
                [220.0, 261.63, 329.63], // Am
                [174.61, 220.0, 261.63], // F
                [261.63, 329.63, 392.0], // C
                [196.0, 246.94, 293.66], // G
            ];
            let sr = SAMPLE_RATE as f64;
            for sample in out {
                let t = self.n as f64 / sr;
                self.n += 1;
                let beat = (t / BEAT_SECS).floor();
                let tb = t - beat * BEAT_SECS;
                let beat_i = beat as usize;

                // Kick: frequency 50 + 100·e^(−tb/0.03) Hz (phase is its integral).
                let kick_phase = TAU * (50.0 * tb + 3.0 * (1.0 - (-tb / 0.03).exp()));
                let kick = kick_phase.sin() * (-tb / 0.15).exp() * 0.9;

                // Bass: one note per beat, short attack, 0.4 s decay.
                let bp = TAU * BASS_HZ[beat_i % BASS_HZ.len()] * tb;
                let bass = (bp.sin() + 0.5 * (2.0 * bp).sin() + 0.25 * (3.0 * bp).sin())
                    * 0.25
                    * (1.0 - (-tb / 0.01).exp())
                    * (-tb / 0.4).exp();

                // Chord pad: three partials, chord changes every bar (4 beats).
                let chord = CHORDS_HZ[(beat_i / 4) % CHORDS_HZ.len()];
                let tremolo = 0.8 + 0.2 * (TAU * 0.5 * t).sin();
                let pad = chord.iter().map(|f| (TAU * f * t).sin()).sum::<f64>() * 0.05 * tremolo;

                // Hi-hat: first-difference (high-passed) xorshift noise, 30 ms decay,
                // accented on the off-beat 8ths.
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                let white = (self.rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                let hp = white - self.prev_white;
                self.prev_white = white;
                let eighth = (t / EIGHTH_SECS).floor();
                let t8 = t - eighth * EIGHTH_SECS;
                let accent = if (eighth as u64) % 2 == 1 { 1.0 } else { 0.6 };
                let hat = hp * (-t8 / 0.03).exp() * 0.15 * accent;

                *sample = ((kick + bass + pad + hat) * 0.7).clamp(-1.0, 1.0) as f32;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Process metrics (same sources as perf_harness)
    // ---------------------------------------------------------------------------

    /// Process CPU time (user + sys, all threads) in microseconds.
    fn process_cpu_us() -> u64 {
        // SAFETY: getrusage writes a fully initialised `rusage` for RUSAGE_SELF.
        let ru = unsafe {
            let mut ru: libc::rusage = std::mem::zeroed();
            libc::getrusage(libc::RUSAGE_SELF, &mut ru);
            ru
        };
        let tv = |t: libc::timeval| t.tv_sec as u64 * 1_000_000 + t.tv_usec as u64;
        tv(ru.ru_utime) + tv(ru.ru_stime)
    }

    /// `(phys_footprint, lifetime_max_phys_footprint)` in bytes.
    fn phys_footprint_bytes() -> (u64, u64) {
        // SAFETY: proc_pid_rusage fills a `rusage_info_v4` for RUSAGE_INFO_V4.
        unsafe {
            let mut info: libc::rusage_info_v4 = std::mem::zeroed();
            let rc = libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                (&mut info as *mut libc::rusage_info_v4).cast(),
            );
            if rc != 0 {
                return (0, 0);
            }
            (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
        }
    }

    fn mb(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// One `ioreg` read of every `AGXDeviceUserClient`'s summed
    /// `AppUsage[*].accumulatedGPUTime` (ns), keyed by registry entry id.
    struct GpuUsageSnapshot {
        /// `(registry id, created by this pid, accumulated GPU ns)`.
        clients: Vec<(String, bool, u64)>,
    }

    /// GPU nanoseconds accumulated between two snapshots.
    #[derive(Clone, Copy, Debug, Default)]
    struct GpuUsageDelta {
        /// This process's clients.
        ours_ns: u64,
        /// Every other process's clients (WindowServer compositing included).
        other_ns: u64,
    }

    impl GpuUsageSnapshot {
        /// Shells out to `ioreg -l -w0 -r -c AGXDeviceUserClient`. `None` when ioreg
        /// fails or has no usage entry for this process.
        fn read() -> Option<Self> {
            let out = Command::new("ioreg")
                .args(["-l", "-w0", "-r", "-c", "AGXDeviceUserClient"])
                .output()
                .ok()?;
            let text = String::from_utf8_lossy(&out.stdout);
            let creator = format!("\"IOUserClientCreator\" = \"pid {},", std::process::id());
            let mut clients = Vec::new();
            let mut ours_found = false;
            for block in text.split("+-o AGXDeviceUserClient").skip(1) {
                let Some(id) = block
                    .split_once(", id ")
                    .and_then(|(_, rest)| rest.split(',').next())
                else {
                    continue;
                };
                let Some(usage_line) = block.lines().find(|l| l.contains("\"AppUsage\" = ")) else {
                    continue;
                };
                const KEY: &str = "\"accumulatedGPUTime\"=";
                let mut total = 0u64;
                let mut entries = 0usize;
                let mut rest = usage_line;
                while let Some(pos) = rest.find(KEY) {
                    rest = &rest[pos + KEY.len()..];
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(v) = digits.parse::<u64>() {
                        total += v;
                        entries += 1;
                    }
                }
                let ours = block.contains(&creator);
                ours_found |= ours && entries > 0;
                clients.push((id.to_owned(), ours, total));
            }
            ours_found.then_some(Self { clients })
        }

        /// Usage accumulated since `earlier`. Clients that vanished are ignored; clients
        /// created in between count in full.
        fn since(&self, earlier: &Self) -> GpuUsageDelta {
            let mut delta = GpuUsageDelta::default();
            for (id, ours, total) in &self.clients {
                let before = earlier
                    .clients
                    .iter()
                    .find(|(eid, _, _)| eid == id)
                    .map_or(0, |(_, _, t)| *t);
                let d = total.saturating_sub(before);
                if *ours {
                    delta.ours_ns += d;
                } else {
                    delta.other_ns += d;
                }
            }
            delta
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct WorkCounts {
        layers: u64,
        instances: u64,
        points: u64,
        cells: u64,
    }

    impl WorkCounts {
        fn add(&mut self, layers: &[LayerDraw]) {
            self.layers += layers.len() as u64;
            for layer in layers {
                match layer {
                    LayerDraw::Instanced { instances, .. } => {
                        self.instances += instances.len() as u64
                    }
                    LayerDraw::Polyline { points, .. } => self.points += points.len() as u64,
                    LayerDraw::Field { cells, .. } => self.cells += cells.len() as u64,
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // The live render callback
    // ---------------------------------------------------------------------------

    /// One artifact's measured window.
    struct ArtifactResult {
        id: String,
        surface: (u32, u32),
        scale_factor: f64,
        wall_secs: f64,
        frames: u64,
        cpu_pct: f64,
        /// `None` when ioreg had no usage entry for this pid.
        gpu_pct: Option<f64>,
        other_gpu_pct: Option<f64>,
        max_gap_ms: f64,
        footprint_mb: f64,
        footprint_peak_mb: f64,
        work: WorkCounts,
        errors: u64,
    }

    /// Shared between the callback (owned by `App`) and `main`.
    #[derive(Default)]
    struct Report {
        results: Vec<ArtifactResult>,
        panics: u64,
        /// Set after the last artifact's window; the event loop exits on its next
        /// `about_to_wait`.
        done: bool,
    }

    struct MeasureWindow {
        start: Instant,
        cpu_before_us: u64,
        gpu_before: Option<GpuUsageSnapshot>,
        errors_before: u64,
        frames: u64,
        work: WorkCounts,
        last_frame_at: Instant,
        max_gap: Duration,
    }

    enum Phase {
        Warmup { until: Instant },
        Measure(MeasureWindow),
    }

    struct LiveHarness {
        artifacts: Vec<LoadedArtifact>,
        current: usize,
        /// `None` until the current artifact's first frame (built lazily so the switch
        /// happens inside a frame, as on a dropdown switch).
        runtime: Option<ArtifactRuntime>,
        phase: Phase,
        renderer: Option<Renderer>,
        overlay: Option<Overlay>,
        analyzer: Analyzer,
        signal: SynthSignal,
        push_buf: Vec<f32>,
        audio_start: Option<Instant>,
        samples_pushed: u64,
        feature: FeatureFrame,
        prev_t: Option<f64>,
        secs: Duration,
        warmup: Duration,
        errors: Arc<AtomicU64>,
        report: Rc<RefCell<Report>>,
    }

    impl LiveHarness {
        /// Pushes the samples that elapsed on the wall clock since the previous frame,
        /// keeping the latest feature frame (as the capture thread + `latest_frame` do).
        fn advance_audio(&mut self, now: Instant) {
            let start = *self.audio_start.get_or_insert(now);
            let target = (now.duration_since(start).as_secs_f64() * SAMPLE_RATE as f64) as u64;
            let feature = &mut self.feature;
            while self.samples_pushed < target {
                let n = ((target - self.samples_pushed) as usize).min(MAX_PUSH_SAMPLES);
                let chunk = &mut self.push_buf[..n];
                self.signal.fill(chunk);
                self.analyzer
                    .push_samples(chunk, |frame, _beats| *feature = *frame);
                self.samples_pushed += n as u64;
            }
        }

        fn render_scene(&mut self, frame: &mut FrameContext<'_>) {
            let (Some(renderer), Some(runtime)) = (self.renderer.as_mut(), self.runtime.as_mut())
            else {
                return;
            };
            let dt = match self.prev_t {
                None => 0.0,
                Some(prev) => (self.feature.t - prev).clamp(0.0, MAX_DT),
            };
            self.prev_t = Some(self.feature.t);
            let aspect = frame.config.width as f32 / frame.config.height.max(1) as f32;
            let mut scene = runtime.frame(&self.feature, dt, aspect, 1.0);
            scene.params.transparent_bg = false;
            if let Phase::Measure(m) = &mut self.phase {
                m.work.add(&scene.layers);
            }
            renderer.render(
                frame.device,
                frame.queue,
                frame.encoder,
                frame.view,
                &scene.params,
                &scene.layers,
            );
        }

        fn begin_measure(&self) -> MeasureWindow {
            let gpu_before = GpuUsageSnapshot::read();
            let errors_before = self.errors.load(Ordering::Relaxed);
            let cpu_before_us = process_cpu_us();
            let start = Instant::now();
            MeasureWindow {
                start,
                cpu_before_us,
                gpu_before,
                errors_before,
                frames: 0,
                work: WorkCounts::default(),
                last_frame_at: start,
                max_gap: Duration::ZERO,
            }
        }

        fn finish_measure(&self, m: &MeasureWindow, frame: &FrameContext<'_>) -> ArtifactResult {
            let wall = m.start.elapsed();
            let cpu_us = process_cpu_us().saturating_sub(m.cpu_before_us);
            let gpu = match (GpuUsageSnapshot::read(), m.gpu_before.as_ref()) {
                (Some(after), Some(before)) => Some(after.since(before)),
                _ => None,
            };
            let (footprint, footprint_peak) = phys_footprint_bytes();
            let wall_s = wall.as_secs_f64();
            let pct_ns = |ns: u64| ns as f64 / (wall_s * 1e9) * 100.0;
            ArtifactResult {
                id: self.artifacts[self.current].id.0.clone(),
                surface: (frame.config.width, frame.config.height),
                scale_factor: frame.window.scale_factor(),
                wall_secs: wall_s,
                frames: m.frames,
                cpu_pct: cpu_us as f64 / (wall_s * 1e6) * 100.0,
                gpu_pct: gpu.map(|d| pct_ns(d.ours_ns)),
                other_gpu_pct: gpu.map(|d| pct_ns(d.other_ns)),
                max_gap_ms: m.max_gap.as_secs_f64() * 1e3,
                footprint_mb: mb(footprint),
                footprint_peak_mb: mb(footprint_peak),
                work: m.work,
                errors: self.errors.load(Ordering::Relaxed) - m.errors_before,
            }
        }

        /// Moves on to the next artifact (or finishes the run).
        fn next_artifact(&mut self) {
            self.runtime = None;
            self.current += 1;
            if self.current >= self.artifacts.len() {
                self.report.borrow_mut().done = true;
            }
        }
    }

    impl RenderCallback for LiveHarness {
        fn render(&mut self, frame: &mut FrameContext<'_>) {
            if self.report.borrow().done {
                return;
            }
            if let Some(r) = self.renderer.as_mut() {
                r.resize(frame.device, frame.config.width, frame.config.height);
            }
            let now = Instant::now();
            if self.runtime.is_none() {
                let mut runtime = ArtifactRuntime::new(self.artifacts[self.current].clone());
                runtime.activate();
                self.runtime = Some(runtime);
                self.phase = Phase::Warmup {
                    until: now + self.warmup,
                };
            }
            if let Phase::Measure(m) = &mut self.phase {
                m.max_gap = m.max_gap.max(now.duration_since(m.last_frame_at));
                m.last_frame_at = now;
                m.frames += 1;
            }

            let rendered = catch_unwind(AssertUnwindSafe(|| {
                self.advance_audio(now);
                self.render_scene(frame);
                if let Some(overlay) = self.overlay.as_mut() {
                    overlay.render(frame);
                }
            }));
            if rendered.is_err() {
                eprintln!(
                    "!!! PANIC while running artifact {}",
                    self.artifacts[self.current].id.0
                );
                self.report.borrow_mut().panics += 1;
                self.next_artifact();
                return;
            }

            let now = Instant::now();
            match &self.phase {
                Phase::Warmup { until } if now >= *until => {
                    self.phase = Phase::Measure(self.begin_measure());
                }
                Phase::Measure(m) if now.duration_since(m.start) >= self.secs => {
                    let result = self.finish_measure(m, frame);
                    print_row(&result);
                    self.report.borrow_mut().results.push(result);
                    self.next_artifact();
                }
                _ => {}
            }
        }

        fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
            match self.overlay.as_mut() {
                Some(overlay) => overlay.on_window_event(window, event),
                None => false,
            }
        }

        fn owns_surface_clear(&self) -> bool {
            true
        }

        fn on_post_submit(&mut self) {
            if let Some(r) = self.renderer.as_mut() {
                r.recall_staging();
            }
        }
    }

    /// Delegates to the real [`App`] unchanged and only adds the exit after the last
    /// artifact (the `RenderCallback` seam has no access to the event loop).
    struct HarnessApp {
        app: App<LiveHarness>,
        report: Rc<RefCell<Report>>,
    }

    impl ApplicationHandler for HarnessApp {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            self.app.resumed(event_loop);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
            self.app.window_event(event_loop, id, event);
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            if self.report.borrow().done {
                event_loop.exit();
                return;
            }
            self.app.about_to_wait(event_loop);
        }
    }

    // ---------------------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------------------

    fn print_header() {
        println!(
        "{:<17} {:>9} {:>5} {:>6} {:>6} {:>7} {:>6} {:>8} {:>8} {:>8} {:>6} {:>7} {:>7} {:>7} {:>4}",
        "artifact",
        "surface",
        "scale",
        "fps",
        "cpu%",
        "gpu%",
        "oth%",
        "maxgapms",
        "foot MB",
        "peak MB",
        "lay/f",
        "inst/f",
        "pts/f",
        "cells/f",
        "err"
    );
    }

    fn opt_pct(v: Option<f64>) -> String {
        v.map_or_else(|| "n/a".to_owned(), |p| format!("{p:.1}"))
    }

    fn print_row(r: &ArtifactResult) {
        let per_frame = |v: u64| v as f64 / r.frames.max(1) as f64;
        println!(
        "{:<17} {:>9} {:>5.1} {:>6.1} {:>6.1} {:>7} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>6.2} {:>7.0} {:>7.0} {:>7.0} {:>4}",
        r.id,
        format!("{}x{}", r.surface.0, r.surface.1),
        r.scale_factor,
        r.frames as f64 / r.wall_secs,
        r.cpu_pct,
        opt_pct(r.gpu_pct),
        opt_pct(r.other_gpu_pct),
        r.max_gap_ms,
        r.footprint_mb,
        r.footprint_peak_mb,
        per_frame(r.work.layers),
        per_frame(r.work.instances),
        per_frame(r.work.points),
        per_frame(r.work.cells),
        r.errors
    );
    }

    fn mean(values: impl Iterator<Item = f64>) -> f64 {
        let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
        if n == 0 {
            f64::NAN
        } else {
            sum / n as f64
        }
    }

    // ---------------------------------------------------------------------------
    // main
    // ---------------------------------------------------------------------------

    pub fn main() -> ExitCode {
        let args = match parse_args() {
            Ok(a) => a,
            Err(e) => {
                if !e.is_empty() {
                    eprintln!("error: {e}");
                }
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        };

        // Match the app's AppKit main thread QoS (a CLI main thread starts lower).
        // SAFETY: plain libpthread call on the current thread.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }

        let library = ArtifactLibrary::load_builtins(BUILTIN_ARTIFACTS.iter().copied());
        let all: Vec<LoadedArtifact> = library
            .entries()
            .iter()
            .filter_map(|e| e.loaded())
            .cloned()
            .collect();
        if all.len() != BUILTIN_ARTIFACTS.len() {
            eprintln!(
                "!!! only {} of {} built-in artifacts loaded",
                all.len(),
                BUILTIN_ARTIFACTS.len()
            );
            return ExitCode::FAILURE;
        }
        let artifacts: Vec<LoadedArtifact> = match &args.only {
            None => all,
            Some(ids) => {
                if let Some(unknown) = ids.iter().find(|id| !all.iter().any(|a| &a.id.0 == *id)) {
                    eprintln!("unknown artifact id in --only: {unknown}");
                    return ExitCode::from(2);
                }
                all.into_iter().filter(|a| ids.contains(&a.id.0)).collect()
            }
        };
        let artifact_count = artifacts.len();

        // A run that stalls (window occluded or minimized for good) must not hang.
        let budget_secs =
            artifact_count as f64 * (args.secs + args.warmup_secs + WATCHDOG_SLACK_SECS) + 30.0;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(budget_secs));
            eprintln!(
            "!!! live_harness watchdog: run exceeded {budget_secs:.0} s (window paused?), aborting"
        );
            std::process::exit(3);
        });

        let report = Rc::new(RefCell::new(Report::default()));
        let errors = Arc::new(AtomicU64::new(0));
        let callback = LiveHarness {
            artifacts,
            current: 0,
            runtime: None,
            phase: Phase::Warmup {
                until: Instant::now(),
            },
            renderer: None,
            overlay: args.overlay.then(Overlay::new),
            analyzer: Analyzer::new(),
            signal: SynthSignal::new(SIGNAL_SEED),
            push_buf: vec![0.0; MAX_PUSH_SAMPLES],
            audio_start: None,
            samples_pushed: 0,
            feature: FeatureFrame::default(),
            prev_t: None,
            secs: Duration::from_secs_f64(args.secs),
            warmup: Duration::from_secs_f64(args.warmup_secs),
            errors: errors.clone(),
            report: report.clone(),
        };

        let (width_pt, height_pt) = (args.width_pt as f64, args.height_pt as f64);
        let (secs, warmup_secs, overlay) = (args.secs, args.warmup_secs, args.overlay);
        let app = App::new(callback).on_ready(move |gpu, window, harness| {
        let counter = errors.clone();
        gpu.device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            counter.fetch_add(1, Ordering::Relaxed);
            eprintln!("!!! WGPU ERROR: {e}");
        }));

        // Primary display (the Retina built-in on a laptop), so the default size is
        // 3024x1890 px regardless of where the terminal is.
        let monitor = window.primary_monitor().or_else(|| window.current_monitor());
        if let Some(mon) = &monitor {
            let origin = mon.position().to_logical::<f64>(mon.scale_factor());
            window.set_outer_position(winit::dpi::LogicalPosition::new(
                origin.x + WINDOW_OFFSET_PT,
                origin.y + WINDOW_OFFSET_PT,
            ));
        }
        let _ = window.request_inner_size(winit::dpi::LogicalSize::new(width_pt, height_pt));
        window.focus_window();

        let mut renderer = Renderer::new(&gpu.device, &gpu.queue, gpu.format);
        let size = window.inner_size();
        renderer.resize(&gpu.device, size.width, size.height);
        harness.renderer = Some(renderer);
        if let Some(overlay) = harness.overlay.as_mut() {
            overlay.attach(&gpu.device, window, gpu.format);
        }

        println!(
            "live_harness: format={:?} present_mode={:?} max_frame_latency={} window={}x{}pt monitor=\"{}\" refresh={}Hz warmup={warmup_secs}s secs={secs}s overlay={overlay} artifacts={artifact_count}",
            gpu.format,
            gpu.config.present_mode,
            gpu.config.desired_maximum_frame_latency,
            width_pt,
            height_pt,
            monitor
                .as_ref()
                .and_then(|m| m.name())
                .unwrap_or_else(|| "?".to_owned()),
            monitor
                .as_ref()
                .and_then(|m| m.refresh_rate_millihertz())
                .map_or(0.0, |mhz| mhz as f64 / 1000.0),
        );
        println!();
        print_header();
    });

        let event_loop = match EventLoop::new() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("cannot create event loop: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut harness_app = HarnessApp {
            app,
            report: report.clone(),
        };
        let run = event_loop.run_app(&mut harness_app);
        drop(harness_app);

        let report = report.borrow();
        let results = &report.results;
        let wgpu_errors: u64 = results.iter().map(|r| r.errors).sum();
        let errors = wgpu_errors + report.panics;
        let complete = report.done && results.len() + report.panics as usize == artifact_count;
        let (footprint_end, footprint_peak) = phys_footprint_bytes();
        let gpu_pcts: Vec<f64> = results.iter().filter_map(|r| r.gpu_pct).collect();
        let summary = json!({
            "cpu_pct_mean": mean(results.iter().map(|r| r.cpu_pct)),
            "gpu_pct_mean": if gpu_pcts.len() == results.len() { mean(gpu_pcts.iter().copied()) } else { f64::NAN },
            "fps_mean": mean(results.iter().map(|r| r.frames as f64 / r.wall_secs)),
            "max_gap_ms": results.iter().map(|r| r.max_gap_ms).fold(0.0, f64::max),
            "footprint_mean_mb": mean(results.iter().map(|r| r.footprint_mb)),
            "footprint_end_mb": mb(footprint_end),
            "footprint_peak_mb": mb(footprint_peak),
            "surface": results.first().map(|r| [r.surface.0, r.surface.1]),
            "artifacts": results.len(),
            "frames": results.iter().map(|r| r.frames).sum::<u64>(),
            "errors": errors,
            "complete": complete,
            "overlay": overlay,
        });

        println!();
        println!(
            "footprint end={:.1} MB peak={:.1} MB; panics={}",
            mb(footprint_end),
            mb(footprint_peak),
            report.panics
        );
        if let Err(e) = &run {
            eprintln!("event loop error: {e}");
        }
        if !complete {
            println!("!!!!!!!! RUN INCOMPLETE (window closed early?) !!!!!!!!");
        }
        if errors > 0 {
            let banner = format!(
                "!!!!!!!! {errors} ERRORS (wgpu errors + panics) — RUN IS INVALID !!!!!!!!"
            );
            eprintln!("{banner}");
            println!("{banner}");
        }
        println!("{summary}");
        if errors > 0 || !complete || run.is_err() {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}
