//! Frozen, headless performance harness for ViewMusic (the hillclimb "ruler").
//!
//! Exercises the app's steady-state per-frame path minus window/present/egui/Core
//! Audio: a deterministic synthetic 48 kHz signal is pushed through the real
//! [`viz_audio::Analyzer`] (800 samples = 1/60 s per frame), the latest
//! [`FeatureFrame`] drives [`ArtifactRuntime::frame`], and [`Renderer::render`]
//! records into a fresh encoder targeting an offscreen `Bgra8UnormSrgb` texture.
//! Every frame is submitted and waited on (`device.poll(Wait)`) so GPU work does
//! not pile up across frames.
//!
//! Run (frozen defaults): `scripts/perf.sh`, or
//! `nix develop -c cargo run --release -p viz-app --example perf_harness -- [flags]`.
//!
//! Flags:
//! - `--size WxH` (default 3024x1890) offscreen target size in physical pixels.
//! - `--only id,id` restrict to these built-in artifact ids.
//! - `--frames N` (default 600) measured frames per artifact (after 120 warmup frames).
//! - `--repeat R` (default 5) measured sweeps after one global warmup sweep; per
//!   artifact the median over sweeps is reported, with min/max.
//! - `--json PATH` write the full per-artifact result.
//! - `--gpu-source auto|ioreg|timestamp` (default auto = ioreg, falling back to wgpu
//!   timestamp queries when this process has no `AGXDeviceUserClient` usage entry).
//! - `--dump DIR` correctness mode: render frames 1..=600 from activation per artifact
//!   and write `DIR/<artifact-id>-f<frame>.png` (RGBA8, sRGB) at frames 30, 240, 600.
//! - `--compare DIR_A DIR_B` compare two dump dirs; exits non-zero if any image has
//!   more than 0.5% pixels with any channel differing by more than 2.
//!
//! The last stdout line of a measurement run is a machine-parseable JSON summary.

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("perf_harness is macOS-only");
    std::process::ExitCode::FAILURE
}

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    harness::main()
}

/// The harness itself. macOS-only: it reads process and GPU counters through
/// macOS APIs (proc_pid_rusage, QoS classes, ioreg AGX counters).
#[cfg(target_os = "macos")]
mod harness {
    use std::fs;
    use std::io::{BufReader, BufWriter};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::process::{Command, ExitCode};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    use serde_json::json;
    use viz_app::runtime::ArtifactRuntime;
    use viz_audio::Analyzer;
    use viz_contract::{ArtifactLibrary, LoadedArtifact};
    use viz_core::{FeatureFrame, SAMPLE_RATE};
    use viz_render::{LayerDraw, Renderer};

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

    /// The format `GpuContext::new` (window.rs) ends up with on macOS: the surface's
    /// default config takes the first capability format, and wgpu-core sorts sRGB
    /// formats first, so Metal's list yields `Bgra8UnormSrgb`.
    const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
    const DEFAULT_WIDTH: u32 = 3024;
    const DEFAULT_HEIGHT: u32 = 1890;
    const DEFAULT_FRAMES: usize = 600;
    const DEFAULT_REPEAT: usize = 5;
    const WARMUP_FRAMES: usize = 120;
    /// Audio samples fed to the analyzer per rendered frame (1/60 s at 48 kHz).
    const SAMPLES_PER_FRAME: usize = 800;
    /// Same clamp as `MAX_DT` in `crates/viz-app/src/viz.rs` (`VizApp::poll_audio`).
    const MAX_DT: f64 = 0.1;
    /// Fixed synthetic-signal seed (noise hi-hats); no wall clock anywhere.
    const SIGNAL_SEED: u64 = 0x5EED_0FA0_D10C;
    /// Frames (counted from activation, 1-based) read back in `--dump` mode.
    const DUMP_FRAMES: [usize; 3] = [30, 240, 600];
    /// `--compare` threshold: per-channel abs difference counted as a changed pixel.
    const COMPARE_CHANNEL_TOLERANCE: u8 = 2;
    /// `--compare` failure threshold: percent of changed pixels per image.
    const COMPARE_MAX_CHANGED_PCT: f64 = 0.5;

    // ---------------------------------------------------------------------------
    // CLI
    // ---------------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GpuSourceArg {
        Auto,
        Ioreg,
        Timestamp,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GpuSource {
        Ioreg,
        Timestamp,
    }

    impl GpuSource {
        fn name(self) -> &'static str {
            match self {
                GpuSource::Ioreg => "ioreg",
                GpuSource::Timestamp => "timestamp",
            }
        }
    }

    struct Args {
        width: u32,
        height: u32,
        only: Option<Vec<String>>,
        frames: usize,
        repeat: usize,
        json: Option<PathBuf>,
        dump: Option<PathBuf>,
        compare: Option<(PathBuf, PathBuf)>,
        gpu_source: GpuSourceArg,
    }

    const USAGE: &str =
        "usage: perf_harness [--size WxH] [--only id,id] [--frames N] [--repeat R] \
[--json PATH] [--gpu-source auto|ioreg|timestamp] [--dump DIR] [--compare DIR_A DIR_B]";

    fn parse_args() -> Result<Args, String> {
        let mut args = Args {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            only: None,
            frames: DEFAULT_FRAMES,
            repeat: DEFAULT_REPEAT,
            json: None,
            dump: None,
            compare: None,
            gpu_source: GpuSourceArg::Auto,
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
                    args.width = w.parse().map_err(|_| format!("bad width in {v}"))?;
                    args.height = h.parse().map_err(|_| format!("bad height in {v}"))?;
                    if args.width == 0 || args.height == 0 {
                        return Err("--size must be non-zero".into());
                    }
                }
                "--only" => {
                    let v = value("--only")?;
                    args.only = Some(v.split(',').map(|s| s.trim().to_owned()).collect());
                }
                "--frames" => {
                    let v = value("--frames")?;
                    args.frames = v.parse().map_err(|_| format!("bad --frames {v}"))?;
                    if args.frames == 0 {
                        return Err("--frames must be > 0".into());
                    }
                }
                "--repeat" => {
                    let v = value("--repeat")?;
                    args.repeat = v.parse().map_err(|_| format!("bad --repeat {v}"))?;
                    if args.repeat == 0 {
                        return Err("--repeat must be > 0".into());
                    }
                }
                "--json" => args.json = Some(PathBuf::from(value("--json")?)),
                "--dump" => args.dump = Some(PathBuf::from(value("--dump")?)),
                "--compare" => {
                    let a = PathBuf::from(value("--compare")?);
                    let b = PathBuf::from(value("--compare")?);
                    args.compare = Some((a, b));
                }
                "--gpu-source" => {
                    let v = value("--gpu-source")?;
                    args.gpu_source = match v.as_str() {
                        "auto" => GpuSourceArg::Auto,
                        "ioreg" => GpuSourceArg::Ioreg,
                        "timestamp" => GpuSourceArg::Timestamp,
                        _ => return Err(format!("bad --gpu-source {v}")),
                    };
                }
                "-h" | "--help" => return Err(String::new()),
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(args)
    }

    // ---------------------------------------------------------------------------
    // Synthetic music-like signal
    // ---------------------------------------------------------------------------

    /// Generates `len` samples of a deterministic 120 BPM mono "track": a pitch-swept
    /// kick on every beat, a 3-harmonic bass line (one note per beat), a sustained
    /// chord pad changing every bar, and high-passed noise hi-hats on every 8th note.
    fn synth_signal(len: usize, seed: u64) -> Vec<f32> {
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
        let mut rng = seed | 1;
        let mut prev_white = 0.0f64;
        let sr = SAMPLE_RATE as f64;
        (0..len)
            .map(|n| {
                let t = n as f64 / sr;
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
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let white = (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                let hp = white - prev_white;
                prev_white = white;
                let eighth = (t / EIGHTH_SECS).floor();
                let t8 = t - eighth * EIGHTH_SECS;
                let accent = if (eighth as u64) % 2 == 1 { 1.0 } else { 0.6 };
                let hat = hp * (-t8 / 0.03).exp() * 0.15 * accent;

                ((kick + bass + pad + hat) * 0.7).clamp(-1.0, 1.0) as f32
            })
            .collect()
    }

    // ---------------------------------------------------------------------------
    // Process metrics
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

    fn load_average() -> [f64; 3] {
        let mut avg = [0.0f64; 3];
        // SAFETY: getloadavg writes up to 3 doubles into the provided buffer.
        unsafe {
            libc::getloadavg(avg.as_mut_ptr(), 3);
        }
        avg
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
        /// Every other process's clients (background GPU load, informational).
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

    // ---------------------------------------------------------------------------
    // GPU
    // ---------------------------------------------------------------------------

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
        target: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        /// Count of uncaptured wgpu errors (validation / OOM / internal).
        errors: Arc<AtomicU64>,
    }

    impl Gpu {
        /// Headless Metal device with the app's limits (`GpuContext::new`): no surface.
        fn new(width: u32, height: u32, features: wgpu::Features) -> Result<Self, String> {
            let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
            instance_desc.backends = wgpu::Backends::METAL;
            let instance = wgpu::Instance::new(instance_desc);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))
                .map_err(|e| format!("no Metal adapter: {e}"))?;
            let adapter_info = adapter.get_info();
            if adapter_info.backend != wgpu::Backend::Metal {
                return Err(format!(
                    "adapter backend is {:?}, not Metal",
                    adapter_info.backend
                ));
            }
            if !adapter.features().contains(features) {
                return Err(format!("adapter lacks features {features:?}"));
            }
            let required_limits =
                wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("perf-harness-device"),
                    required_features: features | viz_render::device_features(&adapter),
                    required_limits,
                    ..Default::default()
                }))
                .map_err(|e| format!("request_device failed: {e}"))?;

            let errors = Arc::new(AtomicU64::new(0));
            let counter = errors.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                counter.fetch_add(1, Ordering::Relaxed);
                eprintln!("!!! WGPU ERROR: {e}");
            }));

            let max_dim = device.limits().max_texture_dimension_2d;
            if width > max_dim || height > max_dim {
                return Err(format!("--size exceeds max texture dimension {max_dim}"));
            }
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("perf-harness-target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            Ok(Self {
                device,
                queue,
                adapter_info,
                target,
                view,
                width,
                height,
                errors,
            })
        }

        fn error_count(&self) -> u64 {
            self.errors.load(Ordering::Relaxed)
        }

        fn wait(&self, submission: wgpu::SubmissionIndex) {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .expect("device.poll(Wait) failed");
        }

        /// Submits one clear of the target and waits (used to probe ioreg usage).
        fn submit_clear(&self) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("perf-harness-probe"),
                });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("perf-harness-probe-clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            let idx = self.queue.submit(Some(encoder.finish()));
            self.wait(idx);
        }
    }

    /// Fallback GPU timer: a one-invocation no-op compute pass writes a timestamp at the
    /// start and at the end of every measured frame's encoder (Metal drops stage-boundary
    /// samples on an empty encoder, hence the dispatch); the window total is the sum of
    /// `(end - start)` resolved once after the window.
    struct TimestampTimer {
        marker_pipeline: wgpu::ComputePipeline,
        query_set: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        readback: wgpu::Buffer,
    }

    impl TimestampTimer {
        fn new(device: &wgpu::Device, max_frames: u32) -> Self {
            let count = max_frames * 2;
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("perf-harness-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let size = count as u64 * 8;
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("perf-harness-ts-resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("perf-harness-ts-readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("perf-harness-ts-marker"),
                source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
            });
            let marker_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("perf-harness-ts-marker"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            Self {
                marker_pipeline,
                query_set,
                resolve,
                readback,
            }
        }

        fn mark(&self, encoder: &mut wgpu::CommandEncoder, frame: u32, end: bool) {
            let index = frame * 2 + u32::from(end);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("perf-harness-ts-mark"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.query_set,
                    beginning_of_pass_write_index: (!end).then_some(index),
                    end_of_pass_write_index: end.then_some(index),
                }),
            });
            pass.set_pipeline(&self.marker_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }

        /// Total GPU nanoseconds between the start/end marks of frames `0..frames`.
        fn total_ns(&self, gpu: &Gpu, frames: u32) -> u64 {
            let count = frames * 2;
            let bytes = count as u64 * 8;
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("perf-harness-ts-resolve"),
                });
            encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, bytes);
            gpu.wait(gpu.queue.submit(Some(encoder.finish())));
            let slice = self.readback.slice(0..bytes);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("timestamp map failed"));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll failed");
            let period = gpu.queue.get_timestamp_period() as f64;
            let total_ticks: u64 = {
                let data = slice.get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                ticks
                    .chunks_exact(2)
                    .map(|p| p[1].saturating_sub(p[0]))
                    .sum()
            };
            self.readback.unmap();
            (total_ticks as f64 * period) as u64
        }
    }

    // ---------------------------------------------------------------------------
    // One artifact session: audio → runtime → renderer, per frame
    // ---------------------------------------------------------------------------

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

    /// Fresh runtime + renderer + analyzer for one artifact, activated like the app
    /// (`VizApp::attach_gpu`: `Renderer::new`, `resize`, then `activate`).
    struct Session<'a> {
        gpu: &'a Gpu,
        signal: &'a [f32],
        runtime: ArtifactRuntime,
        renderer: Renderer,
        analyzer: Analyzer,
        feature: FeatureFrame,
        prev_t: Option<f64>,
        /// Frames rendered since activation.
        frame_index: usize,
        aspect: f32,
    }

    impl<'a> Session<'a> {
        fn new(gpu: &'a Gpu, signal: &'a [f32], artifact: &LoadedArtifact) -> Self {
            let mut renderer = Renderer::new(&gpu.device, &gpu.queue, TARGET_FORMAT);
            renderer.resize(&gpu.device, gpu.width, gpu.height);
            let mut runtime = ArtifactRuntime::new(artifact.clone());
            runtime.activate();
            Self {
                gpu,
                signal,
                runtime,
                renderer,
                analyzer: Analyzer::new(),
                feature: FeatureFrame::default(),
                prev_t: None,
                frame_index: 0,
                aspect: gpu.width as f32 / gpu.height as f32,
            }
        }

        /// Renders one frame and waits for its submission to finish on the GPU.
        fn step(
            &mut self,
            counts: Option<&mut WorkCounts>,
            timestamps: Option<(&TimestampTimer, u32)>,
            readback: Option<&ReadbackBuffer>,
        ) {
            // Audio: push 1/60 s through the real analyzer (as the capture thread does)
            // and keep the latest frame; dt from its audio clock as in `poll_audio`.
            let start = self.frame_index * SAMPLES_PER_FRAME;
            let chunk = &self.signal[start..start + SAMPLES_PER_FRAME];
            self.frame_index += 1;
            let feature = &mut self.feature;
            self.analyzer
                .push_samples(chunk, |frame, _beats| *feature = *frame);
            let dt = match self.prev_t {
                None => 0.0,
                Some(prev) => (self.feature.t - prev).clamp(0.0, MAX_DT),
            };
            self.prev_t = Some(self.feature.t);

            let gpu = self.gpu;
            let mut scene = self.runtime.frame(&self.feature, dt, self.aspect, 1.0);
            scene.params.transparent_bg = false;
            if let Some(counts) = counts {
                counts.add(&scene.layers);
            }

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("perf-harness-frame"),
                });
            if let Some((ts, frame)) = timestamps {
                ts.mark(&mut encoder, frame, false);
            }
            self.renderer.render(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &gpu.view,
                &scene.params,
                &scene.layers,
            );
            if let Some((ts, frame)) = timestamps {
                ts.mark(&mut encoder, frame, true);
            }
            if let Some(rb) = readback {
                rb.record_copy(gpu, &mut encoder);
            }
            let submission = gpu.queue.submit(Some(encoder.finish()));
            self.renderer.recall_staging();
            gpu.wait(submission);
        }
    }

    // ---------------------------------------------------------------------------
    // Measurement
    // ---------------------------------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    struct RunResult {
        frames: usize,
        cpu_us_per_frame: f64,
        gpu_us_per_frame: f64,
        wall_us_per_frame: f64,
        /// Other processes' GPU time during the window, as % of window wall time
        /// (ioreg source only; a noisy-run indicator).
        background_gpu_pct: Option<f64>,
        phys_footprint_mb: f64,
        work: WorkCounts,
        errors: u64,
    }

    struct GpuTimer {
        source: GpuSource,
        timestamps: Option<TimestampTimer>,
    }

    impl GpuTimer {
        fn snapshot(&self) -> Option<GpuUsageSnapshot> {
            match self.source {
                GpuSource::Ioreg => GpuUsageSnapshot::read(),
                GpuSource::Timestamp => None,
            }
        }
    }

    /// Warmup + measured window for one artifact.
    fn measure_artifact(
        gpu: &Gpu,
        timer: &GpuTimer,
        signal: &[f32],
        artifact: &LoadedArtifact,
        frames: usize,
    ) -> RunResult {
        let mut session = Session::new(gpu, signal, artifact);
        for _ in 0..WARMUP_FRAMES {
            session.step(None, None, None);
        }

        let errors_before = gpu.error_count();
        let usage_before = timer.snapshot();
        let mut work = WorkCounts::default();
        let cpu_before = process_cpu_us();
        let wall_before = Instant::now();
        for i in 0..frames {
            let ts = timer.timestamps.as_ref().map(|t| (t, i as u32));
            session.step(Some(&mut work), ts, None);
        }
        let wall = wall_before.elapsed();
        let cpu_after = process_cpu_us();
        let (gpu_ns, background_gpu_ns) = match timer.source {
            GpuSource::Ioreg => {
                let delta = match (timer.snapshot(), usage_before) {
                    (Some(after), Some(before)) => after.since(&before),
                    _ => {
                        eprintln!("!!! ioreg GPU usage read failed; GPU time recorded as 0");
                        GpuUsageDelta::default()
                    }
                };
                (delta.ours_ns, Some(delta.other_ns))
            }
            GpuSource::Timestamp => (
                timer
                    .timestamps
                    .as_ref()
                    .expect("timestamp timer")
                    .total_ns(gpu, frames as u32),
                None,
            ),
        };
        let (footprint, _) = phys_footprint_bytes();

        let n = frames as f64;
        RunResult {
            frames,
            cpu_us_per_frame: cpu_after.saturating_sub(cpu_before) as f64 / n,
            gpu_us_per_frame: gpu_ns as f64 / 1000.0 / n,
            wall_us_per_frame: wall.as_secs_f64() * 1e6 / n,
            background_gpu_pct: background_gpu_ns
                .map(|ns| ns as f64 / (wall.as_secs_f64() * 1e9) * 100.0),
            phys_footprint_mb: mb(footprint),
            work,
            errors: gpu.error_count() - errors_before,
        }
    }

    /// Runs `f`, converting a panic into `None` (and logging it loudly).
    fn guarded<T>(id: &str, f: impl FnOnce() -> T) -> Option<T> {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(v) => Some(v),
            Err(_) => {
                eprintln!("!!! PANIC while running artifact {id}");
                None
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Stat {
        median: f64,
        min: f64,
        max: f64,
    }

    fn stat(values: &[f64]) -> Stat {
        if values.is_empty() {
            return Stat {
                median: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let mid = v.len() / 2;
        let median = if v.len() % 2 == 1 {
            v[mid]
        } else {
            (v[mid - 1] + v[mid]) / 2.0
        };
        Stat {
            median,
            min: v[0],
            max: v[v.len() - 1],
        }
    }

    fn stat_json(s: Stat) -> serde_json::Value {
        json!({ "median": s.median, "min": s.min, "max": s.max })
    }

    // ---------------------------------------------------------------------------
    // Dump / compare
    // ---------------------------------------------------------------------------

    /// Mappable copy target for one full frame of the offscreen texture.
    struct ReadbackBuffer {
        buffer: wgpu::Buffer,
        padded_bytes_per_row: u32,
    }

    impl ReadbackBuffer {
        fn new(gpu: &Gpu) -> Self {
            let unpadded = gpu.width * 4;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = unpadded.div_ceil(align) * align;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("perf-harness-readback"),
                size: padded_bytes_per_row as u64 * gpu.height as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Self {
                buffer,
                padded_bytes_per_row,
            }
        }

        fn record_copy(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &gpu.target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.padded_bytes_per_row),
                        rows_per_image: Some(gpu.height),
                    },
                },
                wgpu::Extent3d {
                    width: gpu.width,
                    height: gpu.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        /// Maps the (already copied, already waited-on) buffer and returns tight RGBA8.
        fn read_rgba(&self, gpu: &Gpu) -> Vec<u8> {
            let slice = self.buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("readback map failed"));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll failed");
            let row = (gpu.width * 4) as usize;
            let mut rgba = Vec::with_capacity(row * gpu.height as usize);
            {
                let data = slice.get_mapped_range();
                for y in 0..gpu.height as usize {
                    let start = y * self.padded_bytes_per_row as usize;
                    for px in data[start..start + row].chunks_exact(4) {
                        // Target is BGRA; PNG wants RGBA.
                        rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                }
            }
            self.buffer.unmap();
            rgba
        }
    }

    fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
        let file = fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        writer
            .write_image_data(rgba)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    fn read_png(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
        let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut reader = png::Decoder::new(BufReader::new(file))
            .read_info()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let size = reader
            .output_buffer_size()
            .ok_or_else(|| format!("{}: image too large", path.display()))?;
        let mut buf = vec![0u8; size];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
            return Err(format!("{}: expected RGBA8", path.display()));
        }
        buf.truncate(info.buffer_size());
        Ok((info.width, info.height, buf))
    }

    fn run_dump(gpu: &Gpu, signal: &[f32], artifacts: &[&LoadedArtifact], dir: &Path) -> ExitCode {
        if let Err(e) = fs::create_dir_all(dir) {
            eprintln!("cannot create {}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
        let readback = ReadbackBuffer::new(gpu);
        let last = *DUMP_FRAMES.iter().max().expect("dump frames");
        let mut panics = 0u64;
        let mut written = 0usize;
        for artifact in artifacts {
            let id = artifact.id.0.as_str();
            let result = guarded(id, || -> Result<usize, String> {
                let mut session = Session::new(gpu, signal, artifact);
                let mut count = 0;
                for frame in 1..=last {
                    let dump = DUMP_FRAMES.contains(&frame);
                    session.step(None, None, dump.then_some(&readback));
                    if dump {
                        let rgba = readback.read_rgba(gpu);
                        let path = dir.join(format!("{id}-f{frame}.png"));
                        write_png(&path, gpu.width, gpu.height, &rgba)?;
                        count += 1;
                    }
                }
                Ok(count)
            });
            match result {
                Some(Ok(n)) => {
                    written += n;
                    println!("dumped {id}: {n} frames");
                }
                Some(Err(e)) => {
                    eprintln!("dump failed for {id}: {e}");
                    return ExitCode::FAILURE;
                }
                None => panics += 1,
            }
        }
        let errors = gpu.error_count();
        println!(
            "dump: dir={} images={written} size={}x{} wgpu_errors={errors} panics={panics}",
            dir.display(),
            gpu.width,
            gpu.height
        );
        if errors > 0 || panics > 0 {
            eprintln!("!!!!!!!! DUMP HAD {errors} WGPU ERRORS AND {panics} PANICS !!!!!!!!");
            return ExitCode::FAILURE;
        }
        ExitCode::SUCCESS
    }

    fn png_names(dir: &Path) -> Result<Vec<String>, String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".png"))
            .collect();
        names.sort();
        Ok(names)
    }

    fn run_compare(dir_a: &Path, dir_b: &Path) -> ExitCode {
        let (names_a, names_b) = match (png_names(dir_a), png_names(dir_b)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("compare: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut failed = 0usize;
        let mut compared = 0usize;
        let mut worst_pct = 0.0f64;
        let mut worst_max = 0u8;
        for name in names_b.iter().filter(|n| !names_a.contains(n)) {
            println!("{name}: MISSING in {}", dir_a.display());
            failed += 1;
        }
        for name in &names_a {
            if !names_b.contains(name) {
                println!("{name}: MISSING in {}", dir_b.display());
                failed += 1;
                continue;
            }
            let (a, b) = match (read_png(&dir_a.join(name)), read_png(&dir_b.join(name))) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => {
                    println!("{name}: ERROR {e}");
                    failed += 1;
                    continue;
                }
            };
            if (a.0, a.1) != (b.0, b.1) {
                println!("{name}: SIZE MISMATCH {}x{} vs {}x{}", a.0, a.1, b.0, b.1);
                failed += 1;
                continue;
            }
            let mut max_abs = 0u8;
            let mut changed = 0usize;
            for (pa, pb) in a.2.chunks_exact(4).zip(b.2.chunks_exact(4)) {
                let d = pa
                    .iter()
                    .zip(pb)
                    .map(|(x, y)| x.abs_diff(*y))
                    .max()
                    .unwrap_or(0);
                max_abs = max_abs.max(d);
                if d > COMPARE_CHANNEL_TOLERANCE {
                    changed += 1;
                }
            }
            let pixels = (a.0 as usize * a.1 as usize).max(1);
            let pct = changed as f64 * 100.0 / pixels as f64;
            let bad = pct > COMPARE_MAX_CHANGED_PCT;
            if bad {
                failed += 1;
            }
            compared += 1;
            worst_pct = worst_pct.max(pct);
            worst_max = worst_max.max(max_abs);
            println!(
            "{name}: max_abs_diff={max_abs} pct_pixels_diff_gt{COMPARE_CHANNEL_TOLERANCE}={pct:.4}%{}",
            if bad { "  FAIL" } else { "" }
        );
        }
        println!(
            "{}",
            json!({
                "compared": compared,
                "failed": failed,
                "max_abs_diff": worst_max,
                "worst_pct_pixels_diff_gt2": worst_pct,
                "threshold_pct": COMPARE_MAX_CHANGED_PCT,
            })
        );
        if failed > 0 || compared == 0 {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }

    // ---------------------------------------------------------------------------
    // main
    // ---------------------------------------------------------------------------

    /// Picks the GPU time source and builds the matching device.
    fn setup_gpu(args: &Args) -> Result<(Gpu, GpuTimer), String> {
        let with_timestamps = |frames: usize| -> Result<(Gpu, GpuTimer), String> {
            let max_frames = wgpu::QUERY_SET_MAX_QUERIES / 2;
            if frames > max_frames as usize {
                return Err(format!(
                    "timestamp source supports at most {max_frames} frames"
                ));
            }
            let gpu = Gpu::new(args.width, args.height, wgpu::Features::TIMESTAMP_QUERY)?;
            let timer = TimestampTimer::new(&gpu.device, frames as u32);
            Ok((
                gpu,
                GpuTimer {
                    source: GpuSource::Timestamp,
                    timestamps: Some(timer),
                },
            ))
        };
        if args.gpu_source == GpuSourceArg::Timestamp {
            return with_timestamps(args.frames);
        }
        let gpu = Gpu::new(args.width, args.height, wgpu::Features::empty())?;
        gpu.submit_clear();
        if GpuUsageSnapshot::read().is_some() {
            return Ok((
                gpu,
                GpuTimer {
                    source: GpuSource::Ioreg,
                    timestamps: None,
                },
            ));
        }
        if args.gpu_source == GpuSourceArg::Ioreg {
            return Err("no AGXDeviceUserClient AppUsage entry for this pid in ioreg".into());
        }
        eprintln!("ioreg has no GPU usage entry for this pid; falling back to timestamp queries");
        drop(gpu);
        with_timestamps(args.frames)
    }

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
        if let Some((a, b)) = &args.compare {
            return run_compare(a, b);
        }
        // The app renders on its AppKit main thread, which runs at user-interactive QoS;
        // a CLI main thread does not. Match it so the scheduler keeps this (single,
        // all-work) thread on the same cores the app would get.
        // SAFETY: plain libpthread call on the current thread.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }

        let library = ArtifactLibrary::load_builtins(BUILTIN_ARTIFACTS.iter().copied());
        let all: Vec<&LoadedArtifact> = library
            .entries()
            .iter()
            .filter_map(|e| e.loaded())
            .collect();
        if all.len() != BUILTIN_ARTIFACTS.len() {
            eprintln!(
                "!!! only {} of {} built-in artifacts loaded",
                all.len(),
                BUILTIN_ARTIFACTS.len()
            );
            return ExitCode::FAILURE;
        }
        let artifacts: Vec<&LoadedArtifact> = match &args.only {
            None => all,
            Some(ids) => {
                if let Some(unknown) = ids.iter().find(|id| !all.iter().any(|a| &a.id.0 == *id)) {
                    eprintln!("unknown artifact id in --only: {unknown}");
                    return ExitCode::from(2);
                }
                all.into_iter().filter(|a| ids.contains(&a.id.0)).collect()
            }
        };

        let max_frames =
            (WARMUP_FRAMES + args.frames).max(*DUMP_FRAMES.iter().max().expect("dump"));
        let signal = synth_signal(max_frames * SAMPLES_PER_FRAME, SIGNAL_SEED);

        let (gpu, timer) = match setup_gpu(&args) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("GPU setup failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        if let Some(dir) = &args.dump {
            return run_dump(&gpu, &signal, &artifacts, dir);
        }

        let load_start = load_average();
        println!(
        "perf_harness: adapter=\"{}\" backend={:?} size={}x{} format={:?} warmup={} frames={} repeat={} artifacts={} gpu_source={} loadavg_start={:.2}",
        gpu.adapter_info.name,
        gpu.adapter_info.backend,
        gpu.width,
        gpu.height,
        TARGET_FORMAT,
        WARMUP_FRAMES,
        args.frames,
        args.repeat,
        artifacts.len(),
        timer.source.name(),
        load_start[0],
    );

        let mut panics = 0u64;
        // Global warmup sweep (discarded).
        for artifact in &artifacts {
            let id = artifact.id.0.as_str();
            if guarded(id, || {
                measure_artifact(&gpu, &timer, &signal, artifact, args.frames)
            })
            .is_none()
            {
                panics += 1;
            }
        }
        let mut results: Vec<Vec<RunResult>> = vec![Vec::new(); artifacts.len()];
        for _ in 0..args.repeat {
            for (i, artifact) in artifacts.iter().enumerate() {
                let id = artifact.id.0.as_str();
                match guarded(id, || {
                    measure_artifact(&gpu, &timer, &signal, artifact, args.frames)
                }) {
                    Some(r) => results[i].push(r),
                    None => panics += 1,
                }
            }
        }
        let (footprint_end, footprint_peak) = phys_footprint_bytes();
        let load_end = load_average();
        let errors = gpu.error_count() + panics;

        // Per-artifact aggregation + table.
        println!();
        println!(
            "{:<17} {:>22} {:>22} {:>9} {:>8} {:>6} {:>7} {:>7} {:>7} {:>4}",
            "artifact",
            "cpu us/f med [min-max]",
            "gpu us/f med [min-max]",
            "wall us/f",
            "foot MB",
            "lay/f",
            "inst/f",
            "pts/f",
            "cells/f",
            "err"
        );
        let mut per_artifact = Vec::new();
        let (mut cpu_sum, mut gpu_sum, mut wall_sum) = (0.0, 0.0, 0.0);
        let mut measured_frames = 0usize;
        let mut aggregated = 0usize;
        let mut background_all: Vec<f64> = Vec::new();
        for (artifact, runs) in artifacts.iter().zip(&results) {
            let id = artifact.id.0.as_str();
            let col = |f: fn(&RunResult) -> f64| stat(&runs.iter().map(f).collect::<Vec<_>>());
            let cpu = col(|r| r.cpu_us_per_frame);
            let gpu_t = col(|r| r.gpu_us_per_frame);
            let wall = col(|r| r.wall_us_per_frame);
            let foot = col(|r| r.phys_footprint_mb);
            let background: Vec<f64> = runs.iter().filter_map(|r| r.background_gpu_pct).collect();
            background_all.extend_from_slice(&background);
            let background = stat(&background);
            let per_frame = |f: fn(&WorkCounts) -> u64| {
                runs.first()
                    .map(|r| f(&r.work) as f64 / r.frames as f64)
                    .unwrap_or(0.0)
            };
            let layers = per_frame(|w| w.layers);
            let instances = per_frame(|w| w.instances);
            let points = per_frame(|w| w.points);
            let cells = per_frame(|w| w.cells);
            let run_errors: u64 = runs.iter().map(|r| r.errors).sum();
            let frames: usize = runs.iter().map(|r| r.frames).sum();
            measured_frames += frames;
            if !runs.is_empty() {
                cpu_sum += cpu.median;
                gpu_sum += gpu_t.median;
                wall_sum += wall.median;
                aggregated += 1;
            }
            println!(
            "{:<17} {:>7.1} [{:>5.1}-{:>5.1}] {:>7.1} [{:>5.1}-{:>5.1}] {:>9.1} {:>8.1} {:>6.2} {:>7.0} {:>7.0} {:>7.0} {:>4}",
            id,
            cpu.median,
            cpu.min,
            cpu.max,
            gpu_t.median,
            gpu_t.min,
            gpu_t.max,
            wall.median,
            foot.median,
            layers,
            instances,
            points,
            cells,
            run_errors
        );
            per_artifact.push(json!({
                "id": id,
                "runs": runs.len(),
                "frames_per_run": args.frames,
                "cpu_us_per_frame": stat_json(cpu),
                "gpu_us_per_frame": stat_json(gpu_t),
                "wall_us_per_frame": stat_json(wall),
                "phys_footprint_mb": stat_json(foot),
                "background_gpu_pct": stat_json(background),
                "cpu_us_samples": runs.iter().map(|r| r.cpu_us_per_frame).collect::<Vec<_>>(),
                "gpu_us_samples": runs.iter().map(|r| r.gpu_us_per_frame).collect::<Vec<_>>(),
                "layers_per_frame": layers,
                "instances_per_frame": instances,
                "points_per_frame": points,
                "field_cells_per_frame": cells,
                "wgpu_errors": run_errors,
            }));
        }
        let background = stat(&background_all);
        let denom = aggregated.max(1) as f64;
        let summary = json!({
            "cpu_us_mean": cpu_sum / denom,
            "gpu_us_mean": gpu_sum / denom,
            "wall_us_mean": wall_sum / denom,
            "footprint_end_mb": mb(footprint_end),
            "footprint_peak_mb": mb(footprint_peak),
            "errors": errors,
            "frames": measured_frames,
            "gpu_source": timer.source.name(),
        });

        if let Some(path) = &args.json {
            let full = json!({
                "config": {
                    "adapter": gpu.adapter_info.name,
                    "size": [gpu.width, gpu.height],
                    "format": format!("{TARGET_FORMAT:?}"),
                    "warmup_frames": WARMUP_FRAMES,
                    "frames": args.frames,
                    "repeat": args.repeat,
                    "gpu_source": timer.source.name(),
                    "loadavg_start": load_start,
                    "loadavg_end": load_end,
                },
                "background_gpu_pct": stat_json(background),
                "artifacts": per_artifact,
                "panics": panics,
                "summary": summary,
            });
            let text = serde_json::to_string_pretty(&full).expect("serialize result");
            if let Err(e) = fs::write(path, text) {
                eprintln!("cannot write {}: {e}", path.display());
            }
        }

        println!();
        println!(
            "loadavg start={:.2} end={:.2}; footprint end={:.1} MB peak={:.1} MB; panics={panics}",
            load_start[0],
            load_end[0],
            mb(footprint_end),
            mb(footprint_peak)
        );
        if timer.source == GpuSource::Ioreg {
            println!(
            "background GPU (other processes) during windows: median {:.1}% [{:.1}-{:.1}%] of wall time",
            background.median, background.min, background.max
        );
        }
        if errors > 0 {
            let banner = format!(
                "!!!!!!!! {errors} ERRORS (wgpu errors + panics) — RUN IS INVALID !!!!!!!!"
            );
            eprintln!("{banner}");
            println!("{banner}");
        }
        println!("{summary}");
        if errors > 0 {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}
