//! End-to-end headless GPU render test for `retro-spectrum-bars.artifact.json`.
//!
//! Feeds a deterministic musical synthetic signal through [`viz_audio::Analyzer`],
//! evaluates [`ArtifactRuntime`] across all seven 1990s hardware models and key
//! display patterns, renders through [`viz_render::Renderer`] on a headless wgpu
//! device, and asserts zero wgpu validation errors and non-blank rendered frames.
//!
//! Set `VIEWMUSIC_DUMP_RETRO_DIR=<dir>` to also save RGBA8 PNG snapshots for
//! visual inspection (no files are written during normal `cargo test` runs).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use viz_app::runtime::{ArtifactRuntime, SettingValue};
use viz_audio::Analyzer;
use viz_contract::load_artifact;
use viz_core::{FeatureFrame, SAMPLE_RATE};
use viz_render::{Renderer, SceneParams};

const W: u32 = 1280;
const H: u32 = 720;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

fn make_device() -> (wgpu::Device, wgpu::Queue) {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL | wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(desc);

    let force_fallback = std::env::var("VIEWMUSIC_FORCE_FALLBACK_ADAPTER").as_deref() == Ok("1");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: force_fallback,
        compatible_surface: None,
    }))
    .expect("no GPU adapter for headless retro_render test");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("retro-render-device"),
        required_features: viz_render::device_features(&adapter),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("failed to acquire wgpu device");

    device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
        panic!("wgpu validation error: {e}");
    }));
    (device, queue)
}

/// Deterministic broadband musical snippet (kick + bass + chord + hi-hat) so
/// every frequency band and peak-hold cap has lively, realistic energy.
fn synth_frames(frame_count: usize) -> Vec<FeatureFrame> {
    use std::f64::consts::TAU;
    let samples_per_frame = (SAMPLE_RATE / 60) as usize;
    let mut analyzer = Analyzer::new();
    let mut latest = FeatureFrame::default();
    let mut frames = Vec::with_capacity(frame_count);
    let mut lcg: u64 = 0x517c_c1b7_2722_0a95;
    let mut sample_idx: usize = 0;

    for _ in 0..frame_count {
        let mut block = vec![0.0f32; samples_per_frame];
        for s in &mut block {
            let t = sample_idx as f64 / SAMPLE_RATE as f64;
            let beat_phase = (t * 2.0).fract();
            let kick = (TAU * (52.0 + 110.0 * (-beat_phase * 18.0).exp()) * t).sin()
                * (-beat_phase * 8.0).exp()
                * 0.55;
            let bass = ((TAU * 110.0 * t).sin() + 0.5 * (TAU * 220.0 * t).sin()) * 0.25;
            let mid = ((TAU * 440.0 * t).sin()
                + 0.7 * (TAU * 659.25 * t).sin()
                + 0.5 * (TAU * 1318.5 * t).sin())
                * (0.14 + 0.08 * (TAU * 0.5 * t).sin());
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (((lcg >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0)
                * (-((t * 4.0).fract()) * 24.0).exp()
                * 0.28;
            let hi = (TAU * 4200.0 * t).sin() * 0.10 + (TAU * 8600.0 * t).sin() * 0.08 + noise;
            *s = (kick + bass + mid + hi).clamp(-1.0, 1.0) as f32;
            sample_idx += 1;
        }
        analyzer.push_samples(&block, |f, _| latest = *f);
        frames.push(latest);
    }
    frames
}

fn readback_rgba(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let bytes_per_row = (W * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("retro-readback"),
        size: u64::from(bytes_per_row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("retro-copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        let row = &mapped[(y * bytes_per_row) as usize..(y * bytes_per_row + W * 4) as usize];
        for px in row.chunks_exact(4) {
            // BGRA8 -> RGBA8
            rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
        }
    }
    drop(mapped);
    buf.unmap();
    rgba
}

fn save_png(path: &Path, rgba: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(rgba).expect("png data");
}

#[test]
fn retro_spectrum_bars_renders_all_models_and_patterns_headlessly() {
    let json = include_str!("../../../assets/builtin-artifacts/retro-spectrum-bars.artifact.json");
    let loaded = load_artifact("retro-spectrum-bars", json.as_bytes())
        .expect("retro-spectrum-bars must load");

    let (device, queue) = make_device();
    let mut renderer = Renderer::new(&device, &queue, FORMAT);
    renderer.resize(&device, W, H);

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("retro-target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let frames = synth_frames(90);
    let last_frame = frames.last().unwrap();

    let dump_dir: Option<PathBuf> = std::env::var_os("VIEWMUSIC_DUMP_RETRO_DIR").map(PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).expect("create dump dir");
    }

    // (filename stem, model index, pattern index, housing, bloom)
    let cases: &[(&str, f64, f64, f64, f64)] = &[
        ("00-sony-mhc-rx90", 0.0, 0.0, 1.0, 0.85),
        ("01-aiwa-hs-px1000-daylight", 1.0, 0.0, 1.0, 0.0),
        ("01-aiwa-hs-px1000-el-backlight", 1.0, 0.0, 1.0, 1.15),
        ("02-pioneer-carrozzeria", 2.0, 0.0, 1.0, 0.85),
        ("03-technics-sh-ge70", 3.0, 0.0, 1.0, 0.85),
        ("04-aiwa-nsx-jvc-mx", 4.0, 0.0, 1.0, 0.85),
        ("05-kenwood-ge-7030", 5.0, 0.0, 1.0, 0.85),
        ("06-sansui-rack-led", 6.0, 0.0, 1.0, 0.85),
        ("07-sony-floating-crest", 0.0, 1.0, 1.0, 0.85),
        ("08-pioneer-center-mirror", 2.0, 2.0, 1.0, 0.85),
        ("09-sony-transparent-overlay", 0.0, 0.0, 0.0, 0.85),
    ];

    let mut rt = ArtifactRuntime::new(loaded);
    rt.activate();

    let aspect = W as f32 / H as f32;
    for &(name, model, pattern, housing, bloom) in cases {
        assert!(rt.set_setting("model", SettingValue::Scalar(model)));
        assert!(rt.set_setting("pattern", SettingValue::Scalar(pattern)));
        assert!(rt.set_setting("housing", SettingValue::Scalar(housing)));
        assert!(rt.set_setting("bloom", SettingValue::Scalar(bloom)));

        let scene = rt.frame(last_frame, 1.0 / 60.0, aspect, 1.0);
        let params = SceneParams {
            feedback_decay: scene.params.feedback_decay,
            transparent_bg: housing < 0.5,
        };
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("retro-pass"),
        });
        renderer.render(&device, &queue, &mut enc, &view, &params, &scene.layers);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        renderer.recall_staging();

        let rgba = readback_rgba(&device, &queue, &tex);
        let bright_pixels = rgba
            .chunks_exact(4)
            .filter(|px| px[0] > 40 || px[1] > 40 || px[2] > 40)
            .count();
        assert!(
            bright_pixels > 1000,
            "{name}: rendered frame unexpectedly dark ({bright_pixels} bright pixels)"
        );

        if let Some(dir) = &dump_dir {
            save_png(&dir.join(format!("{name}.png")), &rgba);
        }
    }
}
