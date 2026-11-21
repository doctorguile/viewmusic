//! End-to-end headless GPU render test for `winamp-spectrum-bars.artifact.json`.
//!
//! Feeds a deterministic musical synthetic signal through [`viz_audio::Analyzer`],
//! evaluates [`ArtifactRuntime`] across all seven classic `.wsz` Winamp skins and
//! key display modes (Classic Bars, Segmented CRT, Fire, Peaks Only, and
//! Transparent overlay), renders through [`viz_render::Renderer`] on a headless
//! wgpu device, and asserts zero wgpu validation errors and non-blank rendered
//! frames.
//!
//! Set `VIEWMUSIC_DUMP_WINAMP_DIR=<dir>` to also save RGBA8 PNG snapshots for
//! visual inspection (no files are written during normal `cargo test` runs).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use viz_app::runtime::{ArtifactRuntime, SettingValue};
use viz_audio::Analyzer;
use viz_contract::{load_artifact, LayerPrograms};
use viz_core::{FeatureFrame, SAMPLE_RATE};
use viz_render::{LayerDraw, Renderer, SceneParams, ShapeKind};

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
    .expect("no GPU adapter for headless winamp_render test");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("winamp-render-device"),
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
        label: Some("winamp-readback"),
        size: u64::from(bytes_per_row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("winamp-copy"),
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
fn winamp_spectrum_bars_renders_all_skins_and_modes_headlessly() {
    let json = include_str!("../../../assets/builtin-artifacts/winamp-spectrum-bars.artifact.json");
    let loaded = load_artifact("winamp-spectrum-bars", json.as_bytes())
        .expect("winamp-spectrum-bars must load");

    assert_eq!(loaded.artifact.contract, "1.1");
    assert!(loaded.artifact.feedback.is_none());
    assert!(loaded.artifact.scene.len() <= 16);
    assert!(loaded.artifact.vars.len() <= 64);

    let mut max_formula_ops = 0usize;
    let mut total_ops = 0usize;
    let mut visit = |f: &viz_contract::CompiledFormula| {
        let ops = f.op_count();
        max_formula_ops = max_formula_ops.max(ops);
        total_ops += ops;
    };
    for f in &loaded.programs.vars_init {
        visit(f);
    }
    for f in &loaded.programs.vars_frame {
        visit(f);
    }
    for layer in &loaded.programs.layers {
        match layer {
            LayerPrograms::Instanced {
                count,
                visible,
                x,
                y,
                w,
                h,
                rot,
                color,
            } => {
                visit(count);
                if let Some(v) = visible {
                    visit(v);
                }
                for f in [x, y, w, h, &color.c0, &color.c1, &color.c2] {
                    visit(f);
                }
                if let Some(a) = &color.a {
                    visit(a);
                }
                if let Some(r) = rot {
                    visit(r);
                }
            }
            LayerPrograms::Polyline {
                points,
                thickness,
                visible,
                x,
                y,
                color,
                ..
            } => {
                visit(points);
                visit(thickness);
                if let Some(v) = visible {
                    visit(v);
                }
                for f in [x, y, &color.c0, &color.c1, &color.c2] {
                    visit(f);
                }
                if let Some(a) = &color.a {
                    visit(a);
                }
            }
            LayerPrograms::Field {
                resolution,
                visible,
                color,
            } => {
                visit(resolution);
                if let Some(v) = visible {
                    visit(v);
                }
                for f in [&color.c0, &color.c1, &color.c2] {
                    visit(f);
                }
                if let Some(a) = &color.a {
                    visit(a);
                }
            }
        }
    }
    println!(
        "winamp-spectrum-bars: layers={}, vars={}, max_formula_ops={max_formula_ops}/256, total_ops={total_ops}/16384",
        loaded.artifact.scene.len(),
        loaded.artifact.vars.len()
    );
    assert!(
        max_formula_ops <= 256 && total_ops <= 16_384,
        "op counts exceeded limits: max_formula_ops={max_formula_ops}, total_ops={total_ops}"
    );

    let (device, queue) = make_device();
    let mut renderer = Renderer::new(&device, &queue, FORMAT);
    renderer.resize(&device, W, H);

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("winamp-target"),
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

    let dump_dir: Option<PathBuf> =
        std::env::var_os("VIEWMUSIC_DUMP_WINAMP_DIR").map(PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).expect("create dump dir");
    }

    // (filename stem, skin index, mode index, housing, showGrid)
    let cases: &[(&str, f64, f64, f64, f64)] = &[
        ("00-winamp-classic-291", 0.0, 0.0, 1.0, 1.0),
        ("01-green-dimension-v2", 1.0, 0.0, 1.0, 1.0),
        ("02-topazamp", 2.0, 0.0, 1.0, 1.0),
        ("03-macosx-aqua", 3.0, 0.0, 1.0, 1.0),
        ("04-xmms-turquoise", 4.0, 0.0, 1.0, 1.0),
        ("05-zaxon-remake", 5.0, 0.0, 1.0, 1.0),
        ("06-vizor-bronze", 6.0, 0.0, 1.0, 1.0),
        ("07-winamp-fire-mode", 0.0, 2.0, 1.0, 1.0),
        ("08-winamp-segmented-crt", 0.0, 1.0, 1.0, 1.0),
        ("09-winamp-peaks-only", 0.0, 3.0, 1.0, 1.0),
        ("10-winamp-transparent-overlay", 0.0, 0.0, 0.0, 0.0),
    ];

    let mut rt = ArtifactRuntime::new(loaded);
    rt.activate();

    let aspect = W as f32 / H as f32;
    for &(name, skin, mode, housing, show_grid) in cases {
        assert!(rt.set_setting("skin", SettingValue::Scalar(skin)));
        assert!(rt.set_setting("mode", SettingValue::Scalar(mode)));
        assert!(rt.set_setting("housing", SettingValue::Scalar(housing)));
        assert!(rt.set_setting("showGrid", SettingValue::Scalar(show_grid)));

        let scene = rt.frame(last_frame, 1.0 / 60.0, aspect, 1.0);

        // Light skin (Mac OS X Aqua, skin 3) must not emit additive Glow layers.
        if (skin - 3.0).abs() < 0.1 {
            for layer in &scene.layers {
                assert!(
                    !matches!(
                        layer,
                        LayerDraw::Instanced {
                            shape: ShapeKind::Glow,
                            ..
                        }
                    ),
                    "{name}: Mac OS X Aqua light skin must not emit additive Glow layers"
                );
            }
        }

        let params = SceneParams {
            feedback_decay: scene.params.feedback_decay,
            transparent_bg: housing < 0.5,
        };
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("winamp-pass"),
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

    // Verify calm on silence: with housing=false and showGrid=false, a silent
    // frame must produce zero visible pixels.
    assert!(rt.set_setting("skin", SettingValue::Scalar(0.0)));
    assert!(rt.set_setting("mode", SettingValue::Scalar(0.0)));
    assert!(rt.set_setting("housing", SettingValue::Scalar(0.0)));
    assert!(rt.set_setting("showGrid", SettingValue::Scalar(0.0)));
    let silent = FeatureFrame::default();
    let silent_scene = rt.frame(&silent, 1.0 / 60.0, aspect, 1.0);
    let silent_params = SceneParams {
        feedback_decay: silent_scene.params.feedback_decay,
        transparent_bg: true,
    };
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("winamp-silent-pass"),
    });
    renderer.render(
        &device,
        &queue,
        &mut enc,
        &view,
        &silent_params,
        &silent_scene.layers,
    );
    queue.submit([enc.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    renderer.recall_staging();
    let silent_rgba = readback_rgba(&device, &queue, &tex);
    let lit_on_silence = silent_rgba
        .chunks_exact(4)
        .filter(|px| px[3] > 2 || px[0] > 2 || px[1] > 2 || px[2] > 2)
        .count();
    assert_eq!(
        lit_on_silence, 0,
        "bars/peaks must vanish completely on silence (found {lit_on_silence} non-empty pixels)"
    );
}
