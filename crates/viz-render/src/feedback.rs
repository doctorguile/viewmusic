//! Feedback / trails: a persistent offscreen accumulator.
//!
//! When feedback is active each frame runs ONE render pass with two color attachments,
//! the accumulator `curr` and the surface, which receive identical output and blending:
//! 1. **Decay** — draw the previous frame's accumulated image multiplied by `decay`.
//! 2. **Draw** — render this frame's layers on top (handled by [`crate::Renderer`]).
//!
//! The surface therefore ends up equal to `curr` with no copy pass.
//!
//! ## Why ping-pong
//! A pass cannot sample the texture it is rendering to. We keep **two** accumulator
//! textures and alternate roles each active frame: the pass renders into `curr` while
//! sampling `prev` (writing `prev * decay`), the layers then draw into `curr`, and finally
//! the roles swap so next frame's `prev` is this frame's result. Both textures use the
//! surface format and are both sampleable + renderable.
//!
//! ## Lazy allocation, glitch-free resize
//! The textures are allocated by the first active frame, freed after
//! [`RELEASE_AFTER_INACTIVE_FRAMES`] consecutive frames without feedback, and dropped by
//! [`Feedback::resize`] so the next active frame reallocates them at the new size. A fresh
//! pair is always cleared before it is first sampled, so no garbage ever flashes, and the
//! roles are reset deterministically.

use std::num::NonZeroU64;

use crate::upload::{cached_bind_group, CopyTarget, Uploader};
use crate::{PassTargets, CLEAR_COLOR, CLEAR_COLOR_TRANSPARENT};

/// Size of the decay uniform: one f32, padded to 16 bytes (uniform alignment).
const DECAY_BYTES: NonZeroU64 = NonZeroU64::new(16).unwrap();

/// Consecutive feedback-free frames after which the accumulators are freed (~5 s at
/// 60 Hz). Long enough that degrade's feedback shedding does not churn allocations.
const RELEASE_AFTER_INACTIVE_FRAMES: u32 = 300;

/// The two ping-pong textures. Index by the `prev`/`curr` selectors.
struct Accumulators {
    /// Owned so the textures live exactly as long as this pair.
    _textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    /// Bind group that samples `textures[k]` (so `sample_bind[prev]` reads the prev image).
    sample_bind: [wgpu::BindGroup; 2],
}

/// Lazily allocated two-texture ping-pong accumulator + the fullscreen decay pipeline.
///
/// Sizing is owned by [`crate::Renderer`], which reports it via [`Feedback::resize`].
pub struct Feedback {
    /// `None` until the first active frame, after resize, and after a long inactive run.
    accumulators: Option<Accumulators>,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,

    /// Which texture index currently holds the previous frame's accumulation.
    prev: usize,

    sampler: wgpu::Sampler,
    bind_layout: wgpu::BindGroupLayout,
    /// Pipeline that multiplies the sampled texture by a decay uniform, writing both
    /// attachments of the feedback pass (`curr` and the surface).
    decay_pipeline: wgpu::RenderPipeline,
    decay_uniform_layout: wgpu::BindGroupLayout,
    /// Copy destination for the decay uniform; `None` when the draw binds the upload
    /// chunk directly.
    decay_copy_target: Option<CopyTarget>,
    /// Decay uniform bind group per buffer that has held the decay value.
    decay_bind_groups: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// This frame's decay bind group (index into `decay_bind_groups`) and its offset.
    decay_bind: (usize, u32),
    /// Uniform offset alignment, for direct decay uploads.
    uniform_align: NonZeroU64,

    /// Set after (re)allocation or background-mode toggle: both accumulators must be
    /// cleared before they are first sampled. Cleared lazily at the start of the next
    /// active feedback frame (the encoder is available there; no queue needed for a
    /// clear-load render pass).
    dirty: bool,
    /// Tracks the previous `transparent_bg` mode so toggling transparency marks the
    /// accumulators dirty and clears them cleanly to the new background color.
    last_transparent: bool,
    /// Decay value in `decay_copy_target`, so an unchanged decay skips the copy (copy
    /// path only).
    last_decay: Option<f32>,
    /// Consecutive frames rendered without feedback.
    inactive_frames: u32,
}

impl Feedback {
    /// Builds the pipelines, sampler, and the decay copy target unless uploads are
    /// `direct`; the accumulators are allocated at `width × height` by the first active
    /// frame.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        direct: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("feedback-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("feedback.wgsl").into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("feedback-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group 0: sampled texture + sampler.
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("feedback-sample-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Bind group 1 (decay pass only): the decay scalar uniform.
        let decay_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("feedback-decay-uniform-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(DECAY_BYTES),
                    },
                    count: None,
                }],
            });

        let decay_copy_target = (!direct).then(|| {
            CopyTarget::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("feedback-decay-uniform"),
                size: DECAY_BYTES.get(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let uniform_align =
            NonZeroU64::new(device.limits().min_uniform_buffer_offset_alignment.into())
                .expect("non-zero uniform offset alignment");

        let decay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("feedback-decay-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout), Some(&decay_uniform_layout)],
            immediate_size: 0,
        });
        // The fullscreen decay overwrites the whole target → no blending needed.
        let target = wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };

        let decay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("feedback-decay-pipeline"),
            layout: Some(&decay_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_decay"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &PassTargets::FeedbackAndSurface.color_targets(target),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            accumulators: None,
            format,
            width,
            height,
            prev: 0,
            sampler,
            bind_layout,
            decay_pipeline,
            decay_uniform_layout,
            decay_copy_target,
            decay_bind_groups: Vec::new(),
            decay_bind: (0, 0),
            uniform_align,
            dirty: true,
            last_transparent: false,
            last_decay: None,
            inactive_frames: 0,
        }
    }

    /// Allocates the two accumulator textures + their views + sampling bind groups.
    fn allocate(&self, device: &wgpu::Device) -> Accumulators {
        let make_tex = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: self.width.max(1),
                    height: self.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    // COPY_SRC so the GPU smoke test can read this texture back too.
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let textures = [make_tex("feedback-a"), make_tex("feedback-b")];
        let views = [
            textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let make_bind = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("feedback-sample-bind"),
                layout: &self.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let sample_bind = [make_bind(&views[0]), make_bind(&views[1])];
        Accumulators {
            _textures: textures,
            views,
            sample_bind,
        }
    }

    fn accumulators(&self) -> &Accumulators {
        self.accumulators
            .as_ref()
            .expect("prepare_frame allocates the accumulators")
    }

    /// Records the new size and drops the accumulators; the next active frame reallocates
    /// and clears them.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.accumulators = None;
    }

    /// Clears both accumulator textures to `clear_color` if they are dirty
    /// (post-allocation or after toggling transparent background). Records into
    /// the caller's encoder; no queue needed.
    fn clear_if_dirty(&mut self, encoder: &mut wgpu::CommandEncoder, clear_color: wgpu::Color) {
        if !self.dirty {
            return;
        }
        for view in &self.accumulators().views {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback-clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.dirty = false;
    }

    /// Index of the texture currently acting as the render target (`curr`).
    fn curr(&self) -> usize {
        1 - self.prev
    }

    /// The view of the current render target (`curr`) — layers draw here. Call after
    /// [`Feedback::prepare_frame`].
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.accumulators().views[self.curr()]
    }

    /// Records a frame rendered without feedback: the accumulators are cleared before the
    /// next active frame, and freed once feedback has been off for long enough.
    pub fn note_inactive_frame(&mut self) {
        self.dirty = true;
        self.inactive_frames = self.inactive_frames.saturating_add(1);
        if self.inactive_frames >= RELEASE_AFTER_INACTIVE_FRAMES {
            self.accumulators = None;
        }
    }

    /// Prepares the accumulator textures for a merged decay+layers pass: allocates them if
    /// needed, and clears both if freshly allocated, bypassed, or if `transparent_bg`
    /// changed.
    pub fn prepare_frame(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        transparent_bg: bool,
    ) {
        self.inactive_frames = 0;
        if self.accumulators.is_none() {
            self.accumulators = Some(self.allocate(device));
            self.prev = 0;
            self.dirty = true;
        }
        if self.last_transparent != transparent_bg {
            self.last_transparent = transparent_bg;
            self.dirty = true;
        }
        let clear_color = if transparent_bg {
            CLEAR_COLOR_TRANSPARENT
        } else {
            CLEAR_COLOR
        };
        self.clear_if_dirty(encoder, clear_color);
    }

    /// Records the fullscreen decay draw (`curr = prev * decay`) inside an already-open
    /// [`PassTargets::FeedbackAndSurface`] pass whose attachments are
    /// [`Feedback::target_view`] and the surface, so layers composite right on top in the
    /// same pass.
    pub fn draw_decay<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        pass.set_pipeline(&self.decay_pipeline);
        pass.set_bind_group(0, &self.accumulators().sample_bind[self.prev], &[]);
        let (bind_group, offset) = self.decay_bind;
        pass.set_bind_group(1, &self.decay_bind_groups[bind_group].1, &[offset]);
        pass.draw(0..3, 0..1);
    }

    /// Uploads the decay scalar for this frame's [`Feedback::draw_decay`]. The copy path
    /// skips an unchanged value; direct uploads write it every frame into a fresh chunk.
    pub fn set_decay(
        &mut self,
        device: &wgpu::Device,
        upload: &mut Uploader,
        encoder: &mut wgpu::CommandEncoder,
        decay: f32,
    ) {
        let clamped = decay.clamp(0.0, 0.99);
        if let Some(target) = &mut self.decay_copy_target {
            if self.last_decay == Some(clamped) {
                return;
            }
            self.last_decay = Some(clamped);
            target.rewind();
        }
        let v = [clamped, 0.0, 0.0, 0.0];
        let (mut view, buffer, offset) = upload.write(
            encoder,
            self.decay_copy_target.as_mut(),
            DECAY_BYTES,
            self.uniform_align,
        );
        view.copy_from_slice(bytemuck::cast_slice(&v));
        drop(view);
        let layout = &self.decay_uniform_layout;
        let bind_group = cached_bind_group(&mut self.decay_bind_groups, &buffer, |buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("feedback-decay-uniform-bind"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: Some(DECAY_BYTES),
                    }),
                }],
            })
        });
        self.decay_bind = (bind_group, offset as u32);
    }

    /// Advances the ping-pong after the feedback pass so next frame's `prev` is this
    /// frame's accumulator.
    pub fn end_frame(&mut self) {
        self.prev = self.curr();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> (wgpu::Device, wgpu::Queue) {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL | wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let force_fallback =
            std::env::var("VIEWMUSIC_FORCE_FALLBACK_ADAPTER").as_deref() == Ok("1");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_fallback,
            compatible_surface: None,
        }))
        .expect("no GPU adapter for headless test");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("feedback-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("failed to acquire wgpu device");
        (device, queue)
    }

    #[test]
    fn accumulators_allocate_lazily_and_release_after_inactive_run_or_resize() {
        // The queue must outlive the encoder.
        let (device, _queue) = make_device();
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut feedback =
            Feedback::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, 64, 32, false);
        assert!(feedback.accumulators.is_none());

        feedback.prepare_frame(&device, &mut encoder, false);
        assert!(feedback.accumulators.is_some());
        assert!(!feedback.dirty);

        // A short bypass keeps the textures but clears them on return.
        for _ in 1..RELEASE_AFTER_INACTIVE_FRAMES {
            feedback.note_inactive_frame();
        }
        assert!(feedback.accumulators.is_some());
        assert!(feedback.dirty);
        feedback.prepare_frame(&device, &mut encoder, false);
        assert!(!feedback.dirty);

        for _ in 0..RELEASE_AFTER_INACTIVE_FRAMES {
            feedback.note_inactive_frame();
        }
        assert!(feedback.accumulators.is_none());
        feedback.prepare_frame(&device, &mut encoder, false);
        assert!(feedback.accumulators.is_some());

        feedback.resize(80, 40);
        assert!(feedback.accumulators.is_none());
        feedback.end_frame();
        feedback.prepare_frame(&device, &mut encoder, false);
        assert!(feedback.accumulators.is_some());
        assert_eq!(feedback.prev, 0);
        assert!(!feedback.dirty);
    }
}
