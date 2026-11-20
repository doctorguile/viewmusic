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
//! ## Glitch-free resize
//! [`Feedback::resize`] reallocates both textures and **clears** them in the same call, so
//! no garbage from a stale-sized texture ever flashes during a resize. The roles are also
//! reset deterministically.

use std::num::NonZeroU64;

use crate::upload::{cached_bind_group, Uploader};
use crate::{PassTargets, CLEAR_COLOR, CLEAR_COLOR_TRANSPARENT};

/// Size of the decay uniform: one f32, padded to 16 bytes (uniform alignment).
const DECAY_BYTES: NonZeroU64 = NonZeroU64::new(16).unwrap();

/// Two-texture ping-pong accumulator + the fullscreen decay pipeline.
///
/// Sizing is owned by [`crate::Renderer`]; this struct only holds the GPU resources and
/// reallocates them on [`Feedback::resize`].
pub struct Feedback {
    /// The two accumulator textures and their views. Index by the `prev`/`curr` selectors.
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    /// Bind group that samples `textures[k]` (so `sample_bind[prev]` reads the prev image).
    sample_bind: [wgpu::BindGroup; 2],

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
    decay_copy_target: Option<wgpu::Buffer>,
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
    /// Decay value in `decay_copy_target`, so an unchanged decay skips the copy.
    last_decay: Option<f32>,
}

impl Feedback {
    /// Builds the pipelines, sampler, and the two accumulator textures at `width × height`
    /// (and the decay copy target unless uploads are `direct`).
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
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("feedback-decay-uniform"),
                size: DECAY_BYTES.get(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
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

        let (textures, views, sample_bind) =
            Self::make_targets(device, format, width, height, &bind_layout, &sampler);

        Self {
            textures,
            views,
            sample_bind,
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
        }
    }

    /// Allocates the two accumulator textures + their views + sampling bind groups.
    fn make_targets(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        bind_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> (
        [wgpu::Texture; 2],
        [wgpu::TextureView; 2],
        [wgpu::BindGroup; 2],
    ) {
        let make_tex = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
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
                layout: bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let sample_bind = [make_bind(&views[0]), make_bind(&views[1])];
        (textures, views, sample_bind)
    }

    /// Reallocates both accumulators to a new size and clears them (glitch-free).
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        let (textures, views, sample_bind) = Self::make_targets(
            device,
            format,
            width,
            height,
            &self.bind_layout,
            &self.sampler,
        );
        self.textures = textures;
        self.views = views;
        self.sample_bind = sample_bind;
        self.prev = 0;
        // Fresh textures have undefined contents; defer the clear to the next active frame
        // (encoder available there), so resize needs no queue and never flashes garbage.
        self.dirty = true;
    }

    /// Clears both accumulator textures to `clear_color` if they are dirty
    /// (post-allocation or after toggling transparent background). Records into
    /// the caller's encoder; no queue needed.
    fn clear_if_dirty(&mut self, encoder: &mut wgpu::CommandEncoder, clear_color: wgpu::Color) {
        if !self.dirty {
            return;
        }
        for view in &self.views {
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

    /// The view of the current render target (`curr`) — layers draw here.
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.views[self.curr()]
    }

    /// Marks the accumulator textures dirty so they will be cleared before the next
    /// active feedback frame (e.g., after bypassing feedback while `decay == 0`).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Prepares the accumulator textures for a merged decay+layers pass: clears both
    /// accumulators if freshly allocated, bypassed, or if `transparent_bg` changed.
    pub fn prepare_frame(&mut self, encoder: &mut wgpu::CommandEncoder, transparent_bg: bool) {
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
        pass.set_bind_group(0, &self.sample_bind[self.prev], &[]);
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
        if self.decay_copy_target.is_some() && self.last_decay == Some(clamped) {
            return;
        }
        let v = [clamped, 0.0, 0.0, 0.0];
        let (mut view, buffer, offset) = upload.write(
            encoder,
            self.decay_copy_target.as_ref(),
            0,
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
        self.last_decay = Some(clamped);
    }

    /// Advances the ping-pong after the feedback pass so next frame's `prev` is this
    /// frame's accumulator.
    pub fn end_frame(&mut self) {
        self.prev = self.curr();
    }
}
