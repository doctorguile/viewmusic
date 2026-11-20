//! Thick polylines as instanced screen-space segment quads.
//!
//! Each consecutive point pair `(p0, p1)` becomes one instanced quad. The quad is a unit
//! template in `[0,1] × [-0.5, 0.5]`: the x axis runs along the segment, the y axis is the
//! perpendicular offset scaled by `thickness`. Thickness is specified in **logical
//! pixels** and converted to NDC using the current target size — the segment direction is
//! computed in pixel space so the line keeps a constant visual width regardless of aspect.
//! Per-point colors are interpolated along the segment. A `closed` loop adds the final
//! `last → first` segment.
//!
//! Pre-allocated for `MAX_POINTS` points per layer ⇒ `MAX_POINTS` segments (closed).

use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::upload::{CopyTarget, Uploader, VERTEX_ALIGN};
use crate::{BlendMode, PassTargets, PointData, MAX_POINTS};

/// Max polyline layers tracked per frame (contract §12: ≤16 layers per scene).
const MAX_LAYERS: usize = 16;

/// One expanded segment instance: endpoints (NDC) + endpoint colors + half-thickness in
/// NDC for each axis (pre-divided by the viewport so the shader stays trivial).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Segment {
    /// Segment start, NDC.
    p0: [f32; 2],
    /// Segment end, NDC.
    p1: [f32; 2],
    /// Half-thickness in NDC along x and y (= half_px / (width/2), half_px / (height/2)).
    half_ndc: [f32; 2],
    /// Padding to keep the struct 16-byte aligned for tidy vertex attributes.
    _pad: [f32; 2],
    /// Start color (straight alpha).
    c0: [f32; 4],
    /// End color (straight alpha).
    c1: [f32; 4],
}

/// One polyline layer recorded this frame.
struct LayerRecord {
    blend: BlendMode,
    /// Buffer and byte offset holding the segments; `None` for an empty layer.
    data: Option<(wgpu::Buffer, wgpu::BufferAddress)>,
    count: u32,
}

/// Polyline pipelines (one per pass targets × blend) + the shared segment instance buffer.
pub struct PolylinePipelines {
    /// `pipelines[targets][blend]`.
    pipelines: [[wgpu::RenderPipeline; 2]; 2],
    /// Static template quad ([0,1] × [-0.5,0.5]) as 6 vertices.
    template: wgpu::Buffer,
    /// Copy destination for segments (capacity `MAX_LAYERS * MAX_POINTS`); `None` when the
    /// draws bind the upload chunks directly.
    copy_target: Option<CopyTarget>,
    /// CPU staging for the current layer's segments (pre-allocated, no per-frame alloc).
    staging: Vec<Segment>,
    /// Per-frame layer records (capacity `MAX_LAYERS`, reserved up front).
    records: Vec<LayerRecord>,
}

/// Template quad: x along segment in [0,1], y perpendicular in [-0.5, 0.5].
const TEMPLATE: [[f32; 2]; 6] = [
    [0.0, -0.5],
    [1.0, -0.5],
    [1.0, 0.5],
    [0.0, -0.5],
    [1.0, 0.5],
    [0.0, 0.5],
];

impl PolylinePipelines {
    /// Builds the four pipelines and allocates buffers + CPU staging (the segment copy
    /// target only unless uploads are `direct`).
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, direct: bool) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("polyline.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let template_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        };
        let seg_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Segment>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            // p0 (1), p1 (2), half_ndc+pad (3 vec4), c0 (4), c1 (5).
            attributes: &wgpu::vertex_attr_array![
                1 => Float32x2,
                2 => Float32x2,
                3 => Float32x4,
                4 => Float32x4,
                5 => Float32x4
            ],
        };

        let make = |targets: PassTargets, blend: BlendMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("polyline-pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[template_layout.clone(), seg_layout.clone()],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(["fs_main", "fs_main_mrt"][targets.index()]),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets.color_targets(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend.state()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipelines = PassTargets::ALL.map(|targets| {
            [
                make(targets, BlendMode::Alpha),
                make(targets, BlendMode::Add),
            ]
        });

        let template = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("polyline-template"),
            contents: bytemuck::cast_slice(&TEMPLATE),
            // MAP_WRITE (direct uploads only) writes the contents in place instead of
            // through a staging copy, so creating a renderer issues no blit either.
            usage: wgpu::BufferUsages::VERTEX
                | if direct {
                    wgpu::BufferUsages::MAP_WRITE
                } else {
                    wgpu::BufferUsages::empty()
                },
        });
        let copy_target = (!direct).then(|| {
            CopyTarget::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("polyline-segments"),
                size: (std::mem::size_of::<Segment>() * MAX_LAYERS * MAX_POINTS)
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });

        Self {
            pipelines,
            template,
            copy_target,
            staging: Vec::with_capacity(MAX_POINTS),
            records: Vec::with_capacity(MAX_LAYERS),
        }
    }

    /// Resets per-frame bookkeeping.
    pub fn begin_frame(&mut self) {
        self.records.clear();
        if let Some(target) = &mut self.copy_target {
            target.rewind();
        }
    }

    /// Builds and uploads the segments for one polyline layer.
    ///
    /// `thickness_px` is logical pixels, clamped to a sane floor (≥0.1). `width`/`height`
    /// are the current target dimensions in physical pixels (used for the px→NDC scale).
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        upload: &mut Uploader,
        encoder: &mut wgpu::CommandEncoder,
        blend: BlendMode,
        thickness_px: f32,
        closed: bool,
        points: &[PointData],
        width: u32,
        height: u32,
    ) {
        if self.records.len() >= MAX_LAYERS {
            return;
        }
        self.staging.clear();

        let n = points.len().min(MAX_POINTS);
        // A line needs at least two points.
        if n >= 2 {
            // Half-thickness in NDC: NDC spans 2 units across width/height px, so 1 px =
            // 2/dim in NDC, and half-thickness px = thickness/2 px.
            let half_px = (thickness_px.max(0.1)) * 0.5;
            let half_ndc = [
                half_px * 2.0 / width.max(1) as f32,
                half_px * 2.0 / height.max(1) as f32,
            ];
            for pair in points[..n].windows(2) {
                let a = &pair[0];
                let b = &pair[1];
                self.staging.push(Segment {
                    p0: [a.x, a.y],
                    p1: [b.x, b.y],
                    half_ndc,
                    _pad: [0.0, 0.0],
                    c0: a.color,
                    c1: b.color,
                });
            }
            if closed {
                let a = &points[n - 1];
                let b = &points[0];
                self.staging.push(Segment {
                    p0: [a.x, a.y],
                    p1: [b.x, b.y],
                    half_ndc,
                    _pad: [0.0, 0.0],
                    c0: a.color,
                    c1: b.color,
                });
            }
        }

        let take = self.staging.len();
        let bytes: &[u8] = bytemuck::cast_slice(&self.staging);
        let data = NonZeroU64::new(bytes.len() as u64).map(|size| {
            let (mut view, buffer, offset) =
                upload.write(encoder, self.copy_target.as_mut(), size, VERTEX_ALIGN);
            view.copy_from_slice(bytes);
            (buffer, offset)
        });
        self.records.push(LayerRecord {
            blend,
            data,
            count: take as u32,
        });
    }

    /// Records the draw for the `idx`-th polyline layer pushed this frame.
    pub fn draw<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, idx: usize, targets: PassTargets) {
        let Some(rec) = self.records.get(idx) else {
            return;
        };
        let Some((buffer, offset)) = &rec.data else {
            return;
        };
        pass.set_pipeline(&self.pipelines[targets.index()][rec.blend.index()]);
        pass.set_vertex_buffer(0, self.template.slice(..));
        let end = offset + rec.count as u64 * std::mem::size_of::<Segment>() as u64;
        pass.set_vertex_buffer(1, buffer.slice(*offset..end));
        pass.draw(0..TEMPLATE.len() as u32, 0..rec.count);
    }
}
