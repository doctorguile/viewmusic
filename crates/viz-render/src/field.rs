//! `field` layer: a `resolution × resolution` color grid rendered via a single
//! fullscreen triangle sampling a read-only storage buffer of cell colors.
//!
//! Cells are uploaded straight from `&[[f32; 4]]` with zero CPU packing loop
//! (half the upload bytes and 3 vertices instead of `6 * res^2` instanced quads).
//! Row-major from bottom-left (`i = row * resolution + col`), matching the contract.
//!
//! Each layer uploads one region: its [`FieldParams`] in the first `param_stride` bytes
//! (bound as a dynamic-offset uniform) followed by its cells (read through a storage
//! binding of the whole buffer, indexed from `params.base`).
//!
//! A leading alpha field whose cells are all opaque hides everything below it, so the
//! renderer draws it with blending off and skips the pass's decay draw (see the return
//! value of [`FieldPipelines::push`]).

use std::num::NonZeroU64;

use crate::upload::{cached_bind_group, Uploader};
use crate::{BlendMode, PassTargets, MAX_FIELD_CELLS, MAX_FIELD_RES};

/// Max field layers tracked per frame (contract §12: ≤16 layers per scene).
const MAX_LAYERS: usize = 16;

/// Per-layer uniform metadata at the start of the layer's upload region.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FieldParams {
    base: u32,
    res: u32,
    _pad: [u32; 2],
}

/// One field layer recorded this frame.
#[derive(Clone, Copy)]
struct LayerRecord {
    blend: BlendMode,
    has_cells: bool,
    /// Index into [`FieldPipelines::bind_groups`] for the buffer holding the region.
    bind_group: usize,
    /// Byte offset of the layer's region (its params) in that buffer.
    dynamic_offset: u32,
}

/// Field pipelines (one per pass targets × blend, plus unblended) + per-buffer bind groups.
pub struct FieldPipelines {
    /// `pipelines[targets][blend]`.
    pipelines: [[wgpu::RenderPipeline; 2]; 2],
    /// `unblended[targets]`, for a leading opaque layer (see [`FieldPipelines::push`]).
    unblended: [wgpu::RenderPipeline; 2],
    bind_layout: wgpu::BindGroupLayout,
    /// Copy destination holding `MAX_LAYERS` maximal regions; `None` when the draws bind
    /// the upload chunks directly.
    copy_target: Option<wgpu::Buffer>,
    /// Bind group per buffer that has held a region (the copy target, or upload chunks).
    bind_groups: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Bytes reserved for [`FieldParams`] at the start of each region (uniform offset
    /// alignment).
    param_stride: usize,
    /// Per-frame layer records (capacity `MAX_LAYERS`, reserved up front).
    records: Vec<LayerRecord>,
    /// Running byte cursor into `copy_target` for the current frame.
    cursor: u64,
}

impl FieldPipelines {
    /// Builds the six pipelines (and the region copy target unless uploads are `direct`).
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, direct: bool) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("field.wgsl").into()),
        });

        let param_size = std::mem::size_of::<FieldParams>() as u32;
        let align = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(param_size);
        let param_stride = align as usize;

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(param_size as u64),
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let make = |targets: PassTargets, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("field-pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
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
                    entry_point: Some(["fs_main", "fs_main_mrt"][targets.index()]),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets.color_targets(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipelines = PassTargets::ALL.map(|targets| {
            [
                make(targets, Some(BlendMode::Alpha.state())),
                make(targets, Some(BlendMode::Add.state())),
            ]
        });
        let unblended = PassTargets::ALL.map(|targets| make(targets, None));

        let copy_target = (!direct).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("field-regions"),
                size: ((param_stride + std::mem::size_of::<[f32; 4]>() * MAX_FIELD_CELLS)
                    * MAX_LAYERS) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        Self {
            pipelines,
            unblended,
            bind_layout,
            copy_target,
            bind_groups: Vec::new(),
            param_stride,
            records: Vec::with_capacity(MAX_LAYERS),
            cursor: 0,
        }
    }

    /// Resets per-frame bookkeeping.
    pub fn begin_frame(&mut self) {
        self.records.clear();
        self.cursor = 0;
    }

    /// Uploads the raw cell colors for one field layer directly (zero CPU packing loop).
    ///
    /// `resolution` is clamped to `2..=MAX_FIELD_RES`. `cells` holds up to `resolution²`
    /// straight-alpha RGBA colors row-major from bottom-left; missing trailing cells are
    /// uploaded as transparent `[0; 4]`, which both blend modes leave as a no-op.
    ///
    /// Returns `true` when the layer is opaque: alpha blend with all `res²` cells at
    /// alpha exactly 1. Blending it is then `src * 1 + dst * 0`, so as the first layer of
    /// a pass it covers every pixel and may be drawn unblended ([`FieldPipelines::draw`]).
    pub fn push(
        &mut self,
        device: &wgpu::Device,
        upload: &mut Uploader,
        encoder: &mut wgpu::CommandEncoder,
        blend: BlendMode,
        resolution: u32,
        cells: &[[f32; 4]],
    ) -> bool {
        if self.records.len() >= MAX_LAYERS {
            return false;
        }

        let res = resolution.clamp(2, MAX_FIELD_RES);
        let total = (res * res) as usize;
        let take = cells.len().min(total);
        if take == 0 {
            self.records.push(LayerRecord {
                blend,
                has_cells: false,
                bind_group: 0,
                dynamic_offset: 0,
            });
            return false;
        }

        let cell_bytes: &[u8] = bytemuck::cast_slice(&cells[..take]);
        let stride = self.param_stride as u64;
        let region = stride + (total * std::mem::size_of::<[f32; 4]>()) as u64;
        let (mut view, buffer, offset) = upload.write(
            encoder,
            self.copy_target.as_ref(),
            self.cursor,
            NonZeroU64::new(region).expect("region holds params"),
            NonZeroU64::new(stride).expect("non-zero param stride"),
        );
        let params = FieldParams {
            base: ((offset + stride) / std::mem::size_of::<[f32; 4]>() as u64) as u32,
            res,
            _pad: [0; 2],
        };
        view.slice(..std::mem::size_of::<FieldParams>())
            .copy_from_slice(bytemuck::bytes_of(&params));
        let cells_end = self.param_stride + cell_bytes.len();
        view.slice(self.param_stride..cells_end)
            .copy_from_slice(cell_bytes);
        view.slice(cells_end..).fill(0);
        drop(view);
        let opaque =
            blend == BlendMode::Alpha && take == total && cells[..take].iter().all(|c| c[3] == 1.0);

        let layout = &self.bind_layout;
        let bind_group = cached_bind_group(&mut self.bind_groups, &buffer, |buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field-bind-group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: NonZeroU64::new(std::mem::size_of::<FieldParams>() as u64),
                        }),
                    },
                ],
            })
        });

        self.records.push(LayerRecord {
            blend,
            has_cells: true,
            bind_group,
            dynamic_offset: offset as u32,
        });
        self.cursor += region.next_multiple_of(stride);
        opaque
    }

    /// Records the draw for the `idx`-th field layer pushed this frame (3 vertices).
    /// `unblended`: the layer is the pass's first and [`FieldPipelines::push`] reported it
    /// opaque, so it is drawn with blending off.
    pub fn draw<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        idx: usize,
        targets: PassTargets,
        unblended: bool,
    ) {
        let Some(&rec) = self.records.get(idx) else {
            return;
        };
        if !rec.has_cells {
            return;
        }
        pass.set_pipeline(if unblended {
            &self.unblended[targets.index()]
        } else {
            &self.pipelines[targets.index()][rec.blend as usize]
        });
        pass.set_bind_group(
            0,
            &self.bind_groups[rec.bind_group].1,
            &[rec.dynamic_offset],
        );
        pass.draw(0..3, 0..1);
    }
}
