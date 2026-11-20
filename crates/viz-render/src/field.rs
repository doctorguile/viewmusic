//! `field` layer: a `resolution × resolution` color grid rendered via a single
//! fullscreen triangle sampling a read-only storage buffer of cell colors.
//!
//! Cells are uploaded straight from `&[[f32; 4]]` with zero CPU packing loop
//! (half the upload bytes and 3 vertices instead of `6 * res^2` instanced quads).
//! Row-major from bottom-left (`i = row * resolution + col`), matching the contract.

use std::num::NonZeroU64;

use wgpu::util::StagingBelt;

use crate::{BlendMode, PassTargets, MAX_FIELD_CELLS, MAX_FIELD_RES};

/// Max field layers tracked per frame (contract §12: ≤16 layers per scene).
const MAX_LAYERS: usize = 16;

/// Per-layer uniform metadata uploaded into a dynamic-offset uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FieldParams {
    base: u32,
    count: u32,
    res: u32,
    _pad: u32,
}

/// One field layer recorded this frame.
#[derive(Clone, Copy)]
struct LayerRecord {
    blend: BlendMode,
    count: u32,
    dynamic_offset: u32,
}

/// Field pipelines (one per pass targets × blend) + shared cell storage buffer + dynamic
/// params uniform.
pub struct FieldPipelines {
    /// `pipelines[targets][blend]`.
    pipelines: [[wgpu::RenderPipeline; 2]; 2],
    /// Shared cell color buffer (`[f32; 4]` × `MAX_LAYERS * MAX_FIELD_CELLS`).
    cells_buf: wgpu::Buffer,
    /// Dynamic-offset uniform buffer holding `MAX_LAYERS` aligned [`FieldParams`] slots.
    params_buf: wgpu::Buffer,
    /// Bind group binding `cells_buf` + `params_buf` (dynamic offset per layer).
    bind_group: wgpu::BindGroup,
    /// Aligned byte stride between slots in `params_buf`.
    param_stride: usize,
    /// CPU staging for all field layer params this frame (uploaded in one write).
    params_staging: Vec<u8>,
    records: [LayerRecord; MAX_LAYERS],
    len: usize,
    cursor: u32,
}

impl FieldPipelines {
    /// Builds the four pipelines and pre-allocates storage + dynamic uniform buffers.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
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

        let make = |targets: PassTargets, blend: BlendMode| {
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

        let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-cells"),
            size: (std::mem::size_of::<[f32; 4]>() * MAX_LAYERS * MAX_FIELD_CELLS)
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-params"),
            size: (param_stride * MAX_LAYERS) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cells_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buf,
                        offset: 0,
                        size: NonZeroU64::new(param_size as u64),
                    }),
                },
            ],
        });

        let zero = LayerRecord {
            blend: BlendMode::Alpha,
            count: 0,
            dynamic_offset: 0,
        };
        Self {
            pipelines,
            cells_buf,
            params_buf,
            bind_group,
            param_stride,
            params_staging: vec![0u8; param_stride * MAX_LAYERS],
            records: [zero; MAX_LAYERS],
            len: 0,
            cursor: 0,
        }
    }

    /// Resets per-frame bookkeeping.
    pub fn begin_frame(&mut self) {
        self.len = 0;
        self.cursor = 0;
    }

    /// Uploads the raw cell colors for one field layer directly (zero CPU packing loop).
    ///
    /// `resolution` is clamped to `2..=MAX_FIELD_RES`. `cells` holds up to `resolution²`
    /// straight-alpha RGBA colors row-major from bottom-left.
    pub fn push(
        &mut self,
        belt: &mut StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
        blend: BlendMode,
        resolution: u32,
        cells: &[[f32; 4]],
    ) {
        if self.len >= MAX_LAYERS {
            return;
        }

        let res = resolution.clamp(2, MAX_FIELD_RES);
        let total = (res * res) as usize;
        let cap_total = (MAX_LAYERS * MAX_FIELD_CELLS) as u32;
        let remaining = cap_total.saturating_sub(self.cursor) as usize;
        let take = cells.len().min(total).min(remaining);
        let base = self.cursor;

        if take > 0 {
            let offset = base as u64 * std::mem::size_of::<[f32; 4]>() as u64;
            let bytes = bytemuck::cast_slice(&cells[..take]);
            if let Some(size) = NonZeroU64::new(bytes.len() as u64) {
                belt.write_buffer(encoder, &self.cells_buf, offset, size)
                    .copy_from_slice(bytes);
            }
        }

        let slot_offset = self.len * self.param_stride;
        let params = FieldParams {
            base,
            count: take as u32,
            res,
            _pad: 0,
        };
        self.params_staging[slot_offset..slot_offset + std::mem::size_of::<FieldParams>()]
            .copy_from_slice(bytemuck::bytes_of(&params));

        self.records[self.len] = LayerRecord {
            blend,
            count: take as u32,
            dynamic_offset: slot_offset as u32,
        };
        self.len += 1;
        self.cursor += take as u32;
    }

    /// Uploads all field-layer param uniforms recorded this frame in a single write.
    pub fn flush_params(&self, belt: &mut StagingBelt, encoder: &mut wgpu::CommandEncoder) {
        if self.len == 0 {
            return;
        }
        let bytes = &self.params_staging[..self.len * self.param_stride];
        if let Some(size) = NonZeroU64::new(bytes.len() as u64) {
            belt.write_buffer(encoder, &self.params_buf, 0, size)
                .copy_from_slice(bytes);
        }
    }

    /// Records the draw for the `idx`-th field layer pushed this frame (3 vertices).
    pub fn draw<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, idx: usize, targets: PassTargets) {
        if idx >= self.len {
            return;
        }
        let rec = self.records[idx];
        if rec.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipelines[targets.index()][rec.blend as usize]);
        pass.set_bind_group(0, &self.bind_group, &[rec.dynamic_offset]);
        pass.draw(0..3, 0..1);
    }
}
