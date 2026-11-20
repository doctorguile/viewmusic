//! viz-render — the GPU layer renderer for ViewMusic.
//!
//! Consumes the runtime's per-frame draw lists ([`LayerDraw`]) and rasterizes them
//! into the window surface via wgpu 29 (Metal on this target). The renderer is driven
//! from `viz-app`'s `RenderCallback`/`FrameContext` seam: the app acquires the surface
//! texture, records a dark clear pass, then hands us the same command encoder and
//! surface view. We render the whole scene and leave the surface ready for the egui
//! overlay pass, which loads on top with `LoadOp::Load`.
//!
//! ## Coordinate contract
//! Artifact space is `x ∈ −1..1` (right), `y ∈ −1..1` (up), mapping DIRECTLY to NDC.
//! Sizes `w`/`h` are in the same units; `rot` is radians CCW about the element center.
//! Colors arrive as straight-alpha `[f32; 4]` in `0..1` (already clamped/converted by
//! the runtime); the shaders premultiply before blending so alpha-over and additive
//! both behave.
//!
//! ## Real-time discipline
//! Per-frame layer data goes through a reusable staging belt (see [`upload`]). With
//! [`device_features`] enabled on a unified-memory GPU the draws bind the belt's mapped
//! chunks directly and a frame records no copy commands; otherwise the belt copies into
//! buffers pre-allocated to the contract maxima ([`MAX_INSTANCES`], [`MAX_POINTS`],
//! [`MAX_FIELD_CELLS`]). In steady state no buffers are created, and there is no heap
//! allocation, no lock, and no blocking I/O in [`Renderer::render`].
//!
//! ## Module map
//! - [`instanced`]: instanced quads with rect/circle/triangle masking, two blends.
//! - [`polyline`]: thick polylines as instanced screen-space segment quads.
//! - [`field`]: resolution×resolution color grid as instanced cell quads.
//! - [`feedback`]: persistent decay/trails accumulator (rendered alongside the surface).
//! - [`degrade`]: pure adaptive-detail / pause ladder logic (no GPU).
//! - [`upload`]: per-frame upload of layer data (direct-bound or copied).

mod degrade;
mod feedback;
mod field;
mod instanced;
mod polyline;
mod upload;

pub use degrade::DegradeController;

/// Optional device features the renderer uses; request them in `request_device`.
///
/// On a unified-memory GPU, [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] lets the draws
/// bind the mapped upload buffers directly, so frames record no copy commands. It is not
/// requested on discrete GPUs, where mapped buffers live across the bus.
pub fn device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    if adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu {
        adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
    } else {
        wgpu::Features::empty()
    }
}

/// Maximum instances per `instanced` layer (contract §12: 4096).
pub const MAX_INSTANCES: usize = 4096;
/// Maximum points per `polyline` layer (contract §12: 4096).
pub const MAX_POINTS: usize = 4096;
/// Maximum `field` resolution per axis (contract §12: 128 ⇒ 16 384 cells).
pub const MAX_FIELD_RES: u32 = 128;
/// Maximum `field` cells (`MAX_FIELD_RES²`).
pub const MAX_FIELD_CELLS: usize = (MAX_FIELD_RES * MAX_FIELD_RES) as usize;

/// One instanced shape, in artifact/NDC coordinates.
///
/// `repr(C)`, `Pod`+`Zeroable` so it can be `bytemuck`-cast straight into a GPU
/// instance buffer with no per-frame allocation. The runtime fills a slice of these
/// and hands it over via [`LayerDraw::Instanced`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    /// Center x in NDC (−1..1, right).
    pub x: f32,
    /// Center y in NDC (−1..1, up).
    pub y: f32,
    /// Full width in NDC units.
    pub w: f32,
    /// Full height in NDC units.
    pub h: f32,
    /// Rotation in radians, CCW about the center.
    pub rot: f32,
    /// Straight-alpha color, `0..1`, RGBA.
    pub color: [f32; 4],
}

/// One polyline vertex, in artifact/NDC coordinates.
///
/// `repr(C)`, `Pod`+`Zeroable`. The renderer expands consecutive points into thick
/// screen-space segment quads (see [`polyline`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointData {
    /// Point x in NDC (−1..1).
    pub x: f32,
    /// Point y in NDC (−1..1).
    pub y: f32,
    /// Straight-alpha color, `0..1`, RGBA.
    pub color: [f32; 4],
}

/// Fixed-function GPU blend mode for a layer.
///
/// Mapped from the contract's `blend` at the call site in the runtime (this crate does
/// NOT depend on `viz-contract`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Standard alpha-over (premultiplied): `src + dst*(1 − src.a)`.
    Alpha,
    /// Additive (premultiplied): `src + dst` — for glow.
    Add,
}

impl BlendMode {
    /// Index into the two-element pipeline tables (`0 = Alpha`, `1 = Add`).
    #[inline]
    fn index(self) -> usize {
        match self {
            BlendMode::Alpha => 0,
            BlendMode::Add => 1,
        }
    }

    /// The wgpu blend state for this mode. Shaders output premultiplied-alpha color, so
    /// alpha-over uses `src_factor = One` (not `SrcAlpha`).
    fn state(self) -> wgpu::BlendState {
        match self {
            BlendMode::Alpha => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            BlendMode::Add => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
        }
    }
}

/// The color attachments of the pass a layer pipeline draws into.
///
/// The feedback path renders the accumulator and the surface in ONE pass with identical
/// shader output and blend state on both attachments, so the surface ends up equal to the
/// accumulator without a separate full-screen blit pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassTargets {
    /// Direct path: the surface only.
    Surface,
    /// Feedback path: `@location(0)` = accumulator (`curr`), `@location(1)` = surface.
    FeedbackAndSurface,
}

impl PassTargets {
    /// Both variants, in [`PassTargets::index`] order (for building pipeline tables).
    const ALL: [PassTargets; 2] = [PassTargets::Surface, PassTargets::FeedbackAndSurface];

    /// Index into two-element per-target pipeline tables.
    #[inline]
    fn index(self) -> usize {
        match self {
            PassTargets::Surface => 0,
            PassTargets::FeedbackAndSurface => 1,
        }
    }

    /// `state` repeated once per attachment, as the pipeline's fragment target list.
    fn color_targets(self, state: wgpu::ColorTargetState) -> Vec<Option<wgpu::ColorTargetState>> {
        match self {
            PassTargets::Surface => vec![Some(state)],
            PassTargets::FeedbackAndSurface => vec![Some(state.clone()), Some(state)],
        }
    }
}

/// Which primitive an [`LayerDraw::Instanced`] layer rasterizes.
///
/// Selects a fragment mask (circle radius / triangle half-plane); see [`instanced`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    /// Axis-then-rotated rectangle filling the full `w×h` quad.
    Rect,
    /// Inscribed circle (fragments outside radius 0.5 of the quad are discarded).
    Circle,
    /// Upward isoceles triangle inscribed in the quad (half-plane mask).
    Triangle,
}

impl ShapeKind {
    /// Index into the three-element pipeline table (`0 = Rect`, `1 = Circle`, `2 = Tri`).
    #[inline]
    fn index(self) -> usize {
        match self {
            ShapeKind::Rect => 0,
            ShapeKind::Circle => 1,
            ShapeKind::Triangle => 2,
        }
    }
}

/// One layer's worth of geometry to draw, borrowed from the runtime's staging buffers.
///
/// The renderer iterates a `&[LayerDraw]` in order (painter's algorithm) each frame.
/// Slices are truncated to the pre-allocated capacities if a caller ever exceeds them
/// (defence in depth; the runtime already clamps to contract limits).
#[derive(Debug)]
pub enum LayerDraw<'a> {
    /// N instanced shapes of one kind/blend.
    Instanced {
        /// Primitive kind.
        shape: ShapeKind,
        /// Blend mode.
        blend: BlendMode,
        /// Per-instance transform + color.
        instances: &'a [InstanceData],
    },
    /// A connected strip (optionally closed) of thick points.
    Polyline {
        /// Blend mode.
        blend: BlendMode,
        /// Line thickness in logical pixels (converted to NDC using the target size).
        thickness_px: f32,
        /// `true` joins last→first (radial scopes).
        closed: bool,
        /// Ordered points with per-point color (interpolated along each segment).
        points: &'a [PointData],
    },
    /// A `resolution × resolution` color grid (row-major from bottom-left).
    Field {
        /// Blend mode.
        blend: BlendMode,
        /// Cells per axis (clamped to [`MAX_FIELD_RES`]).
        resolution: u32,
        /// `resolution²` straight-alpha RGBA colors, row-major from bottom-left.
        cells: &'a [[f32; 4]],
    },
}

/// Scene-wide parameters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct SceneParams {
    /// Feedback/trails decay in `0..0.99`. `None` disables trails (layers drawn directly
    /// to the surface over a dark clear). `Some(d)` multiplies the previous frame by `d`
    /// before drawing this frame's layers (contract §8).
    pub feedback_decay: Option<f32>,
    /// When `true`, clear the scene background to transparent (`0, 0, 0, 0`) instead of
    /// the opaque dark clear color so the desktop shows behind the visualizer.
    pub transparent_bg: bool,
}

/// Background clear color used both for the no-feedback direct path and for clearing the
/// feedback texture on (re)allocation. Matches the app shell's dark clear.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

/// Transparent background clear color used when [`SceneParams::transparent_bg`] is enabled.
const CLEAR_COLOR_TRANSPARENT: wgpu::Color = wgpu::Color::TRANSPARENT;

/// The GPU scene renderer.
///
/// Holds all pipelines and pre-allocated buffers/targets. Construct once against the
/// app's `device`/`queue`/`surface_format`; drive [`Renderer::render`] each frame from
/// the `RenderCallback`; call [`Renderer::resize`] on surface resize.
pub struct Renderer {
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,

    instanced: instanced::InstancedPipelines,
    polyline: polyline::PolylinePipelines,
    field: field::FieldPipelines,
    feedback: feedback::Feedback,
    /// Per-frame upload of layer data through reusable mapped buffers.
    upload: upload::Uploader,
}

impl Renderer {
    /// Builds the renderer: all pipelines, buffers, and the feedback target.
    ///
    /// `surface_format` MUST match the surface the encoder's view targets (and the egui
    /// overlay that follows). Blocking-free; safe to call during app `on_ready`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // Start at a 1×1 target; the app calls resize() with the real size before/at the
        // first frame. Feedback allocates lazily to the current size.
        let width = 1;
        let height = 1;
        let upload = upload::Uploader::new(device);
        let direct = upload.direct();
        let instanced = instanced::InstancedPipelines::new(device, surface_format, direct);
        let polyline = polyline::PolylinePipelines::new(device, surface_format, direct);
        let field = field::FieldPipelines::new(device, surface_format, direct);
        let feedback = feedback::Feedback::new(device, surface_format, width, height, direct);
        let _ = queue; // queue not needed at construction; kept for API symmetry.
        Self {
            format: surface_format,
            width,
            height,
            instanced,
            polyline,
            field,
            feedback,
            upload,
        }
    }

    /// Reclaims completed staging chunks after `queue.submit(...)` so the next frame
    /// reuses mapped GPU buffers with zero kernel allocations.
    pub fn recall_staging(&mut self) {
        self.upload.recall();
    }

    /// Reallocates size-dependent targets for a new surface size (glitch-free: the
    /// feedback texture is recreated and cleared so no garbage flashes during a resize).
    ///
    /// Zero dimensions are clamped to 1 to keep wgpu happy on a minimized window.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.feedback.resize(device, self.format, width, height);
        tracing::debug!(width, height, "viz-render: resized targets");
    }

    /// Renders the whole scene for one frame.
    ///
    /// Records into the caller's `encoder` and targets `surface_view`. After this returns
    /// the surface holds the composited scene and is ready for the egui overlay pass
    /// (which uses `LoadOp::Load`). No allocation occurs on this path.
    ///
    /// Two paths:
    /// - `feedback_decay = None`: clear the surface dark, draw layers directly onto it.
    /// - `feedback_decay = Some(d)`: decay the persistent feedback texture by `d` and draw
    ///   layers into it, writing the surface as a second attachment of the same pass.
    ///
    /// The feedback pass needs `surface_view` to match the accumulator size, so the
    /// renderer follows the surface size here (a no-op when [`Renderer::resize`] already
    /// matched it).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
        params: &SceneParams,
        layers: &[LayerDraw],
    ) {
        let _ = queue; // uploads go through `self.upload`.
        let surface_size = surface_view.texture().size();
        self.resize(device, surface_size.width, surface_size.height);

        // Reclaim any staging chunks whose previous submit finished (safe even if the
        // caller did not invoke `recall_staging` after submit).
        self.upload.recall();

        // Upload all per-layer geometry once, up front (zero per-frame MTLBuffer creation
        // or extra command buffer).
        self.upload_layers(device, encoder, layers);

        let clear_color = if params.transparent_bg {
            CLEAR_COLOR_TRANSPARENT
        } else {
            CLEAR_COLOR
        };

        let active_decay = params
            .feedback_decay
            .map(|d| d.clamp(0.0, 0.99))
            .filter(|&d| d > 0.0);

        match active_decay {
            None => {
                // Mark accumulators dirty so re-enabling feedback later clears stale history.
                self.feedback.mark_dirty();
                // Direct path: clear surface, draw layers on top.
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viz-render-direct"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: surface_view,
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
                self.draw_layers(&mut pass, layers, PassTargets::Surface);
            }
            Some(decay) => {
                // Queue the decay scalar (cached) before recording the pass that reads it.
                self.feedback
                    .set_decay(device, &mut self.upload, encoder, decay);
                // Clear accumulators lazily if resized, bypassed, or transparency toggled.
                self.feedback.prepare_frame(encoder, params.transparent_bg);
                // One pass draws `prev * decay` fullscreen AND all layers into both `curr`
                // and the surface, so no blit pass has to copy `curr` to the surface.
                {
                    let ops = wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viz-render-feedback-and-surface"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: self.feedback.target_view(),
                                resolve_target: None,
                                depth_slice: None,
                                ops,
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: surface_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops,
                            }),
                        ],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.feedback.draw_decay(&mut pass);
                    self.draw_layers(&mut pass, layers, PassTargets::FeedbackAndSurface);
                }
                self.feedback.end_frame();
            }
        }

        // Unmap active staging chunks before the caller submits `encoder.finish()`.
        self.upload.finish();
    }

    /// Uploads all geometry for this frame through [`Renderer::upload`].
    fn upload_layers(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layers: &[LayerDraw],
    ) {
        // Reset per-frame draw bookkeeping, then append each layer's data.
        self.instanced.begin_frame();
        self.polyline.begin_frame();
        self.field.begin_frame();
        for layer in layers {
            match layer {
                LayerDraw::Instanced {
                    shape,
                    blend,
                    instances,
                } => {
                    self.instanced
                        .push(&mut self.upload, encoder, *shape, *blend, instances);
                }
                LayerDraw::Polyline {
                    blend,
                    thickness_px,
                    closed,
                    points,
                } => {
                    self.polyline.push(
                        &mut self.upload,
                        encoder,
                        *blend,
                        *thickness_px,
                        *closed,
                        points,
                        self.width,
                        self.height,
                    );
                }
                LayerDraw::Field {
                    blend,
                    resolution,
                    cells,
                } => {
                    self.field.push(
                        device,
                        &mut self.upload,
                        encoder,
                        *blend,
                        *resolution,
                        cells,
                    );
                }
            }
        }
    }

    /// Records draw calls for every layer, in order, into an open render pass.
    fn draw_layers<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        layers: &[LayerDraw],
        targets: PassTargets,
    ) {
        // Each sub-module tracks how many of its layers it has emitted this frame so the
        // per-layer GPU buffer offsets line up with the upload() order.
        let mut inst_idx = 0;
        let mut poly_idx = 0;
        let mut field_idx = 0;
        for layer in layers {
            match layer {
                LayerDraw::Instanced { .. } => {
                    self.instanced.draw(pass, inst_idx, targets);
                    inst_idx += 1;
                }
                LayerDraw::Polyline { .. } => {
                    self.polyline.draw(pass, poly_idx, targets);
                    poly_idx += 1;
                }
                LayerDraw::Field { .. } => {
                    self.field.draw(pass, field_idx, targets);
                    field_idx += 1;
                }
            }
        }
    }
}
