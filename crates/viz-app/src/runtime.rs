//! The artifact runtime: deterministic per-frame evaluation of a
//! [`viz_contract::LoadedArtifact`] into renderable geometry. See
//! `docs/reference/artifact-contract.md` (§2.3/§5/§6/§7) for the evaluation rules
//! this implements.
//!
//! An [`ArtifactRuntime`] owns one reusable [`viz_expr::Vm`] (sized to the
//! artifact's `slot_count`), per-layer multi-output LICM + CSE [`MultiProgram`]s,
//! and staging buffers pre-allocated to the contract maxima
//! ([`viz_render::MAX_INSTANCES`] / [`viz_render::MAX_POINTS`] /
//! [`viz_render::MAX_FIELD_CELLS`]).
//!
//! ## Determinism (see `docs/reference/artifact-contract.md` §9)
//! The output is a pure function of the supplied `(FeatureFrame, dt, aspect)`
//! sequence and the setting values. There is no wall-clock read and no
//! `HashMap`-order iteration anywhere on the eval path: vars and settings are
//! visited in [`IndexMap`](indexmap::IndexMap) declaration order, layers in
//! `scene` array order.

use indexmap::IndexMap;

use viz_contract::{
    Blend, ColorModel, ColorSpec, CompiledFormula, Formula, Layer, LayerPrograms, LoadedArtifact,
    SettingSlots, Shape,
};
use viz_core::FeatureFrame;
use viz_expr::{compile_multi, MultiInput, MultiProgram, Vm, MAX_MULTI_OUTS};
use viz_render::{
    BlendMode, InstanceData, LayerDraw, PointData, SceneParams, ShapeKind, MAX_FIELD_CELLS,
    MAX_FIELD_RES, MAX_INSTANCES, MAX_POINTS,
};

/// Per-frame element-count clamp for an `instanced` layer (contract §6.1 / §12).
const INSTANCE_MIN: usize = 1;
/// Per-frame point-count clamp for a `polyline` layer (contract §6.2 / §12).
const POINT_MIN: usize = 2;
/// Per-frame resolution clamps for a `field` layer (contract §6.3 / §12).
const FIELD_RES_MIN: u32 = 2;
/// Thickness clamps for a `polyline` (contract §6.2): logical pixels 0.1..64.
const THICKNESS_MIN: f32 = 0.1;
const THICKNESS_MAX: f32 = 64.0;
/// Feedback decay clamp (contract §8).
const DECAY_MIN: f32 = 0.0;
const DECAY_MAX: f32 = 0.99;

/// The live value of one setting, applied to the VM's setting slots at
/// [`ArtifactRuntime::activate`] and whenever the UI mutates it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SettingValue {
    /// A `number`, `toggle`, or `choice` setting as one scalar.
    Scalar(f64),
    /// A `color` setting: red/green/blue/alpha in `0..1`.
    Color {
        /// Red channel, `0..1`.
        r: f64,
        /// Green channel, `0..1`.
        g: f64,
        /// Blue channel, `0..1`.
        b: f64,
        /// Alpha channel, `0..1`.
        a: f64,
    },
}

/// Per-layer draw metadata recorded each frame, used to build the borrowed
/// [`LayerDraw`] list in [`SceneOut`].
#[derive(Clone, Copy, Debug)]
enum LayerSlot {
    /// `instanced`: `len` instances starting at `offset` in the instance staging.
    Instanced {
        offset: usize,
        len: usize,
        shape: ShapeKind,
        blend: BlendMode,
    },
    /// `polyline`: `len` points starting at `offset` in the point staging.
    Polyline {
        offset: usize,
        len: usize,
        thickness_px: f32,
        closed: bool,
        blend: BlendMode,
    },
    /// `field`: a `resolution²` grid starting at `offset` in the cell staging.
    Field {
        offset: usize,
        resolution: u32,
        blend: BlendMode,
    },
    /// A layer skipped this frame (`visible < 0.5`, or paused) — emits nothing.
    Skipped,
}

/// One frame's renderable scene, borrowing the runtime's staging buffers.
pub struct SceneOut<'a> {
    /// Scene-wide parameters (feedback decay).
    pub params: SceneParams,
    /// Per-layer draw lists, borrowing the staging slices. Bounded by ≤16 layers.
    pub layers: Vec<LayerDraw<'a>>,
}

/// Precomputed per-layer compiled multi-output program with LICM + cross-formula CSE
/// (built once in [`ArtifactRuntime::new`]).
#[derive(Clone, Debug)]
struct CompiledLayer {
    prog: MultiProgram,
    reads_i: bool,
    reads_n: bool,
    reads_u: bool,
    reads_x: bool,
    reads_y: bool,
    color_model: ColorModel,
    /// True if all 4 color channels are frame-invariant (so `pack_color` + HSV→RGB
    /// runs once per layer instead of once per element/point).
    color_frame_invariant: bool,
    /// True if all outputs of a Field layer are frame-invariant (`item_ops` & `row_ops` empty).
    all_frame_invariant: bool,
    /// True if a Field layer varies only by row (`item_ops` empty, `row_ops` non-empty).
    row_only: bool,
}

/// The deterministic per-frame evaluator for one [`LoadedArtifact`].
///
/// Build with [`ArtifactRuntime::new`], call [`ArtifactRuntime::activate`] once
/// (runs `vars.init`), then [`ArtifactRuntime::frame`] per rendered frame.
pub struct ArtifactRuntime {
    /// The compiled artifact (programs, slot layout, shape metadata).
    art: LoadedArtifact,
    /// One reusable VM bank sized to `layout.slot_count`.
    vm: Vm,
    /// Live setting values in `layout.settings` declaration order.
    settings: IndexMap<String, SettingValue>,
    /// Per-layer multi-output LICM + CSE programs (matches `art.programs.layers`).
    compiled_layers: Vec<CompiledLayer>,
    /// Reusable scratch table for `Field` x-only hoisted column values (`res * num_col`).
    col_scratch: Vec<f64>,

    /// Pre-allocated instance staging (capacity [`MAX_INSTANCES`]).
    instances: Vec<InstanceData>,
    /// Pre-allocated point staging (capacity [`MAX_POINTS`]).
    points: Vec<PointData>,
    /// Pre-allocated field-cell staging (capacity [`MAX_FIELD_CELLS`]).
    cells: Vec<[f32; 4]>,

    /// Per-layer draw metadata for the current frame (capacity ≤16).
    layer_slots: Vec<LayerSlot>,
    /// Resolved feedback decay for the current frame (`None` ⇒ no trails).
    feedback_decay: Option<f32>,
}

impl ArtifactRuntime {
    /// Builds a runtime for `art`, pre-allocating every staging buffer to its
    /// contract maximum (no allocation occurs after this on the hot path).
    /// Setting values are initialized from the declared defaults.
    pub fn new(art: LoadedArtifact) -> Self {
        let vm = Vm::new(art.layout.slot_count, art.seed);
        let settings = default_setting_values(&art);
        let compiled_layers = build_compiled_layers(&art);
        let max_col = compiled_layers
            .iter()
            .map(|c| c.prog.num_col())
            .max()
            .unwrap_or(0);

        let mut instances = Vec::new();
        instances.reserve_exact(MAX_INSTANCES);
        let mut points = Vec::new();
        points.reserve_exact(MAX_POINTS);
        let mut cells = Vec::new();
        cells.reserve_exact(MAX_FIELD_CELLS);
        let mut layer_slots = Vec::new();
        layer_slots.reserve_exact(art.programs.layers.len());
        let col_scratch = vec![0.0f64; max_col * MAX_FIELD_RES as usize];

        let mut rt = Self {
            art,
            vm,
            settings,
            compiled_layers,
            col_scratch,
            instances,
            points,
            cells,
            layer_slots,
            feedback_decay: None,
        };
        rt.write_settings();
        rt
    }

    /// The artifact's id, for logging / persistence keys.
    pub fn id(&self) -> &viz_core::ArtifactId {
        &self.art.id
    }

    /// Read access to the live setting values (for panel binding + persistence).
    pub fn settings(&self) -> &IndexMap<String, SettingValue> {
        &self.settings
    }

    /// Sets one setting by name to `value` (a live UI edit). Returns `false`
    /// if the name is unknown or the value kind mismatches the setting's kind.
    pub fn set_setting(&mut self, name: &str, value: SettingValue) -> bool {
        let updated = match self.settings.get_mut(name) {
            Some(slot @ SettingValue::Scalar(_)) if matches!(value, SettingValue::Scalar(_)) => {
                *slot = value;
                true
            }
            Some(slot @ SettingValue::Color { .. })
                if matches!(value, SettingValue::Color { .. }) =>
            {
                *slot = value;
                true
            }
            _ => false,
        };
        if updated {
            if let Some(&slots) = self.art.layout.settings.get(name) {
                write_one_setting(&mut self.vm, slots, value);
            }
        }
        updated
    }

    /// Sets a scalar setting (`number`/`toggle`/`choice`) by name.
    pub fn set_setting_scalar(&mut self, name: &str, value: f64) -> bool {
        self.set_setting(name, SettingValue::Scalar(value))
    }

    /// Sets a `color` setting by name (channels in `0..1`).
    pub fn set_setting_color(&mut self, name: &str, r: f64, g: f64, b: f64, a: f64) -> bool {
        self.set_setting(name, SettingValue::Color { r, g, b, a })
    }

    /// Restores every setting to its declared default (the reset action) and
    /// re-runs [`ArtifactRuntime::activate`] so var state restarts from `init`.
    pub fn reset(&mut self) {
        self.settings = default_setting_values(&self.art);
        self.activate();
    }

    /// Writes the current setting values into their VM slots and runs every
    /// `vars.init` program in declaration order, seeding each var's slot.
    pub fn activate(&mut self) {
        self.write_settings();
        self.vm.set_frame(&FeatureFrame::default());
        let layout = &self.art.layout;
        for (idx, slot) in layout.vars.values().enumerate() {
            let v = self.art.programs.vars_init[idx].value(&mut self.vm);
            self.vm.set_slot(*slot, v);
        }
    }

    /// Evaluates one frame and returns the renderable [`SceneOut`].
    pub fn frame(
        &mut self,
        feature: &FeatureFrame,
        dt: f64,
        aspect: f32,
        detail_factor: f32,
    ) -> SceneOut<'_> {
        // 1) Shared input slots + the feature frame (powers band()/wave()).
        let layout = &self.art.layout;
        self.vm.set_slot(layout.t, feature.t);
        self.vm.set_slot(layout.dt, dt);
        self.vm.set_slot(layout.energy, feature.energy as f64);
        self.vm.set_slot(layout.low, feature.low as f64);
        self.vm.set_slot(layout.mid, feature.mid as f64);
        self.vm.set_slot(layout.high, feature.high as f64);
        self.vm.set_slot(layout.beat, feature.beat as f64);
        self.vm
            .set_slot(layout.beat_count, feature.beat_count as f64);
        self.vm.set_slot(layout.aspect, aspect as f64);
        self.vm.set_frame(feature);

        // 2) Vars, IN DECLARATION ORDER, update-in-place (contract §5 semantics).
        for (idx, slot) in self.art.layout.vars.values().enumerate() {
            let v = self.art.programs.vars_frame[idx].value(&mut self.vm);
            self.vm.set_slot(*slot, v);
        }

        // 3) Per-layer evaluation into the pre-allocated staging buffers.
        self.instances.clear();
        self.points.clear();
        self.cells.clear();
        self.layer_slots.clear();

        let detail = detail_factor.clamp(0.0, 1.0) as f64;
        let n_layers = self.art.programs.layers.len();
        for li in 0..n_layers {
            let slot = self.eval_layer(li, detail);
            self.layer_slots.push(slot);
        }

        // 4) Feedback decay (frame stage).
        self.feedback_decay = match &self.art.programs.feedback_decay {
            Some(p) => Some((p.value(&mut self.vm) as f32).clamp(DECAY_MIN, DECAY_MAX)),
            None => None,
        };

        self.build_scene_out()
    }

    /// Renders this frame's idle/clear scene: no layers, no trails.
    pub fn idle_scene(&mut self) -> SceneOut<'_> {
        self.instances.clear();
        self.points.clear();
        self.cells.clear();
        self.layer_slots.clear();
        self.feedback_decay = None;
        SceneOut {
            params: SceneParams {
                feedback_decay: None,
                ..SceneParams::default()
            },
            layers: Vec::new(),
        }
    }

    /// Whether this artifact requests feedback/trails at all.
    #[allow(dead_code)]
    pub fn has_feedback(&self) -> bool {
        self.art.programs.feedback_decay.is_some()
    }

    // ---- internals -------------------------------------------------------

    /// Writes each live setting value into its VM slot(s).
    fn write_settings(&mut self) {
        for (name, value) in &self.settings {
            let Some(&slots) = self.art.layout.settings.get(name) else {
                continue;
            };
            write_one_setting(&mut self.vm, slots, *value);
        }
    }

    /// Evaluates one layer (`li`) into staging, returning its draw metadata.
    fn eval_layer(&mut self, li: usize, detail: f64) -> LayerSlot {
        let (s_i, s_n, s_u, s_x, s_y) = {
            let l = &self.art.layout;
            (l.i, l.n, l.u, l.x, l.y)
        };
        let scene_layer = &self.art.artifact.scene[li];
        let cl = &self.compiled_layers[li];
        let mut outs = [0.0f64; MAX_MULTI_OUTS];

        match &self.art.programs.layers[li] {
            LayerPrograms::Instanced { count, visible, .. } => {
                if !visible_passes(visible, &mut self.vm) {
                    return LayerSlot::Skipped;
                }
                let raw = count.value(&mut self.vm);
                let n = scale_count(raw, INSTANCE_MIN, MAX_INSTANCES, detail);
                let offset = self.instances.len();
                let denom = (n.saturating_sub(1)).max(1) as f64;
                let inv_denom = 1.0 / denom;

                if cl.reads_n {
                    self.vm.set_slot(s_n, n as f64);
                }
                self.vm.eval_multi_hoisted(&cl.prog, &mut outs);

                let hoisted_col = if cl.color_frame_invariant {
                    pack_color(
                        cl.color_model,
                        outs[5],
                        outs[6],
                        outs[7],
                        outs[8].clamp(0.0, 1.0) as f32,
                    )
                } else {
                    [0.0; 4]
                };

                for i in 0..n {
                    if cl.reads_i {
                        self.vm.set_slot(s_i, i as f64);
                    }
                    if cl.reads_u {
                        self.vm.set_slot(s_u, i as f64 * inv_denom);
                    }
                    self.vm.eval_multi_item(&cl.prog, &mut outs);
                    let col = if cl.color_frame_invariant {
                        hoisted_col
                    } else {
                        pack_color(
                            cl.color_model,
                            outs[5],
                            outs[6],
                            outs[7],
                            outs[8].clamp(0.0, 1.0) as f32,
                        )
                    };
                    self.instances.push(InstanceData {
                        x: outs[0] as f32,
                        y: outs[1] as f32,
                        w: outs[2] as f32,
                        h: outs[3] as f32,
                        rot: outs[4] as f32,
                        color: col,
                    });
                }
                LayerSlot::Instanced {
                    offset,
                    len: n,
                    shape: shape_kind(scene_layer),
                    blend: blend_mode(scene_layer),
                }
            }
            LayerPrograms::Polyline {
                points,
                thickness,
                visible,
                closed,
                ..
            } => {
                if !visible_passes(visible, &mut self.vm) {
                    return LayerSlot::Skipped;
                }
                let raw = points.value(&mut self.vm);
                let n = scale_count(raw, POINT_MIN, MAX_POINTS, detail);
                let thickness_px =
                    (thickness.value(&mut self.vm) as f32).clamp(THICKNESS_MIN, THICKNESS_MAX);
                let offset = self.points.len();
                let denom = (n.saturating_sub(1)).max(1) as f64;
                let inv_denom = 1.0 / denom;

                if cl.reads_n {
                    self.vm.set_slot(s_n, n as f64);
                }
                self.vm.eval_multi_hoisted(&cl.prog, &mut outs);

                let hoisted_col = if cl.color_frame_invariant {
                    pack_color(
                        cl.color_model,
                        outs[2],
                        outs[3],
                        outs[4],
                        outs[5].clamp(0.0, 1.0) as f32,
                    )
                } else {
                    [0.0; 4]
                };

                for i in 0..n {
                    if cl.reads_i {
                        self.vm.set_slot(s_i, i as f64);
                    }
                    if cl.reads_u {
                        self.vm.set_slot(s_u, i as f64 * inv_denom);
                    }
                    self.vm.eval_multi_item(&cl.prog, &mut outs);
                    let col = if cl.color_frame_invariant {
                        hoisted_col
                    } else {
                        pack_color(
                            cl.color_model,
                            outs[2],
                            outs[3],
                            outs[4],
                            outs[5].clamp(0.0, 1.0) as f32,
                        )
                    };
                    self.points.push(PointData {
                        x: outs[0] as f32,
                        y: outs[1] as f32,
                        color: col,
                    });
                }
                LayerSlot::Polyline {
                    offset,
                    len: n,
                    thickness_px,
                    closed: *closed,
                    blend: blend_mode(scene_layer),
                }
            }
            LayerPrograms::Field {
                resolution,
                visible,
                ..
            } => {
                if !visible_passes(visible, &mut self.vm) {
                    return LayerSlot::Skipped;
                }
                let raw = resolution.value(&mut self.vm);
                let res = scale_resolution(raw, detail);
                let res_usize = res as usize;
                let offset = self.cells.len();
                let total = res_usize * res_usize;

                if cl.reads_n {
                    self.vm.set_slot(s_n, total as f64);
                }
                self.vm.eval_multi_hoisted(&cl.prog, &mut outs);

                if cl.all_frame_invariant {
                    let c = pack_color(
                        cl.color_model,
                        outs[0],
                        outs[1],
                        outs[2],
                        outs[3].clamp(0.0, 1.0) as f32,
                    );
                    self.cells.resize(offset + total, c);
                    return LayerSlot::Field {
                        offset,
                        resolution: res,
                        blend: blend_mode(scene_layer),
                    };
                }

                let inv = 2.0 / res as f64;

                if cl.row_only {
                    for row in 0..res {
                        let cy = -1.0 + (row as f64 + 0.5) * inv;
                        self.vm.set_slot(s_y, cy);
                        self.vm.eval_multi_row(&cl.prog, &mut outs);
                        let c = pack_color(
                            cl.color_model,
                            outs[0],
                            outs[1],
                            outs[2],
                            outs[3].clamp(0.0, 1.0) as f32,
                        );
                        let next_len = self.cells.len() + res_usize;
                        self.cells.resize(next_len, c);
                    }
                    return LayerSlot::Field {
                        offset,
                        resolution: res,
                        blend: blend_mode(scene_layer),
                    };
                }

                // Pre-evaluate `x`-only hoisted subexpressions once per column (`res`× instead
                // of `res²`× — e.g. cuts 4 of 5 trig calls in `color-field` by 64×).
                let num_col = cl.prog.num_col();
                if num_col > 0 {
                    for col in 0..res_usize {
                        let cx = -1.0 + (col as f64 + 0.5) * inv;
                        self.vm.set_slot(s_x, cx);
                        let dst = &mut self.col_scratch[col * num_col..(col + 1) * num_col];
                        self.vm.eval_multi_col(&cl.prog, dst);
                    }
                }

                let mut idx = 0usize;
                for row in 0..res {
                    let cy = -1.0 + (row as f64 + 0.5) * inv;
                    if cl.reads_y {
                        self.vm.set_slot(s_y, cy);
                    }
                    self.vm.eval_multi_row(&cl.prog, &mut outs);

                    for col in 0..res_usize {
                        if cl.reads_x {
                            let cx = -1.0 + (col as f64 + 0.5) * inv;
                            self.vm.set_slot(s_x, cx);
                        }
                        if cl.reads_i {
                            self.vm.set_slot(s_i, idx as f64);
                        }
                        if num_col > 0 {
                            let src = &self.col_scratch[col * num_col..(col + 1) * num_col];
                            self.vm.load_multi_col(&cl.prog, src);
                        }
                        self.vm.eval_multi_item(&cl.prog, &mut outs);
                        self.cells.push(pack_color(
                            cl.color_model,
                            outs[0],
                            outs[1],
                            outs[2],
                            outs[3].clamp(0.0, 1.0) as f32,
                        ));
                        idx += 1;
                    }
                }
                LayerSlot::Field {
                    offset,
                    resolution: res,
                    blend: blend_mode(scene_layer),
                }
            }
        }
    }

    /// Builds the borrowed [`SceneOut`] over the just-filled staging buffers.
    fn build_scene_out(&self) -> SceneOut<'_> {
        let mut layers: Vec<LayerDraw> = Vec::with_capacity(self.layer_slots.len());
        for slot in &self.layer_slots {
            match *slot {
                LayerSlot::Skipped => {}
                LayerSlot::Instanced {
                    offset,
                    len,
                    shape,
                    blend,
                } => {
                    layers.push(LayerDraw::Instanced {
                        shape,
                        blend,
                        instances: &self.instances[offset..offset + len],
                    });
                }
                LayerSlot::Polyline {
                    offset,
                    len,
                    thickness_px,
                    closed,
                    blend,
                } => {
                    layers.push(LayerDraw::Polyline {
                        blend,
                        thickness_px,
                        closed,
                        points: &self.points[offset..offset + len],
                    });
                }
                LayerSlot::Field {
                    offset,
                    resolution,
                    blend,
                } => {
                    let total = (resolution * resolution) as usize;
                    layers.push(LayerDraw::Field {
                        blend,
                        resolution,
                        cells: &self.cells[offset..offset + total],
                    });
                }
            }
        }
        SceneOut {
            params: SceneParams {
                feedback_decay: self.feedback_decay,
                ..SceneParams::default()
            },
            layers,
        }
    }
}

/// Writes one setting's value directly into its VM slot(s).
#[inline]
fn write_one_setting(vm: &mut Vm, slots: SettingSlots, value: SettingValue) {
    match (slots, value) {
        (SettingSlots::Scalar(s), SettingValue::Scalar(v)) => {
            vm.set_slot(s, v);
        }
        (
            SettingSlots::Color { r, g, b, a },
            SettingValue::Color {
                r: vr,
                g: vg,
                b: vb,
                a: va,
            },
        ) => {
            vm.set_slot(r, vr);
            vm.set_slot(g, vg);
            vm.set_slot(b, vb);
            vm.set_slot(a, va);
        }
        _ => {}
    }
}

fn formula_to_input(f: &Formula) -> MultiInput<'_> {
    match f.as_expr_or_const() {
        Err(v) => MultiInput::Const(v),
        Ok(s) => MultiInput::Expr(s),
    }
}

fn color_spec_inputs(color: &ColorSpec) -> (ColorModel, [MultiInput<'_>; 4]) {
    match color {
        ColorSpec::Rgba { r, g, b, a } => (
            ColorModel::Rgba,
            [
                formula_to_input(r),
                formula_to_input(g),
                formula_to_input(b),
                a.as_ref()
                    .map(formula_to_input)
                    .unwrap_or(MultiInput::Const(1.0)),
            ],
        ),
        ColorSpec::Hsva { h, s, v, a } => (
            ColorModel::Hsva,
            [
                formula_to_input(h),
                formula_to_input(s),
                formula_to_input(v),
                a.as_ref()
                    .map(formula_to_input)
                    .unwrap_or(MultiInput::Const(1.0)),
            ],
        ),
    }
}

/// Compiles every layer's per-item formulas into a single multi-output [`MultiProgram`]
/// with LICM (frame / row / col hoisting) and cross-formula CSE.
fn build_compiled_layers(art: &LoadedArtifact) -> Vec<CompiledLayer> {
    let scope = art.layout.to_scope();
    let l = &art.layout;
    let elem_item_slots = [l.i, l.u];
    let field_item_slots = [l.i];

    art.artifact
        .scene
        .iter()
        .zip(&art.programs.layers)
        .map(|(layer, layer_progs)| match (layer, layer_progs) {
            (Layer::Instanced { element, .. }, LayerPrograms::Instanced { color, .. }) => {
                let (color_model, [c0, c1, c2, a]) = color_spec_inputs(&element.color);
                let inputs = [
                    formula_to_input(&element.x),
                    formula_to_input(&element.y),
                    formula_to_input(&element.w),
                    formula_to_input(&element.h),
                    element
                        .rot
                        .as_ref()
                        .map(formula_to_input)
                        .unwrap_or(MultiInput::Const(0.0)),
                    c0,
                    c1,
                    c2,
                    a,
                ];
                let prog = compile_multi(&inputs, &scope, &elem_item_slots, None, None)
                    .expect("validated artifact formulas always compile");
                let color_frame_invariant = !color.c0.reads_any_slot(&elem_item_slots)
                    && !color.c1.reads_any_slot(&elem_item_slots)
                    && !color.c2.reads_any_slot(&elem_item_slots)
                    && !color
                        .a
                        .as_ref()
                        .is_some_and(|f| f.reads_any_slot(&elem_item_slots));
                CompiledLayer {
                    reads_i: prog.reads_slot(l.i),
                    reads_n: prog.reads_slot(l.n),
                    reads_u: prog.reads_slot(l.u),
                    reads_x: false,
                    reads_y: false,
                    color_model,
                    color_frame_invariant,
                    all_frame_invariant: prog.item_ops_empty(),
                    row_only: false,
                    prog,
                }
            }
            (Layer::Polyline { point, .. }, LayerPrograms::Polyline { color, .. }) => {
                let (color_model, [c0, c1, c2, a]) = color_spec_inputs(&point.color);
                let inputs = [
                    formula_to_input(&point.x),
                    formula_to_input(&point.y),
                    c0,
                    c1,
                    c2,
                    a,
                ];
                let prog = compile_multi(&inputs, &scope, &elem_item_slots, None, None)
                    .expect("validated artifact formulas always compile");
                let color_frame_invariant = !color.c0.reads_any_slot(&elem_item_slots)
                    && !color.c1.reads_any_slot(&elem_item_slots)
                    && !color.c2.reads_any_slot(&elem_item_slots)
                    && !color
                        .a
                        .as_ref()
                        .is_some_and(|f| f.reads_any_slot(&elem_item_slots));
                CompiledLayer {
                    reads_i: prog.reads_slot(l.i),
                    reads_n: prog.reads_slot(l.n),
                    reads_u: prog.reads_slot(l.u),
                    reads_x: false,
                    reads_y: false,
                    color_model,
                    color_frame_invariant,
                    all_frame_invariant: prog.item_ops_empty(),
                    row_only: false,
                    prog,
                }
            }
            (Layer::Field { cell, .. }, LayerPrograms::Field { .. }) => {
                let (color_model, inputs) = color_spec_inputs(&cell.color);
                let prog = compile_multi(&inputs, &scope, &field_item_slots, Some(l.y), Some(l.x))
                    .expect("validated artifact formulas always compile");
                let all_frame_invariant =
                    prog.item_ops_empty() && !prog.has_row_ops() && !prog.has_col_ops();
                let row_only = prog.item_ops_empty() && prog.has_row_ops() && !prog.has_col_ops();
                CompiledLayer {
                    reads_i: prog.reads_slot(l.i),
                    reads_n: prog.reads_slot(l.n),
                    reads_u: false,
                    reads_x: prog.reads_slot(l.x),
                    reads_y: prog.reads_slot(l.y),
                    color_model,
                    color_frame_invariant: all_frame_invariant,
                    all_frame_invariant,
                    row_only,
                    prog,
                }
            }
            _ => unreachable!("scene layers and compiled layer programs always match"),
        })
        .collect()
}

/// Packs evaluated channel scalars into a straight-alpha RGBA `[f32; 4]`.
#[inline]
fn pack_color(model: ColorModel, c0: f64, c1: f64, c2: f64, a: f32) -> [f32; 4] {
    match model {
        ColorModel::Rgba => [
            c0.clamp(0.0, 1.0) as f32,
            c1.clamp(0.0, 1.0) as f32,
            c2.clamp(0.0, 1.0) as f32,
            a,
        ],
        ColorModel::Hsva => {
            // Fast branchless HSV→RGB in f32 (hue wraps mod 360; s/v clamp 0..1).
            let s = c1.clamp(0.0, 1.0) as f32;
            let v = c2.clamp(0.0, 1.0) as f32;
            if s <= 0.0 {
                return [v, v, v, a];
            }
            let mut h = (c0 * (1.0 / 360.0)).fract() as f32;
            if h < 0.0 {
                h += 1.0;
            }
            let (r, g, b) = hsv_to_rgb_f32(h, s, v);
            [r, g, b, a]
        }
    }
}

/// Branchless HSV→RGB (`h` in `[0, 1)`, `s`/`v` in `[0, 1]`) → `(r, g, b)` in `[0, 1]`.
#[inline]
fn hsv_to_rgb_f32(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let f = |offset: f32| {
        let x = h + offset;
        let frac = x - x.floor();
        let p = (frac * 6.0 - 3.0).abs() - 1.0;
        v * (1.0 + s * (p.clamp(0.0, 1.0) - 1.0))
    };
    (f(0.0), f(2.0 / 3.0), f(1.0 / 3.0))
}

/// Evaluates an optional `visible` gate: absent ⇒ visible; present ⇒ `>= 0.5`.
fn visible_passes(visible: &Option<CompiledFormula>, vm: &mut Vm) -> bool {
    match visible {
        None => true,
        Some(p) => p.value(vm) >= 0.5,
    }
}

/// Floors a raw count, clamps it into `[min, max]`, then scales by `detail`
/// (degradation), never dropping below `min`.
fn scale_count(raw: f64, min: usize, max: usize, detail: f64) -> usize {
    let floored = raw.floor();
    let base = if floored.is_finite() {
        (floored as i64).clamp(min as i64, max as i64) as usize
    } else {
        min
    };
    let scaled = ((base as f64) * detail).floor() as i64;
    scaled.clamp(min as i64, max as i64) as usize
}

/// Floors a raw resolution, clamps 2..=128, then scales by `detail` (min 2).
fn scale_resolution(raw: f64, detail: f64) -> u32 {
    let floored = raw.floor();
    let base = if floored.is_finite() {
        (floored as i64).clamp(FIELD_RES_MIN as i64, MAX_FIELD_RES as i64) as u32
    } else {
        FIELD_RES_MIN
    };
    let scaled = ((base as f64) * detail).floor() as i64;
    scaled.clamp(FIELD_RES_MIN as i64, MAX_FIELD_RES as i64) as u32
}

/// Maps the typed [`Shape`] of an `instanced` layer to a [`ShapeKind`].
fn shape_kind(layer: &Layer) -> ShapeKind {
    match layer {
        Layer::Instanced { shape, .. } => match shape {
            Shape::Rect => ShapeKind::Rect,
            Shape::Circle => ShapeKind::Circle,
            Shape::Triangle => ShapeKind::Triangle,
        },
        _ => ShapeKind::Rect,
    }
}

/// Maps the typed [`Blend`] of a layer to a [`BlendMode`].
fn blend_mode(layer: &Layer) -> BlendMode {
    let blend = match layer {
        Layer::Instanced { blend, .. } => blend,
        Layer::Polyline { blend, .. } => blend,
        Layer::Field { blend, .. } => blend,
    };
    match blend {
        Blend::Alpha => BlendMode::Alpha,
        Blend::Add => BlendMode::Add,
    }
}

/// Builds the default [`SettingValue`] map from an artifact's declarations, in
/// declaration order (see `docs/reference/artifact-contract.md` §4).
fn default_setting_values(art: &LoadedArtifact) -> IndexMap<String, SettingValue> {
    let mut out = IndexMap::with_capacity(art.artifact.settings.len());
    for (name, decl) in &art.artifact.settings {
        out.insert(name.clone(), default_setting_value(decl));
    }
    out
}

/// The declared-default [`SettingValue`] for a single [`SettingDecl`] (contract §4).
pub fn default_setting_value(decl: &viz_contract::SettingDecl) -> SettingValue {
    use viz_contract::SettingDecl;
    match decl {
        SettingDecl::Number { default, .. } => SettingValue::Scalar(*default),
        SettingDecl::Toggle { default, .. } => {
            SettingValue::Scalar(if *default { 1.0 } else { 0.0 })
        }
        SettingDecl::Choice {
            options, default, ..
        } => {
            let idx = options.iter().position(|o| o == default).unwrap_or(0);
            SettingValue::Scalar(idx as f64)
        }
        SettingDecl::Color { default, .. } => {
            let (r, g, b, a) = parse_hex_color(default);
            SettingValue::Color { r, g, b, a }
        }
    }
}

/// Parses a `#RRGGBB` / `#RRGGBBAA` hex color into straight-alpha `0..1` channels.
fn parse_hex_color(s: &str) -> (f64, f64, f64, f64) {
    let body = s.strip_prefix('#').unwrap_or(s);
    let byte = |i: usize| -> f64 {
        u8::from_str_radix(&body[i * 2..i * 2 + 2], 16).unwrap_or(255) as f64 / 255.0
    };
    match body.len() {
        6 => (byte(0), byte(1), byte(2), 1.0),
        8 => (byte(0), byte(1), byte(2), byte(3)),
        _ => (1.0, 1.0, 1.0, 1.0),
    }
}
