//! Bytecode and the stack-machine evaluator.
//!
//! A [`Program`] is a compact flat `Vec<Op>` (4 bytes/instruction) plus a
//! deduplicated `Vec<f64>` constant pool produced by [`compile`](crate::compile).
//! [`Vm::eval`] walks it over a fixed-size `[f64; STACK_SIZE]` stack with **zero
//! heap allocation** and **never panics**: the nesting cap (≤32) guarantees the
//! stack never overflows, constants are sanitized at compile time, slots are
//! sanitized on write, and every potentially non-finite op sanitizes its result
//! so NaN/±Inf collapses to `0.0` at the step that produced it.

use crate::builtins::BuiltinOp;
use viz_core::FeatureFrame;

/// Stack depth. The compiler enforces nesting ≤ 32; an expression tree of depth
/// `d` needs at most `d + 1` stack slots, so 64 is comfortably sufficient and
/// keeps the stack a fixed, cache-friendly array (no per-frame allocation).
pub const STACK_SIZE: usize = 64;

/// Maximum temporary slots available for hoisted + CSE subexpressions per layer.
pub const MAX_TEMPS: usize = 64;

/// Maximum outputs supported by a single [`MultiProgram`] (e.g. Instanced layer:
/// x, y, w, h, rot, c0, c1, c2, a = 9).
pub const MAX_MULTI_OUTS: usize = 10;

/// One flat 4-byte bytecode instruction. Slot and constant indices are resolved
/// at compile time so the hot loop uses a single flat jump table with zero
/// nested matching and zero argument-slice copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Push pre-sanitized constant `program.consts[idx]`.
    Const(u16),
    /// Push the current (pre-sanitized) value of slot `n`.
    Load(u16),
    /// Peek top-of-stack (without popping) and store into `temps[idx]`.
    StoreTemp(u16),
    /// Pop top-of-stack and store into `temps[idx]` (used by hoisted/row/col streams).
    PopTemp(u16),
    /// Push `temps[idx]` onto the stack.
    LoadTemp(u16),
    /// Pop top-of-stack and store into `outs[idx]` (used by [`MultiProgram`]).
    StoreOut(u16),
    // Unary
    Neg,
    Not,
    // Binary
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    /// `x * x` (algebraic fast-path for `pow(x, 2)` / `x ^ 2`).
    Square,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    // Builtins (flat, in-place on the stack)
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Exp,
    Log,
    Log2,
    Log10,
    Sqrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Fract,
    Min,
    Max,
    Clamp,
    Mix,
    Smoothstep,
    Step,
    If,
    Band,
    Wave,
    Rand,
}

const _: () = assert!(std::mem::size_of::<Op>() <= 4);

/// Unary VM operators (kept public for the property test's reference evaluator).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    /// Arithmetic negation.
    Neg,
    /// Logical NOT (truthy when operand `>= 0.5`; yields 0/1).
    Not,
}

/// Binary VM operators (kept public for the property test's reference evaluator).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

/// A compiled formula: flat 4-byte bytecode, deduplicated pre-sanitized constant
/// pool, syntactic op count (for contract caps), and maximum referenced slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub(crate) ops: Vec<Op>,
    pub(crate) consts: Vec<f64>,
    pub(crate) raw_op_count: usize,
    pub(crate) max_slot: Option<u16>,
}

impl Program {
    /// Number of syntactic operations before constant folding (the cap-enforced
    /// contract metric — ≤ 256 per formula, ≤ 4096 per artifact).
    pub fn op_count(&self) -> usize {
        self.raw_op_count
    }

    /// Read-only view of the optimized bytecode (used by tests and tooling).
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// If this program folded to a single constant value, returns `Some(value)`.
    #[inline]
    pub fn const_value(&self) -> Option<f64> {
        match self.ops.as_slice() {
            [Op::Const(idx)] => self.consts.get(*idx as usize).copied(),
            _ => None,
        }
    }

    /// Returns `true` if the optimized bytecode loads any slot in `slots`.
    pub fn reads_any_slot(&self, slots: &[crate::SlotId]) -> bool {
        self.ops.iter().any(|op| match *op {
            Op::Load(id) => slots.iter().any(|s| s.0 == id),
            _ => false,
        })
    }
}

/// Sanitizes a value so the engine can never produce a non-finite result: any
/// non-finite value (NaN, ±Inf) becomes `0.0`.
///
/// Public so the property test's reference evaluator applies the identical
/// sanitization at every step and matches the VM bitwise.
#[inline]
pub fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// True if a scalar is "truthy" for logical operators (contract §2.1: `>= 0.5`).
#[inline]
fn truthy(x: f64) -> bool {
    x >= 0.5
}

/// A compiled multi-output layer program produced by [`crate::compile_multi`].
///
/// Combines all per-item formulas of one layer into shared bytecode with:
/// - `hoisted_ops`: frame-invariant subexpressions (run once per frame → `temps[0..row_base]`)
/// - `row_ops`: `y`-only subexpressions for `Field` (run once per row → `temps[row_base..col_base]`)
/// - `col_ops`: `x`-only subexpressions for `Field` (run once per col → `outs[0..num_col]`,
///   then copied into `temps[col_base..col_base + num_col]` per cell)
/// - `item_ops`: remaining per-item work with cross-formula CSE (`StoreTemp`/`LoadTemp`),
///   writing each layer output via `StoreOut(idx)`.
#[derive(Clone, Debug)]
pub struct MultiProgram {
    pub(crate) hoisted_ops: Vec<Op>,
    pub(crate) row_ops: Vec<Op>,
    pub(crate) col_ops: Vec<Op>,
    pub(crate) item_ops: Vec<Op>,
    pub(crate) consts: Vec<f64>,
    pub(crate) max_slot: Option<u16>,
    pub(crate) col_base: usize,
    pub(crate) num_col: usize,
    pub(crate) num_outputs: usize,
    /// Outputs that are already frame-invariant (stored in `hoisted_ops` or constants):
    /// `fixed_outs[idx] = Some(temp_slot)` or `None` if written by `item_ops`.
    pub(crate) const_outs: [Option<f64>; MAX_MULTI_OUTS],
    pub(crate) reads_slots: Vec<u16>,
}

impl MultiProgram {
    /// Number of `x`-only hoisted values per column (for `Field` layers).
    #[inline]
    pub fn num_col(&self) -> usize {
        self.num_col
    }

    /// True if `item_ops` is empty (all outputs are frame-invariant or constant).
    #[inline]
    pub fn item_ops_empty(&self) -> bool {
        self.item_ops.is_empty()
    }

    /// True if `row_ops` is non-empty.
    #[inline]
    pub fn has_row_ops(&self) -> bool {
        !self.row_ops.is_empty()
    }

    /// True if `col_ops` is non-empty.
    #[inline]
    pub fn has_col_ops(&self) -> bool {
        !self.col_ops.is_empty()
    }

    /// Returns `true` if any phase of this multi-program loads `slot`.
    #[inline]
    pub fn reads_slot(&self, slot: crate::SlotId) -> bool {
        self.reads_slots.contains(&slot.0)
    }

    /// Returns the number of compiled outputs.
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

/// The formula evaluator. Holds the slot bank, the latest [`FeatureFrame`] for
/// `band()`/`wave()`, and the artifact seed for `rand()`. One `Vm` is reused
/// across frames and elements (no per-frame allocation): set
/// the shared/element slots and the frame, then call [`Vm::eval`] repeatedly.
pub struct Vm {
    /// Slot bank: one pre-sanitized f64 per declared identifier in [`Scope`](crate::Scope).
    slots: Vec<f64>,
    /// Latest audio features (powers `band(u)`/`wave(u)`).
    frame: FeatureFrame,
    /// Artifact seed for deterministic `rand(k)`.
    seed: u64,
    /// Fixed-size evaluation stack (never heap-allocated per call).
    stack: [f64; STACK_SIZE],
    /// Fixed-size temporary bank for hoisted + CSE subexpressions.
    temps: [f64; MAX_TEMPS],
}

impl Vm {
    /// Creates a VM with `slot_count` zeroed slots and the artifact `seed`.
    pub fn new(slot_count: usize, seed: u64) -> Self {
        Self {
            slots: vec![0.0; slot_count],
            frame: FeatureFrame::default(),
            seed,
            stack: [0.0; STACK_SIZE],
            temps: [0.0; MAX_TEMPS],
        }
    }

    /// Number of slots this VM was constructed with.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Writes a slot value (sanitized to finite on write so `Op::Load` skips
    /// per-read sanitization). Out-of-range ids are ignored.
    #[inline]
    pub fn set_slot(&mut self, slot: crate::SlotId, value: f64) {
        if let Some(s) = self.slots.get_mut(slot.0 as usize) {
            *s = sanitize(value);
        }
    }

    /// Reads a slot value (0.0 if the id is out of range).
    #[inline]
    pub fn get_slot(&self, slot: crate::SlotId) -> f64 {
        self.slots.get(slot.0 as usize).copied().unwrap_or(0.0)
    }

    /// Installs the latest audio feature frame (POD copy). Powers `band()` and
    /// `wave()` for subsequent evaluations.
    #[inline]
    pub fn set_frame(&mut self, frame: &FeatureFrame) {
        self.frame = *frame;
    }

    /// Deterministic stateless `rand(k)`: a splitmix64-style hash mixing the
    /// artifact `seed` with `k.to_bits()`, finished into a uniform value in
    /// `[0, 1)`. Same `(seed, k)` always yields the same value — stable across
    /// frames, elements, and VM instances unless `k` changes. Public so the
    /// reference evaluator reproduces it exactly.
    #[inline]
    pub fn rand(&self, k: f64) -> f64 {
        // Normalize the bit pattern so that -0.0 and +0.0 (and any NaN) map to a
        // single stable key: callers reason about `rand(k)` numerically, not by
        // bit pattern.
        let key = if k == 0.0 { 0.0 } else { k };
        let kbits = if key.is_nan() { 0 } else { key.to_bits() };
        let mut z = self.seed ^ kbits.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // splitmix64 finisher.
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 53 bits → uniform [0, 1) (the standard f64 construction).
        ((z >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Evaluates a compiled [`Program`] against the current slots/frame/seed.
    /// Returns the single result, guaranteed finite. Never panics.
    #[inline]
    pub fn eval(&mut self, program: &Program) -> f64 {
        let ops = program.ops.as_slice();
        let consts = program.consts.as_slice();
        if ops.len() == 1 {
            return match ops[0] {
                Op::Const(idx) => consts.get(idx as usize).copied().unwrap_or(0.0),
                Op::Load(n) => self.slots.get(n as usize).copied().unwrap_or(0.0),
                _ => 0.0,
            };
        }

        let slots = self.slots.as_slice();
        let slots_in_bounds = program.max_slot.is_none_or(|m| (m as usize) < slots.len());
        let mut dummy_outs = [0.0f64; MAX_MULTI_OUTS];
        run_ops(
            ops,
            consts,
            slots,
            slots_in_bounds,
            &mut self.stack,
            &mut self.temps,
            &mut dummy_outs,
            &self.frame,
            self.seed,
        )
    }

    /// Evaluates frame-invariant hoisted subexpressions for `mp` once per frame,
    /// and seeds `outs` with any constant or frame-invariant outputs.
    #[inline]
    pub fn eval_multi_hoisted(&mut self, mp: &MultiProgram, outs: &mut [f64; MAX_MULTI_OUTS]) {
        for (dst, val) in outs.iter_mut().zip(&mp.const_outs) {
            if let Some(v) = *val {
                *dst = v;
            }
        }
        if mp.hoisted_ops.is_empty() {
            return;
        }
        let slots = self.slots.as_slice();
        let slots_in_bounds = mp.max_slot.is_none_or(|m| (m as usize) < slots.len());
        run_ops(
            &mp.hoisted_ops,
            &mp.consts,
            slots,
            slots_in_bounds,
            &mut self.stack,
            &mut self.temps,
            outs,
            &self.frame,
            self.seed,
        );
    }

    /// Evaluates `y`-only row-invariant subexpressions for a `Field` layer once per row.
    #[inline]
    pub fn eval_multi_row(&mut self, mp: &MultiProgram, outs: &mut [f64; MAX_MULTI_OUTS]) {
        if mp.row_ops.is_empty() {
            return;
        }
        let slots = self.slots.as_slice();
        let slots_in_bounds = mp.max_slot.is_none_or(|m| (m as usize) < slots.len());
        run_ops(
            &mp.row_ops,
            &mp.consts,
            slots,
            slots_in_bounds,
            &mut self.stack,
            &mut self.temps,
            outs,
            &self.frame,
            self.seed,
        );
    }

    /// Evaluates `x`-only column-invariant subexpressions for one column into `col_dst`.
    #[inline]
    pub fn eval_multi_col(&mut self, mp: &MultiProgram, col_dst: &mut [f64]) {
        if mp.col_ops.is_empty() {
            return;
        }
        let slots = self.slots.as_slice();
        let slots_in_bounds = mp.max_slot.is_none_or(|m| (m as usize) < slots.len());
        let mut dummy_outs = [0.0f64; MAX_MULTI_OUTS];
        run_ops(
            &mp.col_ops,
            &mp.consts,
            slots,
            slots_in_bounds,
            &mut self.stack,
            &mut self.temps,
            &mut dummy_outs,
            &self.frame,
            self.seed,
        );
        let src = &self.temps[mp.col_base..mp.col_base + mp.num_col];
        col_dst[..mp.num_col].copy_from_slice(src);
    }

    /// Loads precomputed column values for `col` into `self.temps` before `eval_multi_item`.
    #[inline]
    pub fn load_multi_col(&mut self, mp: &MultiProgram, col_src: &[f64]) {
        if mp.num_col > 0 {
            self.temps[mp.col_base..mp.col_base + mp.num_col]
                .copy_from_slice(&col_src[..mp.num_col]);
        }
    }

    /// Evaluates per-item bytecode for `mp`, writing all varying outputs into `outs`.
    #[inline]
    pub fn eval_multi_item(&mut self, mp: &MultiProgram, outs: &mut [f64; MAX_MULTI_OUTS]) {
        if mp.item_ops.is_empty() {
            return;
        }
        let slots = self.slots.as_slice();
        let slots_in_bounds = mp.max_slot.is_none_or(|m| (m as usize) < slots.len());
        run_ops(
            &mp.item_ops,
            &mp.consts,
            slots,
            slots_in_bounds,
            &mut self.stack,
            &mut self.temps,
            outs,
            &self.frame,
            self.seed,
        );
    }

    /// Applies a built-in to its operand slice `a` (length == the op's arity).
    /// All inputs are already sanitized (finite); the caller sanitizes the
    /// result. Public so the property test's reference evaluator uses identical
    /// semantics, including the frame-dependent `band`/`wave`/`rand`.
    #[inline]
    pub fn apply_builtin(&self, b: BuiltinOp, a: &[f64]) -> f64 {
        match b {
            BuiltinOp::Band => self.frame.band(a[0]),
            BuiltinOp::Wave => self.frame.wave(a[0]),
            BuiltinOp::Rand => self.rand(a[0]),
            other => apply_pure_builtin(other, a).unwrap_or(0.0),
        }
    }
}

/// Core flat bytecode interpreter shared by single-output [`Vm::eval`] and
/// multi-output [`MultiProgram`] execution.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_ops(
    ops: &[Op],
    consts: &[f64],
    slots: &[f64],
    slots_in_bounds: bool,
    stack: &mut [f64; STACK_SIZE],
    temps: &mut [f64; MAX_TEMPS],
    outs: &mut [f64; MAX_MULTI_OUTS],
    frame: &FeatureFrame,
    seed: u64,
) -> f64 {
    let mut sp: usize = 0;

    for &op in ops {
        match op {
            Op::Const(idx) => {
                if sp < STACK_SIZE {
                    stack[sp] = consts[idx as usize];
                    sp += 1;
                }
            }
            Op::Load(n) => {
                let v = if slots_in_bounds {
                    slots[n as usize]
                } else {
                    slots.get(n as usize).copied().unwrap_or(0.0)
                };
                if sp < STACK_SIZE {
                    stack[sp] = v;
                    sp += 1;
                }
            }
            Op::StoreTemp(idx) => {
                if sp >= 1 && (idx as usize) < MAX_TEMPS {
                    temps[idx as usize] = stack[sp - 1];
                }
            }
            Op::PopTemp(idx) => {
                if sp >= 1 && (idx as usize) < MAX_TEMPS {
                    sp -= 1;
                    temps[idx as usize] = stack[sp];
                }
            }
            Op::LoadTemp(idx) => {
                if sp < STACK_SIZE && (idx as usize) < MAX_TEMPS {
                    stack[sp] = temps[idx as usize];
                    sp += 1;
                }
            }
            Op::StoreOut(idx) => {
                if sp >= 1 && (idx as usize) < MAX_MULTI_OUTS {
                    sp -= 1;
                    outs[idx as usize] = stack[sp];
                }
            }
            Op::Neg => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = -stack[sp - 1];
            }
            Op::Not => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = bool_val(!truthy(stack[sp - 1]));
            }
            Op::Add => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = sanitize(stack[sp - 1] + b);
            }
            Op::Sub => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = sanitize(stack[sp - 1] - b);
            }
            Op::Mul => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = sanitize(stack[sp - 1] * b);
            }
            Op::Div => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = sanitize(stack[sp - 1] / b);
            }
            Op::Rem => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                let a = stack[sp - 1];
                // Fast path for exact non-negative u32 integers (e.g. `i % 18`, `i % 20`):
                // avoids libsystem_m `fmod` (~80–150 cycles) while remaining bitwise
                // identical (including `-0.0` sign bit preservation).
                stack[sp - 1] =
                    if b > 0.0 && (0.0..4_294_967_296.0).contains(&a) && b < 4_294_967_296.0 {
                        let ai = a as u32;
                        let bi = b as u32;
                        if ai as f64 == a && bi as f64 == b && bi != 0 {
                            if a == 0.0 {
                                a
                            } else {
                                (ai % bi) as f64
                            }
                        } else {
                            sanitize(a % b)
                        }
                    } else {
                        sanitize(a % b)
                    };
            }
            Op::Pow => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = sanitize(stack[sp - 1].powf(b));
            }
            Op::Square => {
                if sp < 1 {
                    return 0.0;
                }
                let a = stack[sp - 1];
                stack[sp - 1] = sanitize(a * a);
            }
            Op::Lt => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] < b);
            }
            Op::Le => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] <= b);
            }
            Op::Gt => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] > b);
            }
            Op::Ge => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] >= b);
            }
            Op::Eq => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] == b);
            }
            Op::Ne => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(stack[sp - 1] != b);
            }
            Op::And => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(truthy(stack[sp - 1]) && truthy(b));
            }
            Op::Or => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = bool_val(truthy(stack[sp - 1]) || truthy(b));
            }
            Op::Sin => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].sin();
            }
            Op::Cos => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].cos();
            }
            Op::Tan => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].tan());
            }
            Op::Asin => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].asin());
            }
            Op::Acos => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].acos());
            }
            Op::Atan => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].atan();
            }
            Op::Atan2 => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = stack[sp - 1].atan2(b);
            }
            Op::Exp => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].exp());
            }
            Op::Log => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].ln());
            }
            Op::Log2 => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].log2());
            }
            Op::Log10 => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].log10());
            }
            Op::Sqrt => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sanitize(stack[sp - 1].sqrt());
            }
            Op::Abs => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].abs();
            }
            Op::Sign => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = sign(stack[sp - 1]);
            }
            Op::Floor => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].floor();
            }
            Op::Ceil => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].ceil();
            }
            Op::Round => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].round();
            }
            Op::Fract => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = stack[sp - 1].fract();
            }
            Op::Min => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = stack[sp - 1].min(b);
            }
            Op::Max => {
                if sp < 2 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = stack[sp - 1].max(b);
            }
            Op::Clamp => {
                if sp < 3 {
                    return 0.0;
                }
                let hi = stack[sp - 1];
                let lo = stack[sp - 2];
                let x = stack[sp - 3];
                sp -= 2;
                stack[sp - 1] = clamp(x, lo, hi);
            }
            Op::Mix => {
                if sp < 3 {
                    return 0.0;
                }
                let t = stack[sp - 1];
                let b = stack[sp - 2];
                let a = stack[sp - 3];
                sp -= 2;
                stack[sp - 1] = sanitize(mix(a, b, t));
            }
            Op::Smoothstep => {
                if sp < 3 {
                    return 0.0;
                }
                let x = stack[sp - 1];
                let e1 = stack[sp - 2];
                let e0 = stack[sp - 3];
                sp -= 2;
                stack[sp - 1] = sanitize(smoothstep(e0, e1, x));
            }
            Op::Step => {
                if sp < 2 {
                    return 0.0;
                }
                let x = stack[sp - 1];
                sp -= 1;
                stack[sp - 1] = step(stack[sp - 1], x);
            }
            Op::If => {
                if sp < 3 {
                    return 0.0;
                }
                let b = stack[sp - 1];
                let a = stack[sp - 2];
                let cond = stack[sp - 3];
                sp -= 2;
                stack[sp - 1] = if truthy(cond) { a } else { b };
            }
            Op::Band => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = frame.band(stack[sp - 1]);
            }
            Op::Wave => {
                if sp < 1 {
                    return 0.0;
                }
                stack[sp - 1] = frame.wave(stack[sp - 1]);
            }
            Op::Rand => {
                if sp < 1 {
                    return 0.0;
                }
                let k = stack[sp - 1];
                let key = if k == 0.0 { 0.0 } else { k };
                let mut z = seed ^ key.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                stack[sp - 1] = ((z >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
            }
        }
    }
    if sp == 0 {
        0.0
    } else {
        stack[sp - 1]
    }
}

/// Evaluates any pure (non-frame, non-seed) builtin on sanitized arguments.
/// Used both by [`Vm::apply_builtin`] and compile-time constant folding.
#[inline]
pub(crate) fn apply_pure_builtin(b: BuiltinOp, a: &[f64]) -> Option<f64> {
    Some(match b {
        BuiltinOp::Sin => a[0].sin(),
        BuiltinOp::Cos => a[0].cos(),
        BuiltinOp::Tan => a[0].tan(),
        BuiltinOp::Asin => a[0].asin(),
        BuiltinOp::Acos => a[0].acos(),
        BuiltinOp::Atan => a[0].atan(),
        BuiltinOp::Atan2 => a[0].atan2(a[1]),
        BuiltinOp::Exp => a[0].exp(),
        BuiltinOp::Log => a[0].ln(),
        BuiltinOp::Log2 => a[0].log2(),
        BuiltinOp::Log10 => a[0].log10(),
        BuiltinOp::Pow => a[0].powf(a[1]),
        BuiltinOp::Sqrt => a[0].sqrt(),
        BuiltinOp::Abs => a[0].abs(),
        BuiltinOp::Sign => sign(a[0]),
        BuiltinOp::Floor => a[0].floor(),
        BuiltinOp::Ceil => a[0].ceil(),
        BuiltinOp::Round => a[0].round(),
        BuiltinOp::Fract => a[0].fract(),
        BuiltinOp::Min => a[0].min(a[1]),
        BuiltinOp::Max => a[0].max(a[1]),
        BuiltinOp::Clamp => clamp(a[0], a[1], a[2]),
        BuiltinOp::Mix => mix(a[0], a[1], a[2]),
        BuiltinOp::Smoothstep => smoothstep(a[0], a[1], a[2]),
        BuiltinOp::Step => step(a[0], a[1]),
        BuiltinOp::If => {
            if truthy(a[0]) {
                a[1]
            } else {
                a[2]
            }
        }
        BuiltinOp::Band | BuiltinOp::Wave | BuiltinOp::Rand => return None,
    })
}

/// Arity of a built-in op (mirrors the table in `builtins`; kept local so the VM
/// has no name lookups in the hot path). Public for the reference evaluator.
#[inline]
pub fn builtin_arity(b: BuiltinOp) -> usize {
    match b {
        BuiltinOp::Atan2 | BuiltinOp::Pow | BuiltinOp::Min | BuiltinOp::Max | BuiltinOp::Step => 2,
        BuiltinOp::Clamp | BuiltinOp::Mix | BuiltinOp::Smoothstep | BuiltinOp::If => 3,
        _ => 1,
    }
}

/// Applies a unary operator (result sanitized by the caller). Public so the
/// reference evaluator shares the exact semantics.
#[inline]
pub fn apply_unary(kind: UnaryKind, a: f64) -> f64 {
    match kind {
        UnaryKind::Neg => -a,
        UnaryKind::Not => bool_val(!truthy(a)),
    }
}

/// Applies a binary operator (result sanitized by the caller). Public so the
/// reference evaluator shares the exact semantics.
#[inline]
pub fn apply_binary(kind: BinaryKind, a: f64, b: f64) -> f64 {
    match kind {
        BinaryKind::Add => a + b,
        BinaryKind::Sub => a - b,
        BinaryKind::Mul => a * b,
        BinaryKind::Div => a / b,
        BinaryKind::Rem => a % b,
        BinaryKind::Pow => a.powf(b),
        BinaryKind::Lt => bool_val(a < b),
        BinaryKind::Le => bool_val(a <= b),
        BinaryKind::Gt => bool_val(a > b),
        BinaryKind::Ge => bool_val(a >= b),
        BinaryKind::Eq => bool_val(a == b),
        BinaryKind::Ne => bool_val(a != b),
        BinaryKind::And => bool_val(truthy(a) && truthy(b)),
        BinaryKind::Or => bool_val(truthy(a) || truthy(b)),
    }
}

/// Maps a boolean to the contract's 0.0/1.0 numeric truth values.
#[inline]
fn bool_val(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// `sign(x)`: −1, 0, or +1. `sign(0) == 0`; non-finite handled by sanitization.
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `clamp(x, lo, hi)`. Tolerates `lo > hi` gracefully (swaps) instead of panicking.
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    x.clamp(lo, hi)
}

/// Linear interpolation `a + (b - a) * t` (unclamped, matching GLSL `mix`).
#[inline]
fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Smooth Hermite interpolation between `edge0` and `edge1` (GLSL `smoothstep`).
/// Equal edges degrade cleanly to `step(edge0, x)` rather than dividing by zero.
#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let span = edge1 - edge0;
    if span == 0.0 {
        return step(edge0, x);
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Step function: `0.0` if `x < edge`, else `1.0` (GLSL `step`).
#[inline]
fn step(edge: f64, x: f64) -> f64 {
    bool_val(x >= edge)
}
