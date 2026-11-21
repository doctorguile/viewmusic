//! Compilation: source → [`Program`] bytecode, with cap enforcement.
//!
//! Pipeline: enforce source length → lex → parse (Pratt, arity-checked) → check
//! nesting depth → resolve every identifier against the [`Scope`] / built-in
//! constants → emit flat bytecode → enforce op count. Any unknown identifier or
//! cap violation is reported with the offending token verbatim so the diagnostic
//! log can point the author at the exact problem.

use std::collections::{HashMap, HashSet};

use crate::builtins::{is_reserved, lookup_constant, BuiltinOp};
use crate::lexer::lex;
use crate::parser::{parse, BinaryOp, Expr, UnaryOp};
use crate::vm::{
    apply_binary, apply_pure_builtin, apply_unary, sanitize, BinaryKind, MultiProgram, Op, Program,
    UnaryKind, MAX_MULTI_OUTS, MAX_TEMPS,
};

/// Maximum formula source length, in characters.
pub const MAX_SOURCE_LEN: usize = 1024;
/// Maximum AST nesting depth (contract §12).
pub const MAX_DEPTH: usize = 32;
/// Maximum compiled operations per formula (contract §12).
pub const MAX_OPS: usize = 256;

/// A compile-time slot index for a declared identifier. `Copy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u16);

/// The set of identifiers a formula may read, mapped to dense slot indices.
///
/// Built once per stage (the host assigns slots for `t`, `dt`, audio aggregates,
/// `settings.*`, vars, and stage extras like `i`/`u`/`n`/`x`/`y`). Built-in
/// function and constant names are reserved and rejected as slot names.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    /// name → slot id, in insertion order (slot ids are assigned densely).
    names: Vec<String>,
}

impl Scope {
    /// Starts building a scope.
    pub fn builder() -> ScopeBuilder {
        ScopeBuilder {
            scope: Scope { names: Vec::new() },
        }
    }

    /// Number of declared slots (the VM's slot-bank size).
    pub fn slot_count(&self) -> usize {
        self.names.len()
    }

    /// Resolves an identifier to its slot, or `None` if it is not declared.
    pub fn slot_id(&self, name: &str) -> Option<SlotId> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| SlotId(i as u16))
    }
}

/// Incremental builder for a [`Scope`].
pub struct ScopeBuilder {
    scope: Scope,
}

impl ScopeBuilder {
    /// Declares a slot named `name`, returning its assigned [`SlotId`].
    ///
    /// Rejects duplicates and names colliding with built-in functions or
    /// constants (contract §4: built-in identifiers are reserved). Also rejects
    /// the empty name and overflow past `u16::MAX` slots.
    pub fn slot(&mut self, name: &str) -> Result<SlotId, ScopeError> {
        if name.is_empty() {
            return Err(ScopeError::EmptyName);
        }
        if is_reserved(name) {
            return Err(ScopeError::ReservedName(name.to_owned()));
        }
        if self.scope.names.iter().any(|n| n == name) {
            return Err(ScopeError::Duplicate(name.to_owned()));
        }
        if self.scope.names.len() >= u16::MAX as usize {
            return Err(ScopeError::TooManySlots);
        }
        let id = SlotId(self.scope.names.len() as u16);
        self.scope.names.push(name.to_owned());
        Ok(id)
    }

    /// Finalizes the scope.
    pub fn build(self) -> Scope {
        self.scope
    }
}

/// Failure declaring a slot in a [`ScopeBuilder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeError {
    /// The name was already declared in this scope.
    Duplicate(String),
    /// The name collides with a built-in function or constant (reserved).
    ReservedName(String),
    /// An empty slot name was supplied.
    EmptyName,
    /// More than `u16::MAX` slots were declared.
    TooManySlots,
}

impl std::fmt::Display for ScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeError::Duplicate(n) => write!(f, "duplicate slot name '{n}'"),
            ScopeError::ReservedName(n) => {
                write!(
                    f,
                    "slot name '{n}' collides with a built-in function or constant"
                )
            }
            ScopeError::EmptyName => f.write_str("slot name must not be empty"),
            ScopeError::TooManySlots => f.write_str("too many slots declared"),
        }
    }
}

impl std::error::Error for ScopeError {}

/// A compilation failure. `token` and `position` name the offending source text
/// verbatim so the diagnostics log can point the author at the exact problem.
#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    /// The error classification.
    pub kind: CompileErrorKind,
    /// Offending token text, verbatim (named in the diagnostics log entry).
    pub token: String,
    /// Byte offset of the offending token within the source.
    pub position: usize,
}

/// Classifications of compile failures.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileErrorKind {
    /// Source exceeded [`MAX_SOURCE_LEN`] characters.
    SourceTooLong,
    /// A lexing error (unexpected character / malformed number).
    Lex,
    /// A parse/grammar error (includes arity and unknown-function errors).
    Parse,
    /// AST nesting exceeded [`MAX_DEPTH`].
    NestingTooDeep,
    /// An identifier is neither a declared slot nor a built-in constant.
    UnknownIdentifier,
    /// Compiled bytecode exceeded [`MAX_OPS`] operations.
    TooManyOps,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            CompileErrorKind::SourceTooLong => write!(
                f,
                "formula source exceeds {MAX_SOURCE_LEN} characters (at position {})",
                self.position
            ),
            CompileErrorKind::Lex => {
                write!(
                    f,
                    "unexpected character '{}' at position {}",
                    self.token, self.position
                )
            }
            CompileErrorKind::Parse => {
                write!(
                    f,
                    "syntax error near '{}' at position {}",
                    self.token, self.position
                )
            }
            CompileErrorKind::NestingTooDeep => write!(
                f,
                "expression nesting exceeds depth {MAX_DEPTH} near '{}' at position {}",
                self.token, self.position
            ),
            CompileErrorKind::UnknownIdentifier => write!(
                f,
                "unknown identifier '{}' at position {}",
                self.token, self.position
            ),
            CompileErrorKind::TooManyOps => {
                write!(f, "formula compiles to more than {MAX_OPS} operations")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Compiles `src` against `scope` into a [`Program`], enforcing all load-time
/// caps. Returns a [`CompileError`] naming the offending token on any failure.
pub fn compile(src: &str, scope: &Scope) -> Result<Program, CompileError> {
    // Cap 1: source length (counted in characters per the contract).
    let char_len = src.chars().count();
    if char_len > MAX_SOURCE_LEN {
        return Err(CompileError {
            kind: CompileErrorKind::SourceTooLong,
            token: String::new(),
            position: src.len(),
        });
    }

    // Lex.
    let tokens = lex(src).map_err(|e| CompileError {
        kind: CompileErrorKind::Lex,
        token: e.token,
        position: e.position,
    })?;

    // Parse (also checks function arity and unknown functions).
    let ast = parse(&tokens).map_err(|e| CompileError {
        kind: CompileErrorKind::Parse,
        token: e.token,
        position: e.position,
    })?;

    // Cap 2: nesting depth. We measure nesting as the **evaluation-stack depth**
    // the expression requires — the metric that actually bounds the VM's fixed
    // `[f64; STACK_SIZE]` stack. A long left-associative chain (`1+1+…+1`) stays
    // shallow (depth 2) and is bounded instead by the op-count cap; right-nested
    // or genuinely deep trees grow the stack and are caught here. With
    // MAX_DEPTH = 32 ≤ STACK_SIZE = 64, the stack can never overflow.
    check_depth(&ast)?;

    // Cap 3: enforce syntactic op cap BEFORE constant folding so hostile
    // constant-expression bombs are still rejected per contract §12.
    let raw_op_count = count_ast_ops(&ast);
    if raw_op_count > MAX_OPS {
        return Err(CompileError {
            kind: CompileErrorKind::TooManyOps,
            token: String::new(),
            position: 0,
        });
    }

    // Validate identifiers + fold constant subtrees (preserves bitwise exact
    // semantics with the reference evaluator).
    let folded = fold_expr(&ast, scope)?;

    // Emit compact flat bytecode + deduplicated constant pool.
    let mut ops = Vec::with_capacity(raw_op_count);
    let mut consts = Vec::new();
    let mut max_slot: Option<u16> = None;
    emit(&folded, scope, &mut ops, &mut consts, &mut max_slot)?;

    Ok(Program {
        ops,
        consts,
        raw_op_count,
        max_slot,
    })
}

/// Counts the syntactic VM operations an unoptimized AST represents.
fn count_ast_ops(expr: &Expr) -> usize {
    match expr {
        Expr::Num(_) | Expr::Ident(_) => 1,
        Expr::Unary(_, a) => 1 + count_ast_ops(a),
        Expr::Binary(_, a, b) => 1 + count_ast_ops(a) + count_ast_ops(b),
        Expr::Call(_, args) => 1 + args.iter().map(count_ast_ops).sum::<usize>(),
    }
}

/// Validates all identifiers against `scope` and folds pure constant subtrees
/// using the exact same sanitized scalar semantics as [`crate::Vm`].
fn fold_expr(expr: &Expr, scope: &Scope) -> Result<Expr, CompileError> {
    match expr {
        Expr::Num(v) => Ok(Expr::Num(sanitize(*v))),
        Expr::Ident(name) => {
            if scope.slot_id(name).is_some() {
                Ok(Expr::Ident(name.clone()))
            } else if let Some(value) = lookup_constant(name) {
                Ok(Expr::Num(sanitize(value)))
            } else {
                Err(CompileError {
                    kind: CompileErrorKind::UnknownIdentifier,
                    token: name.clone(),
                    position: 0,
                })
            }
        }
        Expr::Unary(op, a) => {
            let a = fold_expr(a, scope)?;
            if let Expr::Num(v) = a {
                let kind = match op {
                    UnaryOp::Neg => UnaryKind::Neg,
                    UnaryOp::Not => UnaryKind::Not,
                };
                return Ok(Expr::Num(sanitize(apply_unary(kind, v))));
            }
            Ok(Expr::Unary(*op, Box::new(a)))
        }
        Expr::Binary(op, a, b) => {
            let a = fold_expr(a, scope)?;
            let b = fold_expr(b, scope)?;
            if let (Expr::Num(va), Expr::Num(vb)) = (&a, &b) {
                return Ok(Expr::Num(sanitize(apply_binary(
                    to_binary_kind(*op),
                    *va,
                    *vb,
                ))));
            }
            Ok(Expr::Binary(*op, Box::new(a), Box::new(b)))
        }
        Expr::Call(func, args) => {
            let mut folded_args = Vec::with_capacity(args.len());
            let mut all_const = true;
            for arg in args {
                let f = fold_expr(arg, scope)?;
                if !matches!(f, Expr::Num(_)) {
                    all_const = false;
                }
                folded_args.push(f);
            }
            if all_const {
                let mut vals = [0.0f64; 3];
                for (dst, arg) in vals.iter_mut().zip(&folded_args) {
                    if let Expr::Num(v) = *arg {
                        *dst = v;
                    }
                }
                if let Some(res) = apply_pure_builtin(func.op, &vals[..folded_args.len()]) {
                    return Ok(Expr::Num(sanitize(res)));
                }
            }
            Ok(Expr::Call(*func, folded_args))
        }
    }
}

/// Verifies the expression's required evaluation-stack depth ≤ [`MAX_DEPTH`] and
/// returns it. The stack requirement is the classic Ershov number:
/// - leaf (literal/identifier): 1 slot;
/// - unary: same as its operand (the operand's slot is reused in place);
/// - binary `a op b`: `max(need(a), 1 + need(b))` — `a`'s result occupies one
///   slot while `b` is evaluated;
/// - call with args `a0..ak`: while arg `j` is computed, the `j` earlier results
///   sit on the stack, so `max_j (j + need(aj))`.
///
/// Returns [`CompileErrorKind::NestingTooDeep`] (naming a representative token)
/// if any subtree's requirement exceeds the cap.
fn check_depth(expr: &Expr) -> Result<usize, CompileError> {
    let need = match expr {
        Expr::Num(_) | Expr::Ident(_) => 1,
        Expr::Unary(_, a) => check_depth(a)?,
        Expr::Binary(_, a, b) => {
            let na = check_depth(a)?;
            let nb = check_depth(b)?;
            na.max(1 + nb)
        }
        Expr::Call(_, args) => {
            let mut max_need = 1;
            for (j, arg) in args.iter().enumerate() {
                let n = check_depth(arg)?;
                max_need = max_need.max(j + n);
            }
            max_need
        }
    };
    if need > MAX_DEPTH {
        return Err(CompileError {
            kind: CompileErrorKind::NestingTooDeep,
            token: token_for(expr),
            position: 0,
        });
    }
    Ok(need)
}

/// A representative token string for an AST node, for depth-overflow diagnostics.
fn token_for(expr: &Expr) -> String {
    match expr {
        Expr::Num(v) => v.to_string(),
        Expr::Ident(n) => n.clone(),
        Expr::Unary(_, _) => "(unary)".to_owned(),
        Expr::Binary(_, _, _) => "(operator)".to_owned(),
        Expr::Call(f, _) => f.name.to_owned(),
    }
}

/// Interns a sanitized constant into `consts` and returns its pool index.
fn intern_const(consts: &mut Vec<f64>, value: f64) -> u16 {
    let v = sanitize(value);
    let bits = v.to_bits();
    if let Some(pos) = consts.iter().position(|c| c.to_bits() == bits) {
        return pos as u16;
    }
    let idx = consts.len() as u16;
    consts.push(v);
    idx
}

/// Lowers an AST into compact post-order bytecode, resolving identifiers to slot
/// loads or deduplicated constant pushes.
fn emit(
    expr: &Expr,
    scope: &Scope,
    ops: &mut Vec<Op>,
    consts: &mut Vec<f64>,
    max_slot: &mut Option<u16>,
) -> Result<(), CompileError> {
    match expr {
        Expr::Num(v) => {
            let idx = intern_const(consts, *v);
            ops.push(Op::Const(idx));
            Ok(())
        }
        Expr::Ident(name) => {
            if let Some(slot) = scope.slot_id(name) {
                *max_slot = Some(max_slot.map_or(slot.0, |m| m.max(slot.0)));
                ops.push(Op::Load(slot.0));
                Ok(())
            } else if let Some(value) = lookup_constant(name) {
                let idx = intern_const(consts, value);
                ops.push(Op::Const(idx));
                Ok(())
            } else {
                Err(CompileError {
                    kind: CompileErrorKind::UnknownIdentifier,
                    token: name.clone(),
                    position: 0,
                })
            }
        }
        Expr::Unary(op, a) => {
            emit(a, scope, ops, consts, max_slot)?;
            ops.push(match op {
                UnaryOp::Neg => Op::Neg,
                UnaryOp::Not => Op::Not,
            });
            Ok(())
        }
        Expr::Binary(op, a, b) => {
            // Fast path: `x ^ 2` -> `Square` (`x * x`, bitwise equal to powf(2.0)
            // for all finite inputs after sanitization).
            if matches!(op, BinaryOp::Pow) && matches!(b.as_ref(), Expr::Num(v) if *v == 2.0) {
                emit(a, scope, ops, consts, max_slot)?;
                ops.push(Op::Square);
                return Ok(());
            }
            emit(a, scope, ops, consts, max_slot)?;
            emit(b, scope, ops, consts, max_slot)?;
            ops.push(binary_to_op(*op));
            Ok(())
        }
        Expr::Call(func, args) => {
            // Fast path: `pow(x, 2)` -> `Square`.
            if func.op == BuiltinOp::Pow
                && args.len() == 2
                && matches!(args[1], Expr::Num(v) if v == 2.0)
            {
                emit(&args[0], scope, ops, consts, max_slot)?;
                ops.push(Op::Square);
                return Ok(());
            }
            for arg in args {
                emit(arg, scope, ops, consts, max_slot)?;
            }
            ops.push(builtin_to_op(func.op));
            Ok(())
        }
    }
}

fn binary_to_op(op: BinaryOp) -> Op {
    match op {
        BinaryOp::Add => Op::Add,
        BinaryOp::Sub => Op::Sub,
        BinaryOp::Mul => Op::Mul,
        BinaryOp::Div => Op::Div,
        BinaryOp::Rem => Op::Rem,
        BinaryOp::Pow => Op::Pow,
        BinaryOp::Lt => Op::Lt,
        BinaryOp::Le => Op::Le,
        BinaryOp::Gt => Op::Gt,
        BinaryOp::Ge => Op::Ge,
        BinaryOp::Eq => Op::Eq,
        BinaryOp::Ne => Op::Ne,
        BinaryOp::And => Op::And,
        BinaryOp::Or => Op::Or,
    }
}

fn builtin_to_op(b: BuiltinOp) -> Op {
    match b {
        BuiltinOp::Sin => Op::Sin,
        BuiltinOp::Cos => Op::Cos,
        BuiltinOp::Tan => Op::Tan,
        BuiltinOp::Asin => Op::Asin,
        BuiltinOp::Acos => Op::Acos,
        BuiltinOp::Atan => Op::Atan,
        BuiltinOp::Atan2 => Op::Atan2,
        BuiltinOp::Exp => Op::Exp,
        BuiltinOp::Log => Op::Log,
        BuiltinOp::Log2 => Op::Log2,
        BuiltinOp::Log10 => Op::Log10,
        BuiltinOp::Pow => Op::Pow,
        BuiltinOp::Sqrt => Op::Sqrt,
        BuiltinOp::Abs => Op::Abs,
        BuiltinOp::Sign => Op::Sign,
        BuiltinOp::Floor => Op::Floor,
        BuiltinOp::Ceil => Op::Ceil,
        BuiltinOp::Round => Op::Round,
        BuiltinOp::Fract => Op::Fract,
        BuiltinOp::Min => Op::Min,
        BuiltinOp::Max => Op::Max,
        BuiltinOp::Clamp => Op::Clamp,
        BuiltinOp::Mix => Op::Mix,
        BuiltinOp::Smoothstep => Op::Smoothstep,
        BuiltinOp::Step => Op::Step,
        BuiltinOp::If => Op::If,
        BuiltinOp::Band => Op::Band,
        BuiltinOp::Level => Op::Level,
        BuiltinOp::Peak => Op::Peak,
        BuiltinOp::Wave => Op::Wave,
        BuiltinOp::Rand => Op::Rand,
    }
}

/// One input formula to [`compile_multi`].
#[derive(Clone, Copy, Debug)]
pub enum MultiInput<'a> {
    /// A numeric constant.
    Const(f64),
    /// A formula expression string.
    Expr(&'a str),
}

/// Canonical, hashable representation of a folded AST node for LICM + CSE.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CanonExpr {
    Num(u64),
    Slot(u16),
    Unary(UnaryKind, usize),
    Binary(BinaryKind, usize, usize),
    Call(BuiltinOp, Vec<usize>),
}

const DEP_ITEM: u8 = 1;
const DEP_ROW: u8 = 2;
const DEP_COL: u8 = 4;

struct DagBuilder<'a> {
    scope: &'a Scope,
    item_slots: &'a [SlotId],
    row_slot: Option<SlotId>,
    col_slot: Option<SlotId>,
    nodes: Vec<CanonExpr>,
    deps: Vec<u8>,
    intern: HashMap<CanonExpr, usize>,
}

impl<'a> DagBuilder<'a> {
    fn intern_node(&mut self, node: CanonExpr, dep: u8) -> usize {
        if let Some(&id) = self.intern.get(&node) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.deps.push(dep);
        self.intern.insert(node, id);
        id
    }

    fn lower(&mut self, expr: &Expr) -> Result<usize, CompileError> {
        match expr {
            Expr::Num(v) => Ok(self.intern_node(CanonExpr::Num(sanitize(*v).to_bits()), 0)),
            Expr::Ident(name) => {
                if let Some(slot) = self.scope.slot_id(name) {
                    let mut dep = 0u8;
                    if self.item_slots.contains(&slot) {
                        dep |= DEP_ITEM;
                    }
                    if self.row_slot == Some(slot) {
                        dep |= DEP_ROW;
                    }
                    if self.col_slot == Some(slot) {
                        dep |= DEP_COL;
                    }
                    Ok(self.intern_node(CanonExpr::Slot(slot.0), dep))
                } else if let Some(val) = lookup_constant(name) {
                    Ok(self.intern_node(CanonExpr::Num(sanitize(val).to_bits()), 0))
                } else {
                    Err(CompileError {
                        kind: CompileErrorKind::UnknownIdentifier,
                        token: name.clone(),
                        position: 0,
                    })
                }
            }
            Expr::Unary(op, a) => {
                let a_id = self.lower(a)?;
                let kind = match op {
                    UnaryOp::Neg => UnaryKind::Neg,
                    UnaryOp::Not => UnaryKind::Not,
                };
                let dep = self.deps[a_id];
                Ok(self.intern_node(CanonExpr::Unary(kind, a_id), dep))
            }
            Expr::Binary(op, a, b) => {
                let a_id = self.lower(a)?;
                let b_id = self.lower(b)?;
                let dep = self.deps[a_id] | self.deps[b_id];
                Ok(self.intern_node(CanonExpr::Binary(to_binary_kind(*op), a_id, b_id), dep))
            }
            Expr::Call(func, args) => {
                let mut arg_ids = Vec::with_capacity(args.len());
                let mut dep = 0u8;
                for arg in args {
                    let id = self.lower(arg)?;
                    dep |= self.deps[id];
                    arg_ids.push(id);
                }
                Ok(self.intern_node(CanonExpr::Call(func.op, arg_ids), dep))
            }
        }
    }

    fn is_non_trivial(&self, id: usize) -> bool {
        !matches!(self.nodes[id], CanonExpr::Num(_) | CanonExpr::Slot(_))
    }
}

/// Compiles all per-item formulas of one layer into a single [`MultiProgram`] with:
/// - **Constant folding** + `pow(x, 2) -> Square`
/// - **LICM (Loop-Invariant Code Motion)**: maximal frame-invariant subtrees hoist
///   into `hoisted_ops` (1×/frame); for `Field`, `y`-only hoist into `row_ops`
///   (`res`×/frame) and `x`-only hoist into `col_ops` (`res`×/frame)
/// - **Cross-formula CSE**: any remaining varying subtree used ≥ 2 times across the
///   layer's formulas computes once (`StoreTemp`) and reloads (`LoadTemp`).
///
/// Output semantics are **bitwise identical** to evaluating each formula individually.
pub fn compile_multi(
    inputs: &[MultiInput<'_>],
    scope: &Scope,
    item_slots: &[SlotId],
    row_slot: Option<SlotId>,
    col_slot: Option<SlotId>,
) -> Result<MultiProgram, CompileError> {
    debug_assert!(inputs.len() <= MAX_MULTI_OUTS);

    let mut dag = DagBuilder {
        scope,
        item_slots,
        row_slot,
        col_slot,
        nodes: Vec::new(),
        deps: Vec::new(),
        intern: HashMap::new(),
    };

    let mut roots = Vec::with_capacity(inputs.len());
    for input in inputs {
        let root_id = match *input {
            MultiInput::Const(v) => dag.intern_node(CanonExpr::Num(sanitize(v).to_bits()), 0),
            MultiInput::Expr(src) => {
                let tokens = lex(src).map_err(|e| CompileError {
                    kind: CompileErrorKind::Lex,
                    token: e.token,
                    position: e.position,
                })?;
                let ast = parse(&tokens).map_err(|e| CompileError {
                    kind: CompileErrorKind::Parse,
                    token: e.token,
                    position: e.position,
                })?;
                check_depth(&ast)?;
                let folded = fold_expr(&ast, scope)?;
                dag.lower(&folded)?
            }
        };
        roots.push(root_id);
    }

    // Step 1: Collect maximal hoist candidates (frame-invariant, row-only, col-only).
    // Walk top-down from every root:
    // - Frame-invariant whole roots go straight to `hoisted_outs` (written via `StoreOut`
    //   inside `hoisted_ops`, or `const_outs` if constant).
    // - Inside varying subtrees, maximal `dep == 0` non-trivial subtrees hoist to `frame_hoist`.
    // - For Field (`row_slot`/`col_slot` set), maximal `dep == DEP_ROW` non-trivial subtrees
    //   hoist to `row_hoist`, and maximal `dep == DEP_COL` non-trivial subtrees hoist to `col_hoist`.
    #[derive(Default)]
    struct HoistBuckets {
        frame: Vec<usize>,
        frame_set: HashSet<usize>,
        row: Vec<usize>,
        row_set: HashSet<usize>,
        col: Vec<usize>,
        col_set: HashSet<usize>,
    }

    let mut hoists = HoistBuckets::default();

    fn collect_hoists(
        id: usize,
        dag: &DagBuilder<'_>,
        in_row_or_col: bool,
        hoists: &mut HoistBuckets,
    ) {
        let dep = dag.deps[id];
        if dep == 0 {
            if dag.is_non_trivial(id) && hoists.frame_set.insert(id) {
                hoists.frame.push(id);
            }
            return;
        }
        if !in_row_or_col {
            if dep == DEP_ROW && dag.row_slot.is_some() {
                if dag.is_non_trivial(id) && hoists.row_set.insert(id) {
                    hoists.row.push(id);
                    // Also hoist any frame-invariant subtrees inside this row subtree!
                    for child in children_of(&dag.nodes[id]) {
                        collect_hoists(child, dag, true, hoists);
                    }
                }
                return;
            }
            if dep == DEP_COL && dag.col_slot.is_some() {
                if dag.is_non_trivial(id) && hoists.col_set.insert(id) {
                    hoists.col.push(id);
                    for child in children_of(&dag.nodes[id]) {
                        collect_hoists(child, dag, true, hoists);
                    }
                }
                return;
            }
        }
        for child in children_of(&dag.nodes[id]) {
            collect_hoists(child, dag, in_row_or_col, hoists);
        }
    }

    let mut const_outs = [None; MAX_MULTI_OUTS];
    let mut hoisted_out_roots: Vec<(usize, usize)> = Vec::new();
    let mut row_out_roots: Vec<(usize, usize)> = Vec::new();
    let mut item_out_roots: Vec<(usize, usize)> = Vec::new();

    for (out_idx, &root_id) in roots.iter().enumerate() {
        if let CanonExpr::Num(bits) = dag.nodes[root_id] {
            const_outs[out_idx] = Some(f64::from_bits(bits));
            continue;
        }
        let dep = dag.deps[root_id];
        if dep == 0 {
            hoisted_out_roots.push((out_idx, root_id));
        } else if dep == DEP_ROW && dag.row_slot.is_some() {
            row_out_roots.push((out_idx, root_id));
            // Hoist frame-invariant subtrees inside this row-only root.
            for child in children_of(&dag.nodes[root_id]) {
                collect_hoists(child, &dag, true, &mut hoists);
            }
        } else {
            item_out_roots.push((out_idx, root_id));
            collect_hoists(root_id, &dag, false, &mut hoists);
        }
    }

    // Allocate temp slots (bounded by MAX_TEMPS):
    //   [0 .. row_base)        : frame_hoist
    //   [row_base .. col_base) : row_hoist
    //   [col_base .. cse_base) : col_hoist
    //   [cse_base .. MAX_TEMPS): varying CSE
    let mut next_temp: usize = 0;
    let mut precomputed_slot: HashMap<usize, u16> = HashMap::new();

    let mut active_frame_hoist = Vec::new();
    for &id in &hoists.frame {
        if next_temp >= MAX_TEMPS {
            break;
        }
        precomputed_slot.insert(id, next_temp as u16);
        active_frame_hoist.push((id, next_temp as u16));
        next_temp += 1;
    }

    let mut active_row_hoist = Vec::new();
    for &id in &hoists.row {
        if next_temp >= MAX_TEMPS {
            break;
        }
        precomputed_slot.insert(id, next_temp as u16);
        active_row_hoist.push((id, next_temp as u16));
        next_temp += 1;
    }

    let col_base = next_temp;
    let mut active_col_hoist = Vec::new();
    for &id in &hoists.col {
        if next_temp >= MAX_TEMPS {
            break;
        }
        precomputed_slot.insert(id, next_temp as u16);
        active_col_hoist.push((id, next_temp as u16));
        next_temp += 1;
    }
    let num_col = active_col_hoist.len();

    // Step 2: Count occurrences of remaining varying non-trivial nodes across
    // all item outputs to find cross-formula CSE candidates.
    let mut use_counts: HashMap<usize, usize> = HashMap::new();
    let mut cse_order: Vec<usize> = Vec::new();
    fn count_uses(
        id: usize,
        dag: &DagBuilder<'_>,
        precomputed: &HashMap<usize, u16>,
        counts: &mut HashMap<usize, usize>,
        order: &mut Vec<usize>,
    ) {
        if precomputed.contains_key(&id) || !dag.is_non_trivial(id) {
            return;
        }
        let c = counts.entry(id).or_insert(0);
        *c += 1;
        if *c == 2 {
            order.push(id);
        }
        if *c == 1 {
            for child in children_of(&dag.nodes[id]) {
                count_uses(child, dag, precomputed, counts, order);
            }
        }
    }
    for &(_, root_id) in &item_out_roots {
        count_uses(
            root_id,
            &dag,
            &precomputed_slot,
            &mut use_counts,
            &mut cse_order,
        );
    }

    let mut cse_slot: HashMap<usize, u16> = HashMap::new();
    for &id in &cse_order {
        if next_temp >= MAX_TEMPS {
            break;
        }
        cse_slot.insert(id, next_temp as u16);
        next_temp += 1;
    }

    // Step 3: Emit bytecode for each phase.
    let mut consts = Vec::new();
    let mut max_slot: Option<u16> = None;
    let mut reads_slots_set: HashSet<u16> = HashSet::new();

    struct EmitCtx<'a> {
        dag: &'a DagBuilder<'a>,
        consts: &'a mut Vec<f64>,
        max_slot: &'a mut Option<u16>,
        reads_slots: &'a mut HashSet<u16>,
    }

    fn emit_dag(
        id: usize,
        ctx: &mut EmitCtx<'_>,
        ops: &mut Vec<Op>,
        precomputed: &HashMap<usize, u16>,
        cse_slots: &HashMap<usize, u16>,
        cse_emitted: &mut HashSet<usize>,
        skip_precomputed_for: Option<usize>,
    ) {
        if skip_precomputed_for != Some(id) {
            if let Some(&t) = precomputed.get(&id) {
                ops.push(Op::LoadTemp(t));
                return;
            }
        }
        if let Some(&t) = cse_slots.get(&id) {
            if cse_emitted.contains(&id) {
                ops.push(Op::LoadTemp(t));
                return;
            }
        }
        match &ctx.dag.nodes[id] {
            CanonExpr::Num(bits) => {
                let idx = intern_const(ctx.consts, f64::from_bits(*bits));
                ops.push(Op::Const(idx));
            }
            CanonExpr::Slot(slot) => {
                *ctx.max_slot = Some(ctx.max_slot.map_or(*slot, |m| m.max(*slot)));
                ctx.reads_slots.insert(*slot);
                ops.push(Op::Load(*slot));
            }
            CanonExpr::Unary(kind, a) => {
                emit_dag(*a, ctx, ops, precomputed, cse_slots, cse_emitted, None);
                ops.push(match kind {
                    UnaryKind::Neg => Op::Neg,
                    UnaryKind::Not => Op::Not,
                });
            }
            CanonExpr::Binary(kind, a, b) => {
                if *kind == BinaryKind::Pow
                    && matches!(ctx.dag.nodes[*b], CanonExpr::Num(bits) if f64::from_bits(bits) == 2.0)
                {
                    emit_dag(*a, ctx, ops, precomputed, cse_slots, cse_emitted, None);
                    ops.push(Op::Square);
                } else {
                    emit_dag(*a, ctx, ops, precomputed, cse_slots, cse_emitted, None);
                    emit_dag(*b, ctx, ops, precomputed, cse_slots, cse_emitted, None);
                    ops.push(binary_kind_to_op(*kind));
                }
            }
            CanonExpr::Call(builtin, args) => {
                if *builtin == BuiltinOp::Pow
                    && args.len() == 2
                    && matches!(ctx.dag.nodes[args[1]], CanonExpr::Num(bits) if f64::from_bits(bits) == 2.0)
                {
                    emit_dag(args[0], ctx, ops, precomputed, cse_slots, cse_emitted, None);
                    ops.push(Op::Square);
                } else {
                    for &arg in args {
                        emit_dag(arg, ctx, ops, precomputed, cse_slots, cse_emitted, None);
                    }
                    ops.push(builtin_to_op(*builtin));
                }
            }
        }
        if let Some(&t) = cse_slots.get(&id) {
            cse_emitted.insert(id);
            ops.push(Op::StoreTemp(t));
        }
    }

    let empty_map: HashMap<usize, u16> = HashMap::new();
    let mut empty_set: HashSet<usize> = HashSet::new();

    let mut ctx = EmitCtx {
        dag: &dag,
        consts: &mut consts,
        max_slot: &mut max_slot,
        reads_slots: &mut reads_slots_set,
    };

    // 3a. hoisted_ops (frame-invariant subtrees + frame-invariant whole outputs).
    let mut hoisted_ops = Vec::new();
    let mut frame_only_map: HashMap<usize, u16> = HashMap::new();
    for &(id, slot) in &active_frame_hoist {
        emit_dag(
            id,
            &mut ctx,
            &mut hoisted_ops,
            &frame_only_map,
            &empty_map,
            &mut empty_set,
            Some(id),
        );
        hoisted_ops.push(Op::PopTemp(slot));
        frame_only_map.insert(id, slot);
    }
    for &(out_idx, root_id) in &hoisted_out_roots {
        emit_dag(
            root_id,
            &mut ctx,
            &mut hoisted_ops,
            &frame_only_map,
            &empty_map,
            &mut empty_set,
            None,
        );
        hoisted_ops.push(Op::StoreOut(out_idx as u16));
    }

    // 3b. row_ops (y-only subtrees + y-only whole outputs, using frame_only_map).
    let mut row_ops = Vec::new();
    let mut frame_and_row_map = frame_only_map.clone();
    for &(id, slot) in &active_row_hoist {
        emit_dag(
            id,
            &mut ctx,
            &mut row_ops,
            &frame_and_row_map,
            &empty_map,
            &mut empty_set,
            Some(id),
        );
        row_ops.push(Op::PopTemp(slot));
        frame_and_row_map.insert(id, slot);
    }
    for &(out_idx, root_id) in &row_out_roots {
        emit_dag(
            root_id,
            &mut ctx,
            &mut row_ops,
            &frame_and_row_map,
            &empty_map,
            &mut empty_set,
            None,
        );
        row_ops.push(Op::StoreOut(out_idx as u16));
    }

    // 3c. col_ops (x-only subtrees, using frame_only_map).
    let mut col_ops = Vec::new();
    let mut frame_and_col_map = frame_only_map;
    for &(id, slot) in &active_col_hoist {
        emit_dag(
            id,
            &mut ctx,
            &mut col_ops,
            &frame_and_col_map,
            &empty_map,
            &mut empty_set,
            Some(id),
        );
        col_ops.push(Op::PopTemp(slot));
        frame_and_col_map.insert(id, slot);
    }

    // 3d. item_ops (all remaining varying outputs with cross-formula CSE).
    let mut item_ops = Vec::new();
    let mut cse_emitted: HashSet<usize> = HashSet::new();
    for &(out_idx, root_id) in &item_out_roots {
        emit_dag(
            root_id,
            &mut ctx,
            &mut item_ops,
            &precomputed_slot,
            &cse_slot,
            &mut cse_emitted,
            None,
        );
        item_ops.push(Op::StoreOut(out_idx as u16));
    }

    let mut reads_slots: Vec<u16> = reads_slots_set.into_iter().collect();
    reads_slots.sort_unstable();

    let mut item_lane_temps: u64 = 0;
    for t in col_base..col_base + num_col {
        item_lane_temps |= 1 << t;
    }
    let mut item_outs: u16 = 0;
    for op in &item_ops {
        match *op {
            Op::StoreTemp(t) | Op::PopTemp(t) if (t as usize) < MAX_TEMPS => {
                item_lane_temps |= 1 << t;
            }
            Op::StoreOut(o) if (o as usize) < MAX_MULTI_OUTS => item_outs |= 1 << o,
            _ => {}
        }
    }

    Ok(MultiProgram {
        hoisted_ops,
        row_ops,
        col_ops,
        item_ops,
        consts,
        max_slot,
        col_base,
        num_col,
        num_outputs: inputs.len(),
        const_outs,
        reads_slots,
        item_lane_temps,
        item_outs,
    })
}

fn children_of(node: &CanonExpr) -> Vec<usize> {
    match node {
        CanonExpr::Num(_) | CanonExpr::Slot(_) => Vec::new(),
        CanonExpr::Unary(_, a) => vec![*a],
        CanonExpr::Binary(_, a, b) => vec![*a, *b],
        CanonExpr::Call(_, args) => args.clone(),
    }
}

fn binary_kind_to_op(kind: BinaryKind) -> Op {
    match kind {
        BinaryKind::Add => Op::Add,
        BinaryKind::Sub => Op::Sub,
        BinaryKind::Mul => Op::Mul,
        BinaryKind::Div => Op::Div,
        BinaryKind::Rem => Op::Rem,
        BinaryKind::Pow => Op::Pow,
        BinaryKind::Lt => Op::Lt,
        BinaryKind::Le => Op::Le,
        BinaryKind::Gt => Op::Gt,
        BinaryKind::Ge => Op::Ge,
        BinaryKind::Eq => Op::Eq,
        BinaryKind::Ne => Op::Ne,
        BinaryKind::And => Op::And,
        BinaryKind::Or => Op::Or,
    }
}

/// Maps a parser binary operator to its VM op kind.
fn to_binary_kind(op: crate::parser::BinaryOp) -> BinaryKind {
    use crate::parser::BinaryOp as B;
    match op {
        B::Add => BinaryKind::Add,
        B::Sub => BinaryKind::Sub,
        B::Mul => BinaryKind::Mul,
        B::Div => BinaryKind::Div,
        B::Rem => BinaryKind::Rem,
        B::Pow => BinaryKind::Pow,
        B::Lt => BinaryKind::Lt,
        B::Le => BinaryKind::Le,
        B::Gt => BinaryKind::Gt,
        B::Ge => BinaryKind::Ge,
        B::Eq => BinaryKind::Eq,
        B::Ne => BinaryKind::Ne,
        B::And => BinaryKind::And,
        B::Or => BinaryKind::Or,
    }
}
