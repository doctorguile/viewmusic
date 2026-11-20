//! `Vm::eval_multi_batch` must equal per-item `Vm::eval_multi_item` bitwise.
//!
//! Random multi-output layers (outputs share random subexpressions so LICM and
//! cross-formula CSE kick in) are compiled with [`compile_multi`] in both the
//! element shape (`i`/`u` per item) and the field shape (`y` per row, `x` per
//! column, `i` per cell), then evaluated over a random batch of items both ways.

use proptest::prelude::*;
use viz_core::FeatureFrame;
use viz_expr::{
    compile_multi, MultiInput, MultiProgram, Scope, SlotId, Vm, BATCH_LANES, MAX_MULTI_OUTS,
};

const SLOT_NAMES: &[&str] = &["t", "energy", "low", "settings.k", "i", "u", "n", "x", "y"];
const I: SlotId = SlotId(4);
const U: SlotId = SlotId(5);
const X: SlotId = SlotId(7);
const Y: SlotId = SlotId(8);

const FUNCS: &[(&str, usize)] = &[
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("asin", 1),
    ("acos", 1),
    ("atan", 1),
    ("atan2", 2),
    ("exp", 1),
    ("log", 1),
    ("log2", 1),
    ("log10", 1),
    ("pow", 2),
    ("sqrt", 1),
    ("abs", 1),
    ("sign", 1),
    ("floor", 1),
    ("ceil", 1),
    ("round", 1),
    ("fract", 1),
    ("min", 2),
    ("max", 2),
    ("clamp", 3),
    ("mix", 3),
    ("smoothstep", 3),
    ("step", 2),
    ("if", 3),
    ("band", 1),
    ("wave", 1),
    ("rand", 1),
];

const BINOPS: &[&str] = &[
    "+", "-", "*", "/", "%", "^", "<", "<=", ">", ">=", "==", "!=", "&&", "||",
];

/// Fully parenthesized source; `#0`..`#2` are placeholders for shared subexpressions.
fn arb_src() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        prop_oneof![
            (0.0f64..1000.0).prop_map(|v| format!("{v:?}")),
            (0.0f64..1000.0).prop_map(|v| format!("(-({v:?}))")),
            Just("0.0".to_owned()),
            Just("1.0".to_owned()),
            Just("2.0".to_owned()),
            Just("0.5".to_owned()),
            Just("18.0".to_owned()),
        ],
        (0..SLOT_NAMES.len()).prop_map(|k| SLOT_NAMES[k].to_owned()),
        (0..3usize).prop_map(|k| format!("#{k}")),
    ];
    leaf.prop_recursive(5, 64, 3, |inner| {
        prop_oneof![
            (prop_oneof![Just("-"), Just("!")], inner.clone())
                .prop_map(|(op, a)| format!("({op}({a}))")),
            (0..BINOPS.len(), inner.clone(), inner.clone())
                .prop_map(|(k, a, b)| format!("({a}{}{b})", BINOPS[k])),
            (0..FUNCS.len()).prop_flat_map(move |k| {
                let (name, arity) = FUNCS[k];
                proptest::collection::vec(inner.clone(), arity)
                    .prop_map(move |args| format!("{name}({})", args.join(",")))
            }),
        ]
    })
}

fn expand(src: &str, shared: &[String]) -> String {
    let mut out = src.to_owned();
    for (k, s) in shared.iter().enumerate() {
        let inner = s.replace(['#'], "");
        out = out.replace(&format!("#{k}"), &format!("({inner})"));
    }
    out
}

fn make_scope() -> Scope {
    let mut b = Scope::builder();
    for name in SLOT_NAMES {
        b.slot(name).unwrap();
    }
    b.build()
}

fn make_frame() -> FeatureFrame {
    let mut f = FeatureFrame::default();
    for (k, b) in f.bands.iter_mut().enumerate() {
        *b = (k as f32 / 47.0).sin().abs();
    }
    for (k, w) in f.waveform.iter_mut().enumerate() {
        *w = (k as f32 / 255.0 * std::f32::consts::TAU).sin();
    }
    f
}

fn arb_value() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1000.0f64..1000.0),
        (-2.0f64..2.0),
        Just(0.0),
        Just(-0.0),
        Just(1.0),
        Just(f64::INFINITY),
        Just(f64::NAN),
    ]
}

/// Evaluates `mp` for `lane_a.len()` items per item (on a fresh VM) and in
/// consecutive batches (on another fresh VM); returns `(scalar, batch)` outputs as
/// `[item][out]` bit patterns.
#[allow(clippy::type_complexity)]
fn run_both(
    mp: &MultiProgram,
    field: bool,
    shared_vals: &[f64],
    lane_a: &[f64],
    lane_b: &[f64],
    seed: u64,
) -> (Vec<[u64; MAX_MULTI_OUTS]>, Vec<[u64; MAX_MULTI_OUTS]>) {
    let scalar = run_scalar(mp, field, shared_vals, lane_a, lane_b, seed);
    let mut vm = new_vm(seed);
    let batch = run_batches(&mut vm, mp, field, shared_vals, lane_a, lane_b);
    (scalar, batch)
}

fn new_vm(seed: u64) -> Vm {
    let mut vm = Vm::new(make_scope().slot_count(), seed);
    vm.set_frame(&make_frame());
    vm
}

/// The per-item lane slots of the element shape (`i`, `u`) or field shape (`x`, `i`).
fn lane_slots(field: bool) -> (SlotId, SlotId) {
    if field {
        (X, I)
    } else {
        (I, U)
    }
}

/// Sets every slot to `shared_vals`, runs the hoisted (and, for a field, row) ops, and
/// fills the per-item column table. Returns the hoisted/row outputs and the table.
fn prepare(
    vm: &mut Vm,
    mp: &MultiProgram,
    field: bool,
    shared_vals: &[f64],
    lane_a: &[f64],
) -> ([f64; MAX_MULTI_OUTS], Vec<f64>) {
    for (k, v) in shared_vals.iter().enumerate() {
        vm.set_slot(SlotId(k as u16), *v);
    }
    let mut seed_outs = [0.0f64; MAX_MULTI_OUTS];
    vm.eval_multi_hoisted(mp, &mut seed_outs);
    if field {
        vm.eval_multi_row(mp, &mut seed_outs);
    }
    let num_col = mp.num_col();
    let mut col_src = vec![0.0f64; lane_a.len() * num_col];
    if field && num_col > 0 {
        for (k, &x) in lane_a.iter().enumerate() {
            vm.set_slot(X, x);
            vm.eval_multi_col(mp, &mut col_src[k * num_col..(k + 1) * num_col]);
        }
    }
    (seed_outs, col_src)
}

/// The scalar reference: each item through `Vm::eval_multi_item` on a fresh VM.
fn run_scalar(
    mp: &MultiProgram,
    field: bool,
    shared_vals: &[f64],
    lane_a: &[f64],
    lane_b: &[f64],
    seed: u64,
) -> Vec<[u64; MAX_MULTI_OUTS]> {
    let mut vm = new_vm(seed);
    let (seed_outs, col_src) = prepare(&mut vm, mp, field, shared_vals, lane_a);
    let (slot_a, slot_b) = lane_slots(field);
    let num_col = mp.num_col();
    let mut outs = seed_outs;
    (0..lane_a.len())
        .map(|k| {
            vm.set_slot(slot_a, lane_a[k]);
            vm.set_slot(slot_b, lane_b[k]);
            vm.load_multi_col(mp, &col_src[k * num_col..]);
            vm.eval_multi_item(mp, &mut outs);
            let mut bits = [0u64; MAX_MULTI_OUTS];
            for o in 0..mp.num_outputs() {
                bits[o] = outs[o].to_bits();
            }
            bits
        })
        .collect()
}

/// Evaluates the items on `vm` in consecutive `Vm::eval_multi_batch` calls of up to
/// `BATCH_LANES` items, the way the runtime does.
fn run_batches(
    vm: &mut Vm,
    mp: &MultiProgram,
    field: bool,
    shared_vals: &[f64],
    lane_a: &[f64],
    lane_b: &[f64],
) -> Vec<[u64; MAX_MULTI_OUTS]> {
    let (seed_outs, col_src) = prepare(vm, mp, field, shared_vals, lane_a);
    let (slot_a, slot_b) = lane_slots(field);
    let num_col = mp.num_col();
    let n = lane_a.len();
    let mut batch = Vec::with_capacity(n);
    for start in (0..n).step_by(BATCH_LANES) {
        let lanes = (n - start).min(BATCH_LANES);
        let [la, lb] = vm.batch_slot_lanes_mut();
        la[..lanes].copy_from_slice(&lane_a[start..start + lanes]);
        lb[..lanes].copy_from_slice(&lane_b[start..start + lanes]);
        vm.eval_multi_batch(
            mp,
            lanes,
            [Some(slot_a), Some(slot_b)],
            &col_src[start * num_col..(start + lanes) * num_col],
            &seed_outs,
        );
        let bo = vm.batch_outs();
        batch.extend((0..lanes).map(|k| {
            let mut bits = [0u64; MAX_MULTI_OUTS];
            for o in 0..mp.num_outputs() {
                bits[o] = bo[o][k].to_bits();
            }
            bits
        }));
    }
    batch
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn batch_matches_per_item(
        shared in proptest::collection::vec(arb_src(), 3),
        out_srcs in proptest::collection::vec(arb_src(), 1..=MAX_MULTI_OUTS),
        field in any::<bool>(),
        shared_vals in proptest::collection::vec(arb_value(), SLOT_NAMES.len()),
        lane_ab in proptest::collection::vec((arb_value(), arb_value()), 1..=BATCH_LANES),
        seed in any::<u64>(),
    ) {
        let scope = make_scope();
        let srcs: Vec<String> = out_srcs.iter().map(|s| expand(s, &shared)).collect();
        let inputs: Vec<MultiInput> = srcs.iter().map(|s| MultiInput::Expr(s)).collect();
        let mp = if field {
            compile_multi(&inputs, &scope, &[I], Some(Y), Some(X))
        } else {
            compile_multi(&inputs, &scope, &[I, U], None, None)
        };
        let Ok(mp) = mp else { return Ok(()) };
        let lane_a: Vec<f64> = lane_ab.iter().map(|p| p.0).collect();
        let lane_b: Vec<f64> = lane_ab.iter().map(|p| p.1).collect();
        let (scalar, batch) = run_both(&mp, field, &shared_vals, &lane_a, &lane_b, seed);
        prop_assert_eq!(scalar, batch, "srcs={:?} field={}", srcs, field);
    }
}

/// Integer indices exercise the `%` integer fast path and full 64-lane batches.
#[test]
fn batch_matches_per_item_on_index_formulas() {
    let scope = make_scope();
    let srcs = [
        "i % 18 + x * y",
        "sin(x * 3 + t) * cos(y * 3 - t) + (i % 7) / 7",
        "abs(sin(x * y * 3 + t)) * smoothstep(0.1, 0.9, u) + rand(i)",
        "if(i % 2 == 0, band(u), wave(fract(i * 0.37)))",
    ];
    let inputs: Vec<MultiInput> = srcs.iter().map(|s| MultiInput::Expr(s)).collect();
    let shared_vals = [1.25, 0.5, 0.3, 2.0, 0.0, 0.0, 4096.0, 0.0, 0.3];
    for field in [false, true] {
        let mp = if field {
            compile_multi(&inputs, &scope, &[I], Some(Y), Some(X)).unwrap()
        } else {
            compile_multi(&inputs, &scope, &[I, U], None, None).unwrap()
        };
        for lanes in [1, 7, BATCH_LANES] {
            let lane_a: Vec<f64> = (0..lanes)
                .map(|k| {
                    if field {
                        -1.0 + (k as f64 + 0.5) * (2.0 / 64.0)
                    } else {
                        k as f64
                    }
                })
                .collect();
            let lane_b: Vec<f64> = (0..lanes)
                .map(|k| if field { k as f64 } else { k as f64 / 63.0 })
                .collect();
            let (scalar, batch) = run_both(&mp, field, &shared_vals, &lane_a, &lane_b, 42);
            assert_eq!(scalar, batch, "field={field} lanes={lanes}");
        }
    }
}

/// More than `BATCH_LANES` items (consecutive batches, last one partial) and different
/// programs evaluated back to back on ONE VM still match the scalar path bit for bit, so
/// no lane state leaks between batches or programs.
#[test]
fn consecutive_batches_and_programs_on_one_vm_match_per_item() {
    let scope = make_scope();
    let compile = |srcs: &[&str], field: bool| {
        let inputs: Vec<MultiInput> = srcs.iter().map(|s| MultiInput::Expr(s)).collect();
        if field {
            compile_multi(&inputs, &scope, &[I], Some(Y), Some(X)).unwrap()
        } else {
            compile_multi(&inputs, &scope, &[I, U], None, None).unwrap()
        }
    };
    let elem = compile(
        &[
            "sin(i * 0.1 + t) * u",
            "u * u + rand(i)",
            "if(i % 3 == 0, band(u), wave(fract(u * 5)))",
            "abs(cos(u * 7 - t)) + i % 5 + sin(i * 0.1 + t)",
            "max(u, energy) * low",
        ],
        false,
    );
    let field = compile(
        &[
            "sin(x * 3 + t) * cos(y * 3 - t) + (i % 7) / 7",
            "abs(sin(x * y * 3 + t)) * smoothstep(0.1, 0.9, x)",
            "fract(i * 0.37) + x * settings.k",
        ],
        true,
    );
    let shared_vals = [1.25, 0.5, 0.3, 2.0, 0.0, 0.0, 4096.0, 0.0, 0.3];

    // Elements: 3 full batches plus a partial one.
    let n = 3 * BATCH_LANES + 13;
    let elem_a: Vec<f64> = (0..n).map(|k| k as f64).collect();
    let elem_b: Vec<f64> = (0..n).map(|k| k as f64 / (n - 1) as f64).collect();
    // One field row of `res` cells: a full batch plus a partial one.
    let res = 2 * BATCH_LANES - 28;
    let field_a: Vec<f64> = (0..res)
        .map(|k| -1.0 + (k as f64 + 0.5) * (2.0 / res as f64))
        .collect();
    let field_b: Vec<f64> = (0..res).map(|k| (3 * res + k) as f64).collect();

    let seed = 42;
    let mut vm = new_vm(seed);
    let cases = [
        ("elem", &elem, false, &elem_a, &elem_b),
        ("field", &field, true, &field_a, &field_b),
        ("elem again", &elem, false, &elem_a, &elem_b),
        ("field again", &field, true, &field_a, &field_b),
    ];
    for (name, mp, is_field, lane_a, lane_b) in cases {
        let scalar = run_scalar(mp, is_field, &shared_vals, lane_a, lane_b, seed);
        let batch = run_batches(&mut vm, mp, is_field, &shared_vals, lane_a, lane_b);
        assert_eq!(scalar.len(), lane_a.len());
        assert_eq!(scalar, batch, "{name}");
    }
}
