#!/usr/bin/env bash
# Interleaved A/B comparison of two prebuilt perf_harness binaries.
#
# Usage: scripts/perf-ab.sh <harness-A> <harness-B> [pairs=3] [-- extra harness args]
# Runs A B A B ... (pairs times, each run = 1 warmup sweep + --repeat 2 sweeps of
# all artifacts), then prints the median over runs of cpu_us_mean, gpu_us_mean and
# the mean-over-artifacts phys footprint, plus B-vs-A deltas. Interleaving cancels
# thermal / DVFS drift that a sequential A-then-B would bake into the delta.
# Only one benchmark may use the GPU at a time; do not run two of these at once.
set -euo pipefail

A="$1"; B="$2"; PAIRS="${3:-3}"
shift 2; [[ $# -gt 0 ]] && shift
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

OUT="$(mktemp -d "${TMPDIR:-/tmp}/perf-ab.XXXXXX")"
for i in $(seq 1 "$PAIRS"); do
  for side in A B; do
    bin="$A"; [[ $side == B ]] && bin="$B"
    "$bin" --repeat 2 --json "$OUT/$side-$i.json" ${EXTRA[@]+"${EXTRA[@]}"} > "$OUT/$side-$i.log"
    tail -1 "$OUT/$side-$i.log" | sed "s/^/$side$i /" >&2
  done
done

python3 - "$OUT" "$PAIRS" <<'PY'
import json, statistics, sys
out, pairs = sys.argv[1], int(sys.argv[2])
def load(side):
    runs = []
    for i in range(1, pairs + 1):
        d = json.load(open(f"{out}/{side}-{i}.json"))
        arts = d["artifacts"]
        runs.append({
            "cpu": d["summary"]["cpu_us_mean"],
            "gpu": d["summary"]["gpu_us_mean"],
            "foot": statistics.mean(a["phys_footprint_mb"]["median"] for a in arts),
            "errors": d["summary"]["errors"],
            "per": {a["id"]: (a["cpu_us_per_frame"]["median"], a["gpu_us_per_frame"]["median"], a["phys_footprint_mb"]["median"]) for a in arts},
        })
    return runs
A, B = load("A"), load("B")
def med(runs, k): return statistics.median(r[k] for r in runs)
def spread(runs, k):
    v = [r[k] for r in runs]; m = statistics.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0
print(f"{'metric':<8}{'A med':>10}{'B med':>10}{'delta%':>9}{'A spr%':>8}{'B spr%':>8}")
for k in ("cpu", "gpu", "foot"):
    a, b = med(A, k), med(B, k)
    print(f"{k:<8}{a:>10.1f}{b:>10.1f}{(b-a)/a*100:>+9.1f}{spread(A,k):>8.1f}{spread(B,k):>8.1f}")
print(f"errors A={sum(r['errors'] for r in A)} B={sum(r['errors'] for r in B)}")
print("per-artifact median delta% (cpu / gpu / foot):")
for aid in A[0]["per"]:
    def m(runs, j): return statistics.median(r["per"][aid][j] for r in runs)
    d = [(m(B, j) - m(A, j)) / m(A, j) * 100 for j in range(3)]
    print(f"  {aid:<17}{d[0]:>+7.1f}{d[1]:>+7.1f}{d[2]:>+7.1f}")
summary = {k: {"A": med(A, k), "B": med(B, k)} for k in ("cpu", "gpu", "foot")}
print(json.dumps(summary))
PY
echo "raw: $OUT" >&2
