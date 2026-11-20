#!/usr/bin/env bash
# Interleaved live A/B of prebuilt live_harness binaries, plus WindowServer's GPU
# time while each run is in its measured phase (the compositor cost of our window).
#
# Usage (screen must be unlocked; keep the harness window visible and the desktop idle):
#   BASE=<live-harness-A> live-ab.sh name1=<live-harness-B> [name2=<path> ...]
# Env knobs: ROUNDS (4) ONLY (oscilloscope,color-field) SECS (5) WARMUP (2)
#            BREAKDOWN (0; 1 = footprint -v / vmmap --summary per build, plus the
#            headless binary in HEADLESS if set) OUT_ROOT ($PWD/tmp/perf-live)
#
# Output: $OUT_ROOT/<stamp>/{results.tsv,summary.txt,*.log,breakdown/}
set -euo pipefail

: "${BASE:?set BASE to the baseline live_harness binary}"
[ $# -gt 0 ] || { echo "usage: BASE=<bin> $0 name=<bin> [...]" >&2; exit 2; }
HEADLESS=${HEADLESS:-}
OUT_ROOT=${OUT_ROOT:-$PWD/tmp/perf-live}
ROUNDS=${ROUNDS:-4}
ONLY=${ONLY:-oscilloscope,color-field}
SECS=${SECS:-5}
WARMUP=${WARMUP:-2}
BREAKDOWN=${BREAKDOWN:-0}

if ioreg -n Root -d1 | grep -q '"CGSSessionScreenIsLocked"=Yes'; then
  echo "screen is locked: the harness window would be occluded and never render" >&2
  exit 2
fi

names=(base)
paths=("$BASE")
for arg in "$@"; do names+=("${arg%%=*}"); paths+=("${arg#*=}"); done
for p in "${paths[@]}"; do [ -x "$p" ] || { echo "missing binary $p" >&2; exit 2; }; done

OUT=$OUT_ROOT/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
WS_PID=$(pgrep -x WindowServer | head -1)
n_art=$(awk -F, '{print NF}' <<<"$ONLY")

# Sum of AppUsage[*].accumulatedGPUTime (ns) over every AGXDeviceUserClient created by pid $1.
gpu_ns_of_pid() {
  ioreg -l -w0 -r -c AGXDeviceUserClient | python3 -c '
import re, sys
pid = sys.argv[1]
total = 0
for block in sys.stdin.read().split("+-o AGXDeviceUserClient")[1:]:
    if f"\"IOUserClientCreator\" = \"pid {pid}," in block:
        total += sum(int(v) for v in re.findall(r"\"accumulatedGPUTime\"=(\d+)", block))
print(total)' "$1"
}
now_s() { python3 -c 'import time; print(f"{time.monotonic():.3f}")'; }
sleep_until() { python3 -c 'import sys, time; d = float(sys.argv[1]) - time.monotonic(); time.sleep(max(d, 0))' "$1"; }

# WindowServer GPU% over `secs` with no harness running (desktop reference).
ws_idle_pct() {
  local a t0 b t1
  a=$(gpu_ns_of_pid "$WS_PID"); t0=$(now_s); sleep "$1"; b=$(gpu_ns_of_pid "$WS_PID"); t1=$(now_s)
  python3 -c "print(f'{($b-$a)/(($t1-$t0)*1e9)*100:.2f}')"
}

# One run: name, binary, round. Appends a row to results.tsv.
run_one() {
  local name=$1 bin=$2 round=$3 log="$OUT/$1-r$3.log"
  local t_start ws0 t0 ws1 t1 ws_pct
  t_start=$(now_s)
  "$bin" --only "$ONLY" --secs "$SECS" --warmup-secs "$WARMUP" >"$log" 2>&1 &
  local pid=$!
  # WindowServer window: from the first artifact's measured phase (window up ~1 s
  # after launch) to just before the last artifact's measured phase ends.
  sleep_until "$(python3 -c "print($t_start + 1.0 + $WARMUP + 0.5)")"
  ws0=$(gpu_ns_of_pid "$WS_PID"); t0=$(now_s)
  sleep_until "$(python3 -c "print($t_start + 1.0 + $n_art * ($WARMUP + $SECS) - 0.5)")"
  ws1=$(gpu_ns_of_pid "$WS_PID"); t1=$(now_s)
  wait "$pid" || echo "!!! $name round $round exited non-zero (see $log)" >&2
  ws_pct=$(python3 -c "print(f'{($ws1-$ws0)/(($t1-$t0)*1e9)*100:.2f}')")
  python3 - "$log" "$name" "$round" "$ws_pct" >>"$OUT/results.tsv" <<'PY'
import json, sys
log, name, rnd, ws = sys.argv[1:]
lines = open(log).read().splitlines()
summary = json.loads(next(l for l in reversed(lines) if l.startswith("{")))
oth = []
for l in lines:
    f = l.split()
    # artifact rows: id, WxH, scale, fps, cpu%, gpu%, oth%, ...
    if len(f) >= 15 and "x" in f[1] and f[0] != "artifact":
        try: oth.append(float(f[6]))
        except ValueError: pass
row = [name, rnd, summary["fps_mean"], summary["cpu_pct_mean"], summary["gpu_pct_mean"],
       sum(oth) / len(oth) if oth else float("nan"), float(ws), summary["max_gap_ms"],
       summary["footprint_mean_mb"], summary["footprint_peak_mb"], summary["complete"], summary["errors"]]
print("\t".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row))
PY
}

echo "base=$BASE candidates: ${names[*]:1}; rounds=$ROUNDS only=$ONLY secs=$SECS warmup=$WARMUP ws_pid=$WS_PID" | tee "$OUT/summary.txt"
printf 'name\tround\tfps\tcpu_pct\tgpu_pct\tother_gpu_pct\twindowserver_gpu_pct\tmax_gap_ms\tfootprint_mb\tpeak_mb\tcomplete\terrors\n' >"$OUT/results.tsv"
echo "WindowServer GPU% idle before: $(ws_idle_pct 5)" | tee -a "$OUT/summary.txt"

count=${#names[@]}
for ((r = 1; r <= ROUNDS; r++)); do
  for ((k = 0; k < count; k++)); do
    # Alternate the order each round so drift does not favour one build.
    if ((r % 2)); then i=$k; else i=$((count - 1 - k)); fi
    echo "round $r: ${names[$i]}"
    run_one "${names[$i]}" "${paths[$i]}" "$r"
    sleep 2
  done
done

echo "WindowServer GPU% idle after: $(ws_idle_pct 5)" | tee -a "$OUT/summary.txt"

python3 - "$OUT/results.tsv" >>"$OUT/summary.txt" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
cols = ["fps", "cpu_pct", "gpu_pct", "other_gpu_pct", "windowserver_gpu_pct", "max_gap_ms", "footprint_mb", "peak_mb"]
names = list(dict.fromkeys(r["name"] for r in rows))
med = {n: {c: statistics.median(float(r[c]) for r in rows if r["name"] == n) for c in cols} for n in names}
print("\nmedians over rounds (delta vs base in parentheses)")
print(f"{'name':<16}" + "".join(f"{c:>22}" for c in cols))
for n in names:
    cells = []
    for c in cols:
        v, b = med[n][c], med["base"][c]
        cells.append(f"{v:8.2f}" + (f" ({v - b:+7.2f})" if n != "base" else " " * 10))
    print(f"{n:<16}" + "".join(f"{s:>22}" for s in cells))
bad = [r for r in rows if r["complete"] != "True" or r["errors"] != "0"]
print(f"\nincomplete or erroring runs: {len(bad)}")
PY

if [ "$BREAKDOWN" = 1 ]; then
  mkdir -p "$OUT/breakdown"
  for ((i = 0; i < count; i++)); do
    "${paths[$i]}" --only oscilloscope --secs 12 --warmup-secs 2 >"$OUT/breakdown/${names[$i]}.log" 2>&1 &
    pid=$!
    sleep 9
    footprint -v "$pid" >"$OUT/breakdown/${names[$i]}.footprint.txt" 2>&1 || true
    vmmap --summary "$pid" >"$OUT/breakdown/${names[$i]}.vmmap.txt" 2>&1 || true
    wait "$pid" || true
  done
  if [ -n "$HEADLESS" ]; then
    "$HEADLESS" --only oscilloscope --frames 600 --repeat 5000 >"$OUT/breakdown/headless.log" 2>&1 &
    pid=$!
    sleep 6
    footprint -v "$pid" >"$OUT/breakdown/headless.footprint.txt" 2>&1 || true
    vmmap --summary "$pid" >"$OUT/breakdown/headless.vmmap.txt" 2>&1 || true
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  {
    echo
    echo "footprint categories (oscilloscope, live vs headless); see breakdown/*.footprint.txt"
    for f in "$OUT"/breakdown/*.footprint.txt; do
      echo "== $(basename "$f" .footprint.txt)"
      grep -E 'phys_footprint|Owned physical|IOSurface|IOAccelerator|MALLOC_|CoreAnimation' "$f" \
        | grep -vE '^ +([0-9a-f]+|unmapped) +- |\.\.\./' | head -20
    done
  } >>"$OUT/summary.txt"
fi

cat "$OUT/summary.txt"
echo "results: $OUT"
