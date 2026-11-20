#!/usr/bin/env bash
# Frozen ViewMusic perf ruler.
#
# Builds the headless harness once (release), then runs it with the frozen defaults
# (3024x1890 Bgra8UnormSrgb, all 15 built-ins, 120 warmup + 600 measured frames,
# 1 warmup sweep + 5 measured sweeps, GPU time from ioreg). Extra arguments are
# passed through to the harness (e.g. `--only oscilloscope`, `--dump DIR`).
#
# Output: per-artifact table, then the JSON summary as the LAST stdout line.
# The full log and per-artifact JSON are kept under tmp/perf/.
#
# Never runs the `viewmusic` app binary (no window, no audio permission prompt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

nix develop -c cargo build --release -p viz-app --example perf_harness >&2

OUT_DIR="$ROOT/tmp/perf"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/perf-$STAMP.log"
JSON="$OUT_DIR/perf-$STAMP.json"

# Run the binary directly (outside `nix develop`) so the measured process is only
# the harness. pipefail keeps the harness's exit status (non-zero on wgpu errors).
"$ROOT/target/release/examples/perf_harness" --json "$JSON" "$@" | tee "$LOG"
echo "log: $LOG  json: $JSON" >&2
