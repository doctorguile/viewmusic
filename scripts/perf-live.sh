#!/usr/bin/env bash
# Windowed ViewMusic live perf harness.
#
# Builds the live harness once (release), then runs it: the real winit/wgpu window
# shell (`viz_app::window::App`, Fifo present, 60 Hz cap) rendering each built-in
# artifact for 2 s warmup + 10 s measured from a synthetic signal (no Core Audio,
# no permission prompt). Extra arguments are passed through to the harness
# (e.g. `--only oscilloscope,color-field --secs 5`, `--size 756x472`, `--overlay`).
#
# The window opens on the primary display and must stay visible and unoccluded
# (the app pauses rendering otherwise; the fps column shows it). It closes itself
# after the last artifact.
#
# Output: per-artifact table, then the JSON summary as the LAST stdout line.
# The full log is kept under tmp/perf/.
#
# Never runs the `viewmusic` app binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

nix develop -c cargo build --release -p viz-app --example live_harness >&2

OUT_DIR="$ROOT/tmp/perf"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/perf-live-$STAMP.log"

# Run the binary directly (outside `nix develop`) so the measured process is only
# the harness. pipefail keeps the harness's exit status.
"$ROOT/target/release/examples/live_harness" "$@" | tee "$LOG"
echo "log: $LOG" >&2
