#!/usr/bin/env bash
# Build the headless and live perf harnesses from a checkout and park the binaries.
#
# Usage: build-bins.sh <checkout-dir> <name> [bin-dir]
#   Builds `perf_harness` and `live_harness` (release) in <checkout-dir> and copies
#   them to <bin-dir>/<name> and <bin-dir>/live-<name>. bin-dir defaults to
#   $PWD/tmp/hillclimb/bin. Parked binaries survive worktree removal, so A/B runs
#   can compare any two points of the stack.
set -euo pipefail

SRC="$(cd "$1" && pwd)"; NAME="$2"; BIN="${3:-$PWD/tmp/hillclimb/bin}"
mkdir -p "$BIN"
(cd "$SRC" && nix develop -c cargo build --release -p viz-app \
  --example perf_harness --example live_harness >&2)
cp "$SRC/target/release/examples/perf_harness" "$BIN/$NAME"
cp "$SRC/target/release/examples/live_harness" "$BIN/live-$NAME"
echo "$BIN/$NAME"
echo "$BIN/live-$NAME"
