---
name: viewmusic-perf-hillclimb
description: Hillclimb ViewMusic CPU, GPU and memory with the frozen headless and live perf harnesses, interleaved A/B measurement, a pixel-exact gate and parallel worker subagents in owned worktrees. Invoke with /skill:viewmusic-perf-hillclimb.
disable-model-invocation: true
---

# ViewMusic perf hillclimb

You are the orchestrator. You own the ruler, the measurements, the reviews and the commits. Workers write code in worktrees you create. Only you measure, one run at a time, because the GPU is shared state: parallel benchmarks corrupt each other.

Read `references/lessons.md` before you form any hypothesis. It holds the measured cost model and the lists of done and rejected ideas.

## Rulers (already in the repo; do not edit them during a run)
- **Headless** `crates/viz-app/examples/perf_harness.rs`. It renders all 15 built-in artifacts offscreen at 3024x1890 `Bgra8UnormSrgb` through the real `Analyzer`, `ArtifactRuntime` and `Renderer`, with a seeded synthetic 120 BPM signal. It reports CPU µs/frame (rusage), GPU µs/frame (ioreg `AGXDeviceUserClient` `accumulatedGPUTime` for our pid) and phys_footprint, plus error and work counts. `--dump DIR` / `--compare A B` is the pixel gate (it fails above 0.5% of pixels with diff > 2; demand max_abs_diff 0).
- **Headless A/B** `scripts/perf-ab.sh <bin-A> <bin-B> [pairs=3] [-- harness args]`. It runs A B A B A B and prints medians, spreads and per-artifact deltas.
- **Live** `crates/viz-app/examples/live_harness.rs` (`scripts/perf-live.sh`). It runs the real `viz_app::window::App` (present, 60 Hz pacing, occlusion pause) with synthetic audio, so there is no permission prompt. It needs an unlocked screen and a visible window.
- **Live A/B** `scripts/live-ab.sh` in this skill: `BASE=<live-A> live-ab.sh name=<live-B>`. It also records WindowServer GPU %, which is the compositor cost of our window.
- **Parking binaries** `scripts/build-bins.sh <checkout> <name>` builds both harnesses and copies them to `tmp/hillclimb/bin/<name>` and `live-<name>`.

macOS only. The `.app` is ad-hoc signed, so every rebuild loses the System Audio grant. Never use the real app as a ruler, and never let a child run it.

## Setup (once per run)
1. Create the run directory `tmp/hillclimb/` (untracked) with `plan.md` and `decision.tsv` (header: `id hypothesis change before after delta tests verdict note`, tab-separated). In `plan.md`, write the metric, the direction, the accept rule and a stop predicate that pairs a target with a minimum number of attempts.
2. Park the baseline: `build-bins.sh . base`. Make reference dumps: `tmp/hillclimb/bin/base --dump tmp/hillclimb/dumps-base`.
3. Prove the noise floor with a null A/B: `scripts/perf-ab.sh bin/base bin/base 3`. Expect deltas under 1% and per-run CPU spread up to ~13% on a busy machine.
4. Gate the baseline: `nix develop -c cargo test --workspace --release` and `cargo clippy --workspace --all-targets`. Record the pass count.
5. Create an integration branch (for example `perf/hillclimb`) and a parent-owned worktree for it under `../viewmusic-wt/`.

## Loop
1. **Ground the hypotheses.** Use a profile of the harness (`sample <pid>`, then verify with direct timers or instruction counts), `references/lessons.md`, and read-only scout children for new areas. Rank by the model: CPU follows encoder count and per-item VM work; GPU follows frames × attachments × full-screen pixels; memory follows full-surface textures, per-frame copies and driver blocks.
2. **Fan out.** Give each hypothesis its own worktree with disjoint files: `git worktree add -b perf/wN-<name> ../viewmusic-wt/wN-<name> <integration branch>`. Create the worktree and wait for it to finish BEFORE you launch the child, or the launch fails on a missing cwd. Copy `references/brief-common.md` to `tmp/hillclimb/`, fill it in, and write one section per hypothesis (worktree, allowed files, data shape, mechanism, expected win, gate). Launch workers with `async: true`, `context: "fresh"` and `options.timeoutMs` ≥ 5400000. The default 30 min timeout killed a harness build mid-run.
3. **Measure each candidate alone** after all workers finish. Run `build-bins.sh <worktree> <name>`, then `scripts/perf-ab.sh bin/<current> bin/<name> 3`. Accept only if every pair agrees and the delta clears the spread: about 5% CPU, 3% GPU, 2% footprint. Use a targeted `-- --only <ids>` with 5 pairs when the change only touches some artifacts. Discard any run taken while children were building or timing; a +30% change on an untouched artifact is the tell. Use the live A/B for anything in `window.rs`, present, overlay or compositor, and for memory that only shows across artifact switches on one renderer.
4. **Review the diff yourself.** Then commit in the worktree, staging only the named files, with the A/B numbers in the message. Cherry-pick onto the integration branch and rerun tests and clippy there. Log a decision row whether you kept or reverted.
5. **Repeat.** On a plateau, change category (CPU → GPU → memory → live/compositor) or combine near-misses. Do not relax the stop predicate.

## Rules that came from mistakes
- A visible change is the user's decision. Measure it, keep it on its own branch and explain exactly what changes on screen. The opaque-window win (WindowServer GPU 51% → 21%) waited for that answer.
- Prefer exact-by-construction changes over exactness tuned to one GPU. The app also ships on Windows/Linux.
- Gate new device features to the backend where they were measured (direct uploads are Metal-only).
- Before the final landing, run a fresh `reviewer` over the materialized stack diff (`git diff <baseline> <tip> -- crates/ ':!crates/viz-app/examples'`) and fix its findings with a fresh worker.
- When the screen locks, the live harness cannot render. Continue headless work and queue live runs.
- Do not use `--frames 100000` for long runs (it builds a ~320 MB signal). Use `--frames 600 --repeat 5000`.

## Finish
1. Run a final headless A/B and a final live A/B of the tip against the original baseline. Report both, with spreads and the machine load.
2. Land the work as the user asks (fast-forward or PR). Run `make install` if they want to test the app.
3. Update `references/lessons.md`: move kept items to Done, add rejected items with their reason, refresh the open ideas.
4. Remove the worktrees (`git worktree remove --force`, then `git worktree prune`) and the merged `perf/*` branches. Save uncommitted near-miss diffs to `tmp/hillclimb/patches/` first.

## Report
Open with the user's goal. Give the baseline-to-final table for each metric (headless and live), the iterations kept against reverted, one line per kept commit with its measured win, the decision log path, what is unverified, and the best next idea.
