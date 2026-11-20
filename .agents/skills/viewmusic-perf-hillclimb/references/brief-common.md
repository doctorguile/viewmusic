# Common worker brief (copy to tmp/hillclimb/brief-common.md and fill the <...> fields)

## User goal
<one sentence: e.g. make ViewMusic use less CPU, GPU and memory than <baseline commit>, with no change to what is shown>. We hillclimb: one hypothesis per worker; the parent measures and decides keep/revert.

## Your worktree and authority
- Work ONLY in the worktree named in your hypothesis section (own branch off `<integration branch>`). Do not touch other worktrees or the main checkout except to read.
- Edit ONLY the files your section allows. Do NOT edit `crates/viz-app/examples/perf_harness.rs`, `live_harness.rs`, `scripts/perf*.sh`, tests' expected values or artifact JSON, unless your section says so. Do not commit; the parent reviews, measures and commits.
- Cargo only via `nix develop -c cargo ...` from the worktree root.
- NEVER run the `viewmusic` app binary or open the .app (audio-permission prompt). The live harness (`live_harness`) is allowed when your section says so: it opens a window, it has no audio, and the screen must be unlocked.
- Temp files go in `<worktree>/tmp/`.

## The ruler (frozen; read-only for you)
- Headless: `target/release/examples/perf_harness` (`--only <ids> --frames N --repeat R`, `--dump DIR`, `--compare A B`). Baseline binary: `<abs path to bin/hc-N>`.
- Current numbers: <cpu / gpu / footprint means and notable per-artifact values>.
- Other workers share the GPU, so your timing runs are DIRECTIONAL ONLY: short (`--only <ids> --frames 600 --repeat 3`), interleaved with the baseline binary (same flags), at least 3 alternations. Do not run `scripts/perf.sh` or `scripts/perf-ab.sh`; the parent runs full sweeps alone. Footprint is not GPU-noisy, so footprint numbers count.
- Read `<abs path>/.agents/skills/viewmusic-perf-hillclimb/references/lessons.md` for the cost model. Do not re-propose items listed under Done or Rejected.

## Gates you must run and report
1. `nix develop -c cargo test --workspace --release` green (report the pass count; baseline <N>).
2. `nix develop -c cargo clippy --workspace --all-targets` with 0 warnings; `cargo fmt --all -- --check` clean.
3. Pixel gate: `perf_harness --dump <worktree>/tmp/dumps`, then `perf_harness --compare <abs dumps-base> <worktree>/tmp/dumps`. Target max_abs_diff 0; report the summary line.

## Design rules
- Smallest change that delivers the mechanism. No narrating comments.
- Preserve output, determinism (contract §9), no new per-frame heap allocation, no per-frame copies (they bring back the 56 MB driver block), no locks on the render path.
- Exact by construction beats exact by measurement on one GPU (the app ships on Windows/Linux too).
- If the mechanism is wrong, stop and report with evidence. A measured negative result is a good outcome.

## Final message
Mechanism and files (diff size); gates; directional numbers with flags and both binaries; risks; next idea in your area.
