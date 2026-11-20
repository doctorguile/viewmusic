# What the first hillclimb learned (Nov 2026, Apple M5 Pro, macOS 26)

Read this before you write a hypothesis. Each item is measured unless marked otherwise.
Raw log of that run: `tmp/hillclimb/decision.tsv` (untracked; may be gone).

## Cost model

### CPU
- After the first run the CPU floor is a fixed cost per Metal encoder, not draw work. Measured in process instructions (`proc_pid_rusage` RUSAGE_INFO_V4 `ri_instructions`): about 230k per render pass, 130k per copy pass and 85k per command buffer.
- Instruction counts stay within ±1% while CPU time on a loaded machine swings 3×. Use instructions in throwaway probes. The frozen harness reports CPU time, so judge it only with interleaved A/B.
- `sample` self-time shares can lie. It reported DSP at about 17% of the main thread; a direct `Instant` timer showed about 15× less (one hop costs 4-9 µs). Check any profile share with a direct timer before you act on it.
- The audio analyzer costs about 137k instructions per frame and is constant across artifacts.

### GPU
- At the app's duty cycle (one 60 Hz frame, then idle) the GPU runs at low clock. ioreg `accumulatedGPUTime` per frame is about 140-280 µs of wake-up per frame, plus about 230 µs per full-screen attachment in any pass that draws (per-tile overhead, at 3024x1890). Format, LoadOp/StoreOp and usage flags do not change it. An empty command buffer costs 0.
- So GPU time tracks how many full-screen pixels and attachments a frame touches, and how many frames are rendered. It does NOT track shader ALU. A -22% field-shader ALU win at full clock was invisible in the harness.
- The trails (feedback) frame floor is about 600 µs: one wake-up plus two attachments (accumulator plus surface). The built-ins are at 610-750.
- A blended fullscreen draw on the `Bgra8UnormSrgb` target costs about 150 µs extra (sRGB decode/encode in the blend). That is why one accumulator with blend-constant decay lost.
- Per-process GPU % depends on clock state. When system GPU load drops (for example WindowServer stops compositing us), our own GPU % can rise because the clocks drop. Judge GPU wins by ours plus WindowServer.

### Memory (phys_footprint)
- Apple's driver keeps a ~56 MB block (14 × 4 MiB) resident only while blit/copy/compute work keeps arriving, and frees it ~1-2 s after the last copy. Any per-frame copy (`queue.write_buffer`, StagingBelt copies, init copies) brings it back. Create buffers `MAP_WRITE` and write in place.
- A ~84 MB block (21 × 4 MiB) appears with the first render pass and stays while rendering. It cannot be removed.
- Each 3024x1890 Bgra8 texture costs about 23.8 MB. Textures and buffers cost nothing until first GPU use, and then Metal commits the whole object.
- A second set of per-pass driver buffers (~11 MB) appears when passes to different render targets are in flight.
- The live app holds 2 CAMetalLayer drawables (wgpu sets `maximumDrawableCount` = latency + 1 = 2, the minimum).
- egui overlay, while visible: +3-4% CPU, +17 MB, and it uploads with copies (so the 56 MB block returns). It is skipped once auto-hidden.

### Compositor
- A transparent-capable NSWindow plus a non-opaque CAMetalLayer made WindowServer blend the whole window every frame: ~48-51% WindowServer GPU against ~5% idle, about 20× our own GPU time. An opaque window when Transparent mode is off cut it to ~20%.

## Done (in the stack on doctorguile; do not re-propose)
| Commit | Change | Measured win |
|---|---|---|
| 950d3b8 | VM evaluates per-item ops in 64-lane batches, bit-identical | CPU -19.9% |
| 6aacd91 | Trails pass writes accumulator + surface in one MRT pass; blit deleted | CPU -21.6%, GPU -8.6% |
| 1473ce4 | Mapped belt chunks bound directly (MAPPABLE_PRIMARY_BUFFERS, Metal only): zero copies per frame | CPU -18%, GPU -4%, -62 MB |
| 427a6fc | Leading opaque alpha field drawn unblended; decay draw under it skipped | GPU -10..-15% on 4 artifacts |
| 4740e21 | Opaque window + layer unless Transparent mode | WindowServer GPU 51.5% → 21% |
| f98363d | Feedback textures lazy; freed after 300 inactive frames | -45 MB on no-trail artifacts (live) |

Total vs d618346: headless CPU -41..-47%, GPU -16%, footprint -24.7%. Live: CPU -31%, system GPU -53%, footprint -27%.

## Rejected (do not retry without new evidence)
- DSP micro-optimizations: real (-19% per hop) but about 1-2 µs per frame, below resolution.
- Cheaper field fragment ALU alone: invisible at low clock (it now rides along inside 427a6fc).
- One accumulator with blend-constant decay: -42 MB but GPU +15-19%. Patch: `tmp/hillclimb/patches/one-accumulator-*.patch`.
- Leading-field fusion made exact with a measured sRGB rounding LUT: hardware-specific, not exact on Windows/Linux. Replaced by the opaque-only rule.
- Surface tweaks: drawables are already 2, `framebufferOnly` is already true, a later acquire gains nothing, Immediate present risks tearing.
- Sampled-texture field cells: +4-8 MB for about 3 µs.

## Open ideas (ranked)
1. Skip frames that cannot change: during silence, once the scene data and trails have converged, render nothing instead of 20 Hz. The per-frame wake-up is the GPU floor, so fewer frames is the only way below it. Measuring needs a silence workload in the live harness and the VizApp path (the live harness has its own callback, not VizApp).
2. Trails accumulator at 0.5-0.75× resolution: about 4× less fill and -34 MB, but softer thin lines. Needs the user's approval.
3. Skip a field that is provably a no-op (all alpha 0 under alpha, all colour 0 under add).
4. Present costs a second internal command buffer every frame in wgpu-hal Metal (~85k instructions). It is only avoidable by going below wgpu.
