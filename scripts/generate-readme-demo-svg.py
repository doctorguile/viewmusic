#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Generate a lightweight, GitHub-safe animated SVG showcase of Retro Spectrum Bars.

Requirement:
    Provide an auto-playing, small-size visual demo of ViewMusic's built-in
    `retro-spectrum-bars` visualizer (Sony MHC-RX90 VFD model) that renders
    directly inside GitHub's default README.md renderer without JavaScript.

Solution:
    1. Reproduces the exact geometry, housing, glass vignette, horizontal grid,
       unlit ghost matrix, 13x14 rounded pill grid, color zones, and phosphor
       bloom proportions from `assets/builtin-artifacts/retro-spectrum-bars.artifact.json`,
       converting its linear RGBA colors through the sRGB transfer function so
       they match the real wgpu `Bgra8UnormSrgb` output.
    2. Drives all 13 frequency columns with a deterministic, seamless 4.0-second
       120 BPM musical groove processed through the exact attack/release and
       gravity peak-hold ballistics from `crates/viz-audio/src/dsp.rs`
       (93.75 hops/s, 36-hop peak hold, 0.0022 gravity).
    3. Emits pure SMIL `<animate attributeName="opacity" calcMode="discrete">`
       child tags (no `<script>`, no `xlink:href`, no `<feGaussianBlur>` filters)
       with radial-gradient phosphor halos and run-length keyframe compression
       so the file stays ~35-50 KB and snaps crisply like real 1990s VFD glass.

Usage:
    ./scripts/generate-readme-demo-svg.py
    ./scripts/generate-readme-demo-svg.py --output docs/media/retro-spectrum-demo.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Viewport (16:9, matching retro_render.rs 1280x720)
VIEW_W = 960.0
VIEW_H = 540.0

# Sony MHC-RX90 constants from assets/builtin-artifacts/retro-spectrum-bars.artifact.json
COLS = 13
ROWS = 14
SPAN_X = 1.66
SPAN_Y = 1.46
CELL_W = SPAN_X / COLS
CELL_H = SPAN_Y / ROWS
SEG_W = CELL_W * 0.78
SEG_H = CELL_H * 0.60
GLW_W = SEG_W * 2.20
GLW_H = SEG_H * 2.65
Z1 = 0.64
Z2 = 0.785
GAIN = 1.05

# DSP ballistics from crates/viz-audio/src/dsp.rs
HOP_RATE = 48000.0 / 512.0  # 93.75 hops/s
BAND_DB_ATTACK = 0.45
BAND_DB_RELEASE = 0.85
PEAK_HOLD_HOPS = 34
PEAK_GRAVITY = 0.0024
PEAK_MAX_VEL = 0.06

# Loop parameters (2 bars at 120 BPM = 4.0 s seamless loop)
LOOP_SECS = 4.0
FPS = 25
NUM_FRAMES = int(round(LOOP_SECS * FPS))  # 100 frames (exact 2-digit keyTimes)


def linear_to_srgb_channel(c: float) -> int:
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        s = 12.92 * c
    else:
        s = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, s)) * 255.0))


def linear_rgb_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        linear_to_srgb_channel(r),
        linear_to_srgb_channel(g),
        linear_to_srgb_channel(b),
    )


def ndc_to_px(x: float, y: float) -> tuple[float, float]:
    px = (x + 1.0) * 0.5 * VIEW_W
    py = (1.0 - y) * 0.5 * VIEW_H
    return px, py


def ndc_size_to_px(w: float, h: float) -> tuple[float, float]:
    return w * 0.5 * VIEW_W, h * 0.5 * VIEW_H


def fmt(v: float, digits: int = 2) -> str:
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        return "0"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def groove_target(t: float) -> list[float]:
    """Return raw 13-band musical excitation (0..1) at time t (4.0s periodic).

    High-contrast 120 BPM electronic groove: deep valleys between hits so bars
    bounce dramatically, while kicks, snares, synth stabs, and hi-hats push
    across cyan, amber, and vivid red peak caps.
    """
    tau = math.tau
    phase = (t % LOOP_SECS) / LOOP_SECS  # 0..1 over 2 bars (8 beats)
    beat_f = phase * 8.0
    beat_idx = int(beat_f) % 8
    beat_p = beat_f - math.floor(beat_f)

    # 16th-note grid (32 steps per 4s loop)
    sixteenth_f = phase * 32.0
    step16 = int(sixteenth_f) % 32
    step16_p = sixteenth_f - math.floor(sixteenth_f)

    # 1. Punchy 4-on-the-floor kick (pushes cols 0..2 into vivid red peak rows)
    kick_amp = 1.85 if beat_idx in (0, 4) else 1.45
    kick_env = kick_amp * math.exp(-beat_p * 12.0)
    if step16 in (14, 30):
        kick_env = max(kick_env, 1.25 * math.exp(-step16_p * 14.0))

    # 2. Syncopated acid/funk bass (cols 1..4)
    bass_hits = {
        0: 0.95, 2: 0.72, 4: 0.88, 7: 0.80,
        8: 1.00, 11: 0.76, 12: 0.92, 14: 0.85,
        16: 0.95, 18: 0.68, 20: 0.90, 23: 0.82,
        24: 1.05, 26: 0.78, 28: 0.96, 30: 0.88,
    }
    bass_env = 0.0
    for st, amp in bass_hits.items():
        dist = (sixteenth_f - st) % 32.0
        if dist < 2.2:
            bass_env = max(bass_env, amp * math.exp(-dist * 2.4))

    # 3. Snare + clap backbeat on beats 2, 4, 6, 8 (lights cols 4..8 into amber/red)
    snare_env = 0.0
    if beat_idx in (1, 3, 5, 7):
        snare_env = 1.22 * math.exp(-beat_p * 15.0)
    if step16 in (29, 31):
        snare_env = max(snare_env, 0.92 * math.exp(-step16_p * 13.0))

    # 4. Rhythmic synth chord stabs sweeping across the center (cols 4..9)
    chord_center = 6.0 + 1.8 * math.sin(tau * phase)
    chord_hits = {0: 1.05, 6: 0.96, 12: 1.02, 16: 1.08, 22: 0.98, 27: 0.92}
    chord_env = 0.08
    for st, amp in chord_hits.items():
        dist = (sixteenth_f - st) % 32.0
        if dist < 4.0:
            chord_env = max(chord_env, amp * math.exp(-dist * 1.15))

    # 5. Sparkling synth arpeggio dancing across cols 6..12
    arp_cols = [6, 8, 10, 12, 11, 9, 7, 10, 8, 10, 12, 11, 9, 11, 12, 8]
    arp_idx = (step16 // 2) % len(arp_cols)
    arp_col = arp_cols[arp_idx]
    arp_p = (sixteenth_f * 0.5) % 1.0
    arp_env = 1.12 * math.exp(-arp_p * 6.5)

    # 6. Crisp open/closed hi-hats on off-beats (cols 9..12)
    hat_env = 0.0
    if step16 % 4 == 2:
        hat_env = 1.35 * math.exp(-step16_p * 10.0)
    elif step16 % 2 == 0:
        hat_env = 0.48 * math.exp(-step16_p * 13.0)
    if step16 in (14, 30):
        hat_env = max(hat_env, 1.65 * math.exp(-step16_p * 6.5))

    levels: list[float] = []
    for c in range(COLS):
        kick_w = math.exp(-((c - 0.5) / 1.45) ** 2)
        bass_w = math.exp(-((c - 2.4) / 1.55) ** 2)
        snare_w = 0.88 * math.exp(-((c - 5.2) / 1.85) ** 2) + 0.52 * math.exp(-((c - 9.6) / 1.75) ** 2)
        chord_w = math.exp(-((c - chord_center) / 1.90) ** 2)
        arp_w = math.exp(-((c - arp_col) / 1.05) ** 2)
        hat_w = math.exp(-((c - 11.3) / 1.65) ** 2)

        # Gentle noise floor so inactive bands settle down to 1-2 rows
        floor_wave = 0.06 + 0.05 * math.sin(tau * (phase * 2.0 + c * 0.23))

        raw = (
            kick_env * kick_w
            + bass_env * bass_w * 0.82
            + snare_env * snare_w
            + chord_env * chord_w * 0.78
            + arp_env * arp_w * 0.85
            + hat_env * hat_w * 0.90
            + floor_wave
        )
        # Soft saturation preserving punchy peaks up to top red row
        val = 1.0 - math.exp(-raw * 1.52)
        levels.append(max(0.0, min(1.0, val)))
    return levels


def simulate_seamless_loop() -> tuple[list[list[float]], list[list[float]]]:
    """Run ViewMusic's exact hop-rate ballistics and capture one seamless loop."""
    hops_per_loop = int(round(LOOP_SECS * HOP_RATE))  # 375 hops
    smooth = [0.0] * COLS
    peak_val = [0.0] * COLS
    peak_hold = [0] * COLS
    peak_vel = [0.0] * COLS

    total_loops = 3
    captured_levels: list[list[float]] = []
    captured_peaks: list[list[float]] = []

    for loop_i in range(total_loops):
        for h in range(hops_per_loop):
            t = h / HOP_RATE
            targets = groove_target(t)
            for c in range(COLS):
                target = targets[c]
                prev = smooth[c]
                coeff = BAND_DB_ATTACK if target > prev else BAND_DB_RELEASE
                lvl = prev * coeff + target * (1.0 - coeff)
                smooth[c] = lvl

                if lvl >= peak_val[c]:
                    peak_val[c] = lvl
                    peak_hold[c] = PEAK_HOLD_HOPS
                    peak_vel[c] = 0.0
                elif peak_hold[c] > 0:
                    peak_hold[c] -= 1
                else:
                    peak_vel[c] = min(PEAK_MAX_VEL, peak_vel[c] + PEAK_GRAVITY)
                    peak_val[c] = max(lvl, peak_val[c] - peak_vel[c])

            if loop_i == total_loops - 1:
                captured_levels.append(smooth.copy())
                captured_peaks.append(peak_val.copy())

    frame_levels: list[list[float]] = []
    frame_peaks: list[list[float]] = []
    for f in range(NUM_FRAMES):
        idx = int(round(f * hops_per_loop / NUM_FRAMES)) % hops_per_loop
        frame_levels.append(captured_levels[idx])
        frame_peaks.append(captured_peaks[idx])

    return frame_levels, frame_peaks


def quantize_cell_opacity(level: float, peak: float, row: int) -> float:
    """Discrete VFD segment opacity matching retro-spectrum-bars (Bars + Peak).

    Quantizes into crisp hardware levels (0, 0.45, 0.94, 1.0) so bars retain
    soft phosphor top-edge stepping while peak-hold caps snap cleanly without
    muddy mid-fade artifacts.
    """
    lvl = max(0.0, min(1.0, level * GAIN))
    bar_raw = max(0.0, min(1.0, (lvl * ROWS - row) * 1.45))

    pk = max(0.0, min(1.0, peak * GAIN))
    pk_row = max(0, min(ROWS - 1, int(math.floor(pk * ROWS))))
    if row == pk_row and pk >= 0.05:
        return 1.0

    if bar_raw >= 0.68:
        return 0.94
    if bar_raw >= 0.25:
        return 0.45
    return 0.0


def compress_discrete_keyframes(samples: list[float]) -> tuple[list[float], list[float]]:
    """Run-length compress periodic frame samples for SMIL calcMode='discrete'.

    In SMIL discrete mode, each value holds from its keyTime until the next
    keyTime (and the last value holds until the end of the cycle).
    """
    n = len(samples)
    times: list[float] = [0.0]
    vals: list[float] = [samples[0]]
    for f in range(1, n):
        if samples[f] != vals[-1]:
            times.append(f / n)
            vals.append(samples[f])
    return times, vals


def build_svg() -> str:
    frame_levels, frame_peaks = simulate_seamless_loop()

    # Exact sRGB-converted colors from Sony MHC-RX90 (m0)
    c_bez = linear_rgb_hex(0.015, 0.019, 0.022)
    c_bez_dark = linear_rgb_hex(0.015 * 0.32, 0.019 * 0.32, 0.022 * 0.32)
    c_bez_hi = linear_rgb_hex(0.015 * 1.45 + 0.003, 0.019 * 1.45 + 0.003, 0.022 * 1.45 + 0.0035)
    c_rim_hi = linear_rgb_hex(0.015 * 1.7 + 0.004, 0.019 * 1.7 + 0.004, 0.022 * 1.7 + 0.0045)
    c_rim_lo = linear_rgb_hex(0.015 * 0.45, 0.019 * 0.45, 0.022 * 0.45)

    c_gls_center = linear_rgb_hex(0.0018, 0.0050, 0.0062)
    c_gls_edge = linear_rgb_hex(0.0018 * 0.48, 0.0050 * 0.48, 0.0062 * 0.48)
    c_ghost = linear_rgb_hex(0.010, 0.024, 0.026)
    c_grid = linear_rgb_hex(0.03, 0.18, 0.16)

    c1 = linear_rgb_hex(0.014, 0.860, 0.640)  # Sony VFD mint-cyan
    c2 = linear_rgb_hex(1.000, 0.450, 0.045)  # Warm amber
    c3 = linear_rgb_hex(1.000, 0.022, 0.048)  # Vivid VFD red

    seg_px_w, seg_px_h = ndc_size_to_px(SEG_W, SEG_H)
    glw_px_w, glw_px_h = ndc_size_to_px(GLW_W, GLW_H)
    pill_rx = min(seg_px_w, seg_px_h) * 0.24

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(VIEW_W)} {int(VIEW_H)}" '
        f'width="{int(VIEW_W)}" height="{int(VIEW_H)}" role="img" '
        f'aria-label="ViewMusic Retro Spectrum Bars live visualizer preview">'
    )
    out.append("<title>ViewMusic — Retro Spectrum Bars (Sony MHC-RX90 VFD)</title>")

    # Defs: glass vignette, diagonal sheen, VFD scanlines, 3 radial phosphor bloom gradients
    out.append("<defs>")
    out.append(
        f'<radialGradient id="gls" cx="50%" cy="50%" r="65%">'
        f'<stop offset="0%" stop-color="{c_gls_center}"/>'
        f'<stop offset="100%" stop-color="{c_gls_edge}"/>'
        f"</radialGradient>"
    )
    out.append(
        '<linearGradient id="sheen" x1="15%" y1="0%" x2="85%" y2="100%">'
        '<stop offset="0%" stop-color="#9fe8e0" stop-opacity="0"/>'
        '<stop offset="38%" stop-color="#9fe8e0" stop-opacity="0"/>'
        '<stop offset="48%" stop-color="#c4f5ef" stop-opacity="0.045"/>'
        '<stop offset="58%" stop-color="#9fe8e0" stop-opacity="0"/>'
        '<stop offset="100%" stop-color="#9fe8e0" stop-opacity="0"/>'
        "</linearGradient>"
    )
    out.append(
        '<pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse">'
        '<rect width="4" height="2" fill="#000" fill-opacity="0.14"/>'
        "</pattern>"
    )
    for gid, col, peak_op in (("g1", c1, "0.54"), ("g2", c2, "0.62"), ("g3", c3, "0.68")):
        out.append(
            f'<radialGradient id="{gid}" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0%" stop-color="{col}" stop-opacity="{peak_op}"/>'
            f'<stop offset="44%" stop-color="{col}" stop-opacity="0.25"/>'
            f'<stop offset="78%" stop-color="{col}" stop-opacity="0.06"/>'
            f'<stop offset="100%" stop-color="{col}" stop-opacity="0"/>'
            f"</radialGradient>"
        )
    out.append("</defs>")

    # 1. Outer bezel + recessed VFD glass window
    out.append(f'<rect width="{int(VIEW_W)}" height="{int(VIEW_H)}" fill="{c_bez}"/>')

    gx0, gy0 = ndc_to_px(-0.884, 0.822)
    gx1, gy1 = ndc_to_px(0.884, -0.822)
    gw, gh = gx1 - gx0, gy1 - gy0

    out.append(
        f'<rect x="{fmt(gx0 - 6)}" y="{fmt(gy0 - 5)}" width="{fmt(gw + 12)}" height="{fmt(gh + 10)}" '
        f'fill="none" stroke="{c_rim_hi}" stroke-width="1.5"/>'
    )
    out.append(
        f'<rect x="{fmt(gx0 - 3)}" y="{fmt(gy0 - 3)}" width="{fmt(gw + 6)}" height="{fmt(gh + 6)}" '
        f'fill="{c_bez_dark}"/>'
    )
    out.append(
        f'<line x1="{fmt(gx0 - 4)}" y1="{fmt(gy1 + 4)}" x2="{fmt(gx1 + 4)}" y2="{fmt(gy1 + 4)}" '
        f'stroke="{c_bez_hi}" stroke-width="2"/>'
    )
    out.append(
        f'<line x1="{fmt(gx1 + 4)}" y1="{fmt(gy0 - 4)}" x2="{fmt(gx1 + 4)}" y2="{fmt(gy1 + 4)}" '
        f'stroke="{c_rim_lo}" stroke-width="2"/>'
    )

    out.append(f'<rect x="{fmt(gx0)}" y="{fmt(gy0)}" width="{fmt(gw)}" height="{fmt(gh)}" fill="url(#gls)"/>')
    out.append(f'<rect x="{fmt(gx0)}" y="{fmt(gy0)}" width="{fmt(gw)}" height="{fmt(gh)}" fill="url(#sheen)"/>')
    out.append(f'<rect x="{fmt(gx0)}" y="{fmt(gy0)}" width="{fmt(gw)}" height="{fmt(gh)}" fill="url(#scan)"/>')

    # 2. Horizontal scale grid lines
    grid_x0, _ = ndc_to_px(-(SPAN_X + 0.04) * 0.5, 0)
    grid_x1, _ = ndc_to_px((SPAN_X + 0.04) * 0.5, 0)
    grid_w = grid_x1 - grid_x0
    out.append(f'<g fill="{c_grid}">')
    for i in range(5):
        y_ndc = (-SPAN_Y * 0.5 - CELL_H * 0.14) if i == 0 else (-SPAN_Y * 0.5 + i * SPAN_Y * 0.25)
        _, py = ndc_to_px(0, y_ndc)
        h_px = 1.4 if i == 0 else 0.9
        a = 0.20 if i == 0 else 0.14
        out.append(
            f'<rect x="{fmt(grid_x0)}" y="{fmt(py - h_px * 0.5)}" '
            f'width="{fmt(grid_w)}" height="{fmt(h_px)}" opacity="{fmt(a)}"/>'
        )
    out.append("</g>")

    # Precompute cell centers & zone colors
    cells: list[tuple[int, int, float, float, str, str]] = []
    for c in range(COLS):
        for r in range(ROWS):
            x_ndc = -SPAN_X * 0.5 + (c + 0.5) * CELL_W
            y_ndc = -SPAN_Y * 0.5 + (r + 0.5) * CELL_H
            cx, cy = ndc_to_px(x_ndc, y_ndc)
            y_norm = (r + 0.5) / ROWS
            if y_norm >= Z2:
                col, glow_id = c3, "g3"
            elif y_norm >= Z1:
                col, glow_id = c2, "g2"
            else:
                col, glow_id = c1, "g1"
            cells.append((c, r, cx, cy, col, glow_id))

    # 3. Static unlit ghost matrix
    out.append(f'<g fill="{c_ghost}" opacity="0.52">')
    w_s = fmt(seg_px_w, 1)
    h_s = fmt(seg_px_h, 1)
    rx_s = fmt(pill_rx, 1)
    for _, _, cx, cy, _, _ in cells:
        out.append(
            f'<rect x="{fmt(cx - seg_px_w * 0.5, 1)}" y="{fmt(cy - seg_px_h * 0.5, 1)}" '
            f'width="{w_s}" height="{h_s}" rx="{rx_s}"/>'
        )
    out.append("</g>")

    # 4. Animated lit segments + radial phosphor bloom halos
    dur = f"{fmt(LOOP_SECS, 1)}s"
    rx_glw_s = fmt(glw_px_w * 0.5, 1)
    ry_glw_s = fmt(glw_px_h * 0.5, 1)
    animated_count = 0
    static_lit_count = 0
    for c, r, cx, cy, col, glow_id in cells:
        series = [quantize_cell_opacity(frame_levels[f][c], frame_peaks[f][c], r) for f in range(NUM_FRAMES)]
        if max(series) == 0.0:
            continue

        px = cx - seg_px_w * 0.5
        py = cy - seg_px_h * 0.5
        glow_el = (
            f'<ellipse cx="{fmt(cx, 1)}" cy="{fmt(cy, 1)}" '
            f'rx="{rx_glw_s}" ry="{ry_glw_s}" fill="url(#{glow_id})"/>'
        )
        pill_el = (
            f'<rect x="{fmt(px, 1)}" y="{fmt(py, 1)}" '
            f'width="{w_s}" height="{h_s}" rx="{rx_s}" fill="{col}"/>'
        )

        times, vals = compress_discrete_keyframes(series)
        if len(vals) == 1:
            static_lit_count += 1
            out.append(f'<g opacity="{fmt(vals[0], 2)}">{glow_el}{pill_el}</g>')
            continue

        animated_count += 1
        kt_str = ";".join(fmt(t, 2) for t in times)
        v_str = ";".join(fmt(v, 2) for v in vals)
        out.append(
            f'<g opacity="{fmt(vals[0], 2)}">{glow_el}{pill_el}'
            f'<animate attributeName="opacity" calcMode="discrete" values="{v_str}" '
            f'keyTimes="{kt_str}" dur="{dur}" repeatCount="indefinite"/>'
            f"</g>"
        )

    # Subtle bottom-right deck engraving on bezel
    out.append(
        f'<text x="{fmt(gx1)}" y="{fmt(gy1 + 31)}" text-anchor="end" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
        f'font-size="11" letter-spacing="1.6" fill="#6f8088" opacity="0.85">'
        f"VIEWMUSIC · RETRO SPECTRUM BARS · SONY MHC-RX90 VFD</text>"
    )
    out.append("</svg>")
    svg = "\n".join(out) + "\n"
    print(
        f"Generated SVG: {len(svg.encode('utf-8')):,} bytes "
        f"({animated_count} animated cells, {static_lit_count} static lit cells, "
        f"{COLS * ROWS - animated_count - static_lit_count} always-off cells)"
    )
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate animated SVG demo for README.md")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/media/retro-spectrum-demo.svg"),
        help="Destination SVG file path",
    )
    args = parser.parse_args()
    svg = build_svg()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
