// field: fullscreen triangle sampling a row-major storage buffer of straight-alpha
// RGBA colors laid out from the bottom-left (`i = row * res + col`). Avoids CPU
// per-cell quad packing and shrinks vertex shading from `6 * res^2` down to 3 vertices.

struct FieldParams {
    base: u32,
    res: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: FieldParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // All vertices have w = 1, so linear equals perspective-correct interpolation
    // without the per-fragment 1/w.
    @location(0) @interpolate(linear) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle covering NDC [-1, 1]^2 with uv (0, 0) at bottom-left
    // (-1, -1) and (1, 1) at top-right (1, 1).
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid & 2u) * 2 - 1);
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    return out;
}

fn shade(in: VsOut) -> vec4<f32> {
    // Pixel centers keep uv at least half a pixel inside (0, 1), so col/row are always in
    // 0..res without clamping. The CPU zero-fills missing cells, which both blends leave
    // as a no-op, so there is no bounds check or discard.
    let res_f = f32(params.res);
    let col = u32(in.uv.x * res_f);
    let row = u32(in.uv.y * res_f);
    let c = cells[params.base + row * params.res + col];
    return vec4<f32>(c.rgb * c.a, c.a);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in);
}

// Feedback path: the accumulator (location 0) and the surface (location 1) get the same color.
struct FeedbackOut {
    @location(0) accum: vec4<f32>,
    @location(1) surface: vec4<f32>,
};

@fragment
fn fs_main_mrt(in: VsOut) -> FeedbackOut {
    let c = shade(in);
    return FeedbackOut(c, c);
}
