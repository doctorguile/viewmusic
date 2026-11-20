// field: fullscreen triangle sampling a row-major storage buffer of straight-alpha
// RGBA colors laid out from the bottom-left (`i = row * res + col`). Avoids CPU
// per-cell quad packing and shrinks vertex shading from `6 * res^2` down to 3 vertices.

struct FieldParams {
    base: u32,
    count: u32,
    res: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: FieldParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let res_f = f32(params.res);
    let col = min(u32(max(in.uv.x, 0.0) * res_f), params.res - 1u);
    let row = min(u32(max(in.uv.y, 0.0) * res_f), params.res - 1u);
    let idx = row * params.res + col;
    if idx >= params.count {
        discard;
    }
    let c = cells[params.base + idx];
    return vec4<f32>(c.rgb * c.a, c.a);
}
