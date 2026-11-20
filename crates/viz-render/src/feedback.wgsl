// feedback: fullscreen triangle + decay multiply.
//
// vs_fullscreen emits a single oversized triangle covering the viewport using only the
// vertex index (no vertex buffer). fs_decay samples the previous accumulator, scales it
// by `decay`, and writes the result to both the new accumulator and the surface.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle covering [-1,1]^2 with the third vertex past the corner.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = pos[vid];
    var out: VsOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // UV in [0,1]; y flipped so texel (0,0) maps to the bottom-left of the framebuffer.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct Decay {
    factor: f32,
};
@group(1) @binding(0) var<uniform> decay: Decay;

struct FeedbackOut {
    @location(0) accum: vec4<f32>,
    @location(1) surface: vec4<f32>,
};

@fragment
fn fs_decay(in: VsOut) -> FeedbackOut {
    let c = textureSample(src_tex, src_sampler, in.uv) * decay.factor;
    return FeedbackOut(c, c);
}
