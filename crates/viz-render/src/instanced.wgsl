// instanced shapes: one vertex shader, five fragment paths (rect/circle/triangle/pill/glow).
//
// Coordinate contract: artifact space maps directly to NDC (x right, y up, -1..1).
// The unit quad arrives in local space [-0.5, 0.5]^2; we scale by (w,h), rotate by
// `rot` (radians, CCW), then translate to (x,y). Local coords and absolute quad size
// are forwarded so the fragment stage can mask circle / triangle / pill / glow. Output
// color is premultiplied so the alpha-over and additive blend states (set on the
// pipeline) both behave.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Local position within the quad, in [-0.5, 0.5]^2.
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Absolute quad size (|w|, |h|) in NDC units for aspect-true SDF masking.
    @location(2) size: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) quad: vec2<f32>,        // unit-quad vertex, [-0.5, 0.5]^2
    @location(1) xywh: vec4<f32>,        // x, y, w, h
    @location(2) rot: f32,               // radians, CCW
    @location(3) color: vec4<f32>,       // straight-alpha RGBA, 0..1
) -> VsOut {
    var out: VsOut;
    // Every fragment path outputs premul(rgb, a * k), which is exactly vec4(0) when the
    // straight alpha is <= 0 (the CPU clamps it to 0..1), and vec4(0) is a no-op under
    // both blend states. Collapse such instances to one point so the rasterizer emits no
    // fragments (e.g. the unlit cells of full-grid glow / pill bar layers).
    if (color.a <= 0.0) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return out;
    }

    let center = xywh.xy;
    let size = xywh.zw;
    // Scale into NDC half-extents.
    let scaled = quad * size;
    // Rotate CCW about the center.
    let c = cos(rot);
    let s = sin(rot);
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    let pos = center + rotated;

    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.local = quad;
    out.color = color;
    out.size = abs(size);
    return out;
}

// Premultiply straight-alpha color for blending.
fn premul(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn rect(in: VsOut) -> vec4<f32> {
    // Whole quad is filled.
    return premul(in.color);
}

fn circle(in: VsOut) -> vec4<f32> {
    // Inscribed circle: discard fragments outside radius 0.5 of the local quad.
    if (dot(in.local, in.local) > 0.25) {
        discard;
    }
    return premul(in.color);
}

fn triangle(in: VsOut) -> vec4<f32> {
    // Upward isoceles triangle inscribed in the quad: apex at local (0, +0.5),
    // base from (-0.5, -0.5) to (+0.5, -0.5). Keep fragments inside the two slanted
    // edges (the base edge is the quad bottom, already inside).
    //
    // Edges (apex A=(0,0.5)):
    //   left  edge A -> (-0.5,-0.5): keep points to its right
    //   right edge A -> ( 0.5,-0.5): keep points to its left
    // Using the line through A with the base corner, the inside test reduces to:
    //   |x| <= 0.5 * (0.5 - y)   for y in [-0.5, 0.5]
    let half_width = 0.5 * (0.5 - in.local.y);
    if (abs(in.local.x) > half_width) {
        discard;
    }
    return premul(in.color);
}

fn pill(in: VsOut) -> vec4<f32> {
    // Aspect-true rounded-box SDF in NDC scale so corner radius stays circular
    // across elongated segments.
    let size = max(in.size, vec2<f32>(1e-5));
    let p = in.local * size;
    let half_ext = 0.5 * size;
    let r = min(half_ext.x, half_ext.y) * 0.36;
    let d = length(max(abs(p) - (half_ext - vec2<f32>(r)), vec2<f32>(0.0))) - r;

    let aa = max(fwidth(d), 0.0005);
    let cov = 1.0 - smoothstep(-aa, aa, d);
    if (cov <= 0.001) {
        discard;
    }

    // Subtle physical phosphor/bevel shading only for bright emissive colors
    // (never brightening dark LCD ink or dim unlit ghost segments).
    let luma = dot(in.color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let emissive = smoothstep(0.32, 0.90, luma);
    let center = clamp(1.0 - length(in.local * vec2<f32>(1.2, 1.5)), 0.0, 1.0);
    let edge_shade = mix(1.0, 0.88 + 0.16 * center, emissive);
    let hot_core = emissive * center * center * 0.22;
    let rgb = clamp(
        in.color.rgb * edge_shade + vec3<f32>(hot_core),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return premul(vec4<f32>(rgb, in.color.a * cov));
}

fn glow(in: VsOut) -> vec4<f32> {
    // Smooth optical bloom falloff across [-0.5, 0.5]^2 fading cleanly to 0
    // before the quad boundaries (zero hard box edge).
    let q = abs(in.local);
    let core_d = length(max(q - vec2<f32>(0.14), vec2<f32>(0.0)));
    // 1.0 - smoothstep(lo, hi, x) avoids WGSL/MSL UB from low >= high while
    // remaining algebraically identical to smoothstep(hi, lo, x).
    let edge_mask = (1.0 - smoothstep(0.12, 0.49, max(q.x, q.y)))
        * (1.0 - smoothstep(0.16, 0.54, length(q)));
    let falloff = exp(-14.0 * core_d * core_d) * edge_mask;
    if (falloff <= 0.001) {
        discard;
    }
    return premul(vec4<f32>(in.color.rgb, in.color.a * falloff));
}

@fragment
fn fs_rect(in: VsOut) -> @location(0) vec4<f32> {
    return rect(in);
}

@fragment
fn fs_circle(in: VsOut) -> @location(0) vec4<f32> {
    return circle(in);
}

@fragment
fn fs_triangle(in: VsOut) -> @location(0) vec4<f32> {
    return triangle(in);
}

@fragment
fn fs_pill(in: VsOut) -> @location(0) vec4<f32> {
    return pill(in);
}

@fragment
fn fs_glow(in: VsOut) -> @location(0) vec4<f32> {
    return glow(in);
}

// Feedback path: the accumulator (location 0) and the surface (location 1) get the same color.
struct FeedbackOut {
    @location(0) accum: vec4<f32>,
    @location(1) surface: vec4<f32>,
};

@fragment
fn fs_rect_mrt(in: VsOut) -> FeedbackOut {
    let c = rect(in);
    return FeedbackOut(c, c);
}

@fragment
fn fs_circle_mrt(in: VsOut) -> FeedbackOut {
    let c = circle(in);
    return FeedbackOut(c, c);
}

@fragment
fn fs_triangle_mrt(in: VsOut) -> FeedbackOut {
    let c = triangle(in);
    return FeedbackOut(c, c);
}

@fragment
fn fs_pill_mrt(in: VsOut) -> FeedbackOut {
    let c = pill(in);
    return FeedbackOut(c, c);
}

@fragment
fn fs_glow_mrt(in: VsOut) -> FeedbackOut {
    let c = glow(in);
    return FeedbackOut(c, c);
}
