struct Globals {
    mvp: mat4x4<f32>,
    size: vec2<f32>,
    _pad0: u32,
    _pad1: u32,
};

struct Locals {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Lines2 {
    p: vec4<f32>,
    color: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0)
@binding(0)
var<uniform> globals: Globals;

@group(1)
@binding(0)
var<uniform> locals: Locals;

@group(2)
@binding(0)
var<uniform> lines2: Lines2;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // 矩形の4点0, 1, 2, 3:: (0, 0), (1, 0), (0, 1), (1, 1)
    let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
    // 中心を左上にずらして、中心にpositionが来るようにする
    //let offset = vec2<f32>(tc.x * globals.size.x /2.0, tc.y * globals.size.y /2.0);
    let offset = vec2<f32>(tc.x * globals.size.x, tc.y * globals.size.y);
    let pos = globals.mvp * vec4<f32>(locals.position + offset, 0.0, 1.0);
    let color = vec4<f32>(255.0, 255.0, 255.0, 255.0) / 255.0;


    return VertexOutput(pos, tc, color);
}


@group(0)
@binding(1)
var tex: texture_2d<f32>;
@group(0)
@binding(2)
var sam: sampler;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color * textureSampleLevel(tex, sam, vertex.tex_coords, 0.0);
}

@vertex
fn vs_main_bond(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = globals.mvp * vec4<f32>(lines2.p[0], lines2.p[1], 0.0, 1.0);
    let color = vec4<f32>(255.0, 255.0, 255.0, 255.0) / 255.0;

    return VertexOutput(pos, vec2<f32>(0.0, 0.0), color);
}

@fragment
fn fs_main_bond(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //return vertex.color;
}

