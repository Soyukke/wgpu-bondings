struct Globals {
    mvp: mat4x4<f32>,
    size: vec2<f32>,
    _pad0: u32,
    _pad1: u32,
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
var<uniform> lines2: Lines2;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // lineを表示したいだけ
    // (p[0], p[1])から(p[2], p[3])のLine
    let pos = globals.mvp * vec4<f32>(lines2.p[2u*vi], lines2.p[2u*vi+1u], 0.0, 1.0);
    // 距離に応じて減衰 透明化したいがわからん。
    let color = vec4<f32>(f32(lines2.color), 0.0, 0.0, 255.) / 255.0;

    return VertexOutput(pos, vec2<f32>(0.0, 0.0), color);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    // 距離に応じて変えたいところ
    // terminal_orange: 0.9490196078431372, 0.6352941176470588, 0.43529411764705883
    //return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return vertex.color;
}

