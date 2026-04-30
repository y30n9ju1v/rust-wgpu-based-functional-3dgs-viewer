struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct Gaussian {
    pos: vec3<f32>,
    opacity: f32,
    color_dc: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    _pad1: f32,
    rot: vec4<f32>,
    f_rest: array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) uv: vec2<f32>,
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

/// 쿼드의 6개 꼭짓점 UV를 반환한다 (삼각형 2개, CCW).
///
/// naga는 런타임 값으로 배열 리터럴을 인덱싱하는 것을 허용하지 않아
/// switch로 각 코너를 직접 반환한다.
///
/// 코너 배치:
///   0(-1,-1)  1(+1,-1)  2(-1,+1)   ← 삼각형 1
///   3(-1,+1)  4(+1,-1)  5(+1,+1)   ← 삼각형 2
fn corner_uv(corner_idx: u32) -> vec2<f32> {
    switch corner_idx {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let gaussian_idx = idx / 6u;
    let corner_idx = idx % 6u;
    
    // naga는 런타임 인덱스로 배열 리터럴 접근을 허용하지 않으므로 switch로 분기한다.
    let uv = corner_uv(corner_idx);
    
    let g = gaussians[gaussian_idx];
    
    let pos_view = camera.view * vec4<f32>(g.pos, 1.0);
    let screen_offset = vec4<f32>(uv * max(g.scale.x, g.scale.y) * 0.5, 0.0, 0.0);
    let pos_proj = camera.projection * (pos_view + screen_offset);
    
    var output: VertexOutput;
    output.clip_pos = pos_proj;
    output.color = g.color_dc;
    output.alpha = sigmoid(g.opacity);
    output.uv = uv;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dist2 = dot(input.uv, input.uv);
    if dist2 > 1.0 {
        discard;
    }
    let weight = exp(-0.5 * dist2 * 4.0);
    
    return vec4<f32>(input.color, input.alpha * weight);
}
