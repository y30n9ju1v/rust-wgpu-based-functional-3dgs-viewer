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

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let gaussian_idx = idx / 6u;
    let corner_idx = idx % 6u;
    
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    );
    let uv = corners[corner_idx];
    
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
