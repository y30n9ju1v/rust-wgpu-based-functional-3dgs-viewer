// =============================================================================
// render.wgsl — 3D Gaussian Splatting 빌보드 렌더링 셰이더
//
// 동작 개요:
//   1. 가우시안 1개당 삼각형 2개(= 6 vertex)로 이루어진 빌보드 쿼드를 생성한다.
//   2. 버텍스 버퍼 없이 vertex_index만으로 쿼드 위치를 인라인 계산한다.
//   3. 3D 공분산을 Jacobian으로 2D 화면 공간에 투영해 Conic을 구한다.
//   4. View Direction에 따른 Degree-3 Spherical Harmonics 색상을 계산한다.
//   5. 프래그먼트에서 타원형 가우시안 감쇠와 알파 블렌딩을 적용한다.
// =============================================================================

// -----------------------------------------------------------------------------
// Uniform / Storage 바인딩
// -----------------------------------------------------------------------------

struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // .xy = (width, height) in pixels; .zw unused
    viewport: vec4<f32>,
}

struct Gaussian {
    pos: vec3<f32>,
    // logit 공간 값 — sigmoid 적용 후 [0, 1]
    opacity: f32,
    color_dc: vec3<f32>,
    _pad0: f32,
    // log 공간 값 — exp 적용 후 실제 축 크기
    scale: vec3<f32>,
    _pad1: f32,
    rot: vec4<f32>,
    // 45개 SH 계수를 vec4 × 12(= 48 f32)로 패킹.
    // 채널 순서: R[0..14], G[15..29], B[30..44]
    f_rest: array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;

// -----------------------------------------------------------------------------
// 버텍스 셰이더 출력
// -----------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    // 가우시안 중심에서 현재 프래그먼트까지의 화면 공간 오프셋 (픽셀 단위)
    @location(2) delta_screen: vec2<f32>,
    // 2D 공분산 역행렬의 상삼각 성분 (A, B, C) — 타원 방정식 계수
    @location(3) conic: vec3<f32>,
}

// -----------------------------------------------------------------------------
// 헬퍼 함수
// -----------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// 쿼드 6 vertex의 로컬 UV를 반환한다 (삼각형 2개, CCW).
// naga는 런타임 인덱스로 배열 리터럴을 접근할 수 없어 switch를 사용한다.
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

// packed vec4 배열에서 vec4 하나를 꺼낸다.
// naga는 런타임 값으로 배열을 인덱싱할 수 없어 switch로 분기한다.
fn get_f_rest_vec(f_rest: array<vec4<f32>, 12>, vec_idx: u32) -> vec4<f32> {
    switch vec_idx {
        case  0u: { return f_rest[ 0]; }
        case  1u: { return f_rest[ 1]; }
        case  2u: { return f_rest[ 2]; }
        case  3u: { return f_rest[ 3]; }
        case  4u: { return f_rest[ 4]; }
        case  5u: { return f_rest[ 5]; }
        case  6u: { return f_rest[ 6]; }
        case  7u: { return f_rest[ 7]; }
        case  8u: { return f_rest[ 8]; }
        case  9u: { return f_rest[ 9]; }
        case 10u: { return f_rest[10]; }
        default:  { return f_rest[11]; }
    }
}

// packed vec4 배열에서 스칼라 SH 계수 하나를 꺼낸다.
fn get_f_rest(f_rest: array<vec4<f32>, 12>, index: u32) -> f32 {
    return get_f_rest_vec(f_rest, index / 4u)[index % 4u];
}

// deg_offset 번째 SH 계수의 RGB 트리플렛을 반환한다.
// f_rest 레이아웃: 동일 차수의 R 계수 15개 → G 15개 → B 15개 순으로 저장.
fn sh_coeff(f_rest: array<vec4<f32>, 12>, deg_offset: u32) -> vec3<f32> {
    return vec3<f32>(
        get_f_rest(f_rest, deg_offset),
        get_f_rest(f_rest, deg_offset + 15u),
        get_f_rest(f_rest, deg_offset + 30u),
    );
}

// -----------------------------------------------------------------------------
// SH 차수별 색상 기여 계산
// -----------------------------------------------------------------------------

fn sh_degree1(f_rest: array<vec4<f32>, 12>, x: f32, y: f32, z: f32) -> vec3<f32> {
    let C1 = -0.4886025119029199;
    return - C1 * y * sh_coeff(f_rest, 0u)
           + C1 * z * sh_coeff(f_rest, 1u)
           - C1 * x * sh_coeff(f_rest, 2u);
}

fn sh_degree2(f_rest: array<vec4<f32>, 12>, x: f32, y: f32, z: f32) -> vec3<f32> {
    let C2_0 =  1.0925484305920792;
    let C2_1 = -1.0925484305920792;
    let C2_2 =  0.31539156525252005;
    let C2_3 = -1.0925484305920792;
    let C2_4 =  0.5462742152960396;
    let xx = x*x; let yy = y*y; let zz = z*z;
    let xy = x*y; let yz = y*z; let xz = x*z;
    return C2_0 * xy               * sh_coeff(f_rest, 3u)
         + C2_1 * yz               * sh_coeff(f_rest, 4u)
         + C2_2 * (2.0*zz-xx-yy)  * sh_coeff(f_rest, 5u)
         + C2_3 * xz               * sh_coeff(f_rest, 6u)
         + C2_4 * (xx - yy)        * sh_coeff(f_rest, 7u);
}

fn sh_degree3(f_rest: array<vec4<f32>, 12>, x: f32, y: f32, z: f32) -> vec3<f32> {
    let C3_0 = -0.5900435899266435;
    let C3_1 =  2.890611442640554;
    let C3_2 = -0.4570457994644658;
    let C3_3 =  0.3731763325901154;
    let C3_4 = -0.4570457994644658;
    let C3_5 =  1.445305721320277;
    let C3_6 = -0.5900435899266435;
    let xx = x*x; let yy = y*y; let zz = z*z;
    let xy = x*y; let xz = x*z;
    return C3_0 * y * (3.0*xx - yy)           * sh_coeff(f_rest,  8u)
         + C3_1 * xy * z                       * sh_coeff(f_rest,  9u)
         + C3_2 * y * (4.0*zz - xx - yy)       * sh_coeff(f_rest, 10u)
         + C3_3 * z * (2.0*zz - 3.0*xx-3.0*yy) * sh_coeff(f_rest, 11u)
         + C3_4 * x * (4.0*zz - xx - yy)       * sh_coeff(f_rest, 12u)
         + C3_5 * z * (xx - yy)                * sh_coeff(f_rest, 13u)
         + C3_6 * x * (xx - 3.0*yy)            * sh_coeff(f_rest, 14u);
}

// DC + Degree 1~3 SH를 합산해 view-dependent 색상을 반환한다.
// +0.5 bias는 3DGS 원본 구현의 색상 범위 보정값이다.
fn compute_sh(g: Gaussian, dir: vec3<f32>) -> vec3<f32> {
    let C0 = 0.28209479177387814;
    let x = dir.x; let y = dir.y; let z = dir.z;
    let color = C0 * g.color_dc
        + sh_degree1(g.f_rest, x, y, z)
        + sh_degree2(g.f_rest, x, y, z)
        + sh_degree3(g.f_rest, x, y, z);
    return max(color + 0.5, vec3<f32>(0.0));
}

// -----------------------------------------------------------------------------
// 공분산 투영 헬퍼
// -----------------------------------------------------------------------------

// rot는 (w, x, y, z) 순서로 저장된 unit quaternion이다 — (r.x=w, r.y=x, ...).
fn quat_to_mat(r: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        1.0 - 2.0*(r.z*r.z + r.w*r.w), 2.0*(r.y*r.z - r.x*r.w),        2.0*(r.y*r.w + r.x*r.z),
        2.0*(r.y*r.z + r.x*r.w),        1.0 - 2.0*(r.y*r.y + r.w*r.w), 2.0*(r.z*r.w - r.x*r.y),
        2.0*(r.y*r.w - r.x*r.z),        2.0*(r.z*r.w + r.x*r.y),        1.0 - 2.0*(r.y*r.y + r.z*r.z)
    );
}

// 핀홀 카메라 모델의 국소 선형화 Jacobian (∂π/∂p_view).
// 3행은 0 — 깊이 방향 기여는 2D 투영에서 불필요하다.
fn build_jacobian(tx: f32, ty: f32, tz: f32, fx: f32, fy: f32) -> mat3x3<f32> {
    let tz2 = tz * tz;
    return mat3x3<f32>(
        fx / tz, 0.0, -(fx * tx) / tz2,
        0.0, fy / tz, -(fy * ty) / tz2,
        0.0, 0.0, 0.0
    );
}

// 가우시안의 3D 공분산 행렬 Σ = RS(RS)^T 를 계산한다.
fn compute_cov3d(g: Gaussian) -> mat3x3<f32> {
    let s = exp(g.scale);
    let S = mat3x3<f32>(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
    let M = quat_to_mat(g.rot) * S;
    return M * transpose(M);
}

// 뷰 행렬의 회전 성분(상위 3×3)을 추출한다.
fn view_rotation() -> mat3x3<f32> {
    return mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
}

// 3D 공분산을 화면 공간 2D 공분산으로 투영한다.
// 반환값: (σ_xx, σ_xy, σ_yy) — 대칭 2×2 행렬의 상삼각 성분 (픽셀² 단위).
// low-pass filter (+0.3)는 서브픽셀 aliasing을 억제한다.
fn project_cov3d(g: Gaussian, pos_view: vec3<f32>, fx: f32, fy: f32) -> vec3<f32> {
    let J = build_jacobian(pos_view.x, pos_view.y, pos_view.z, fx, fy);
    let T = J * view_rotation();
    let cov3x3 = T * compute_cov3d(g) * transpose(T);

    var cov2d = vec3<f32>(cov3x3[0][0], cov3x3[0][1], cov3x3[1][1]);
    cov2d.x += 0.3;
    cov2d.z += 0.3;
    return cov2d;
}

// 2D 공분산의 역행렬 계수 (A, B, C)를 반환한다.
// 프래그먼트에서 -0.5*(A·dx² + B·dx·dy + C·dy²)로 가우시안 지수를 계산한다.
// det 하한 1e-7은 퇴화 타원(행렬식 ≈ 0) 에서의 division-by-zero를 방지한다.
fn cov2d_to_conic(cov2d: vec3<f32>) -> vec3<f32> {
    let det_inv = 1.0 / max(cov2d.x * cov2d.z - cov2d.y * cov2d.y, 1e-7);
    return vec3<f32>(cov2d.z * det_inv, -2.0 * cov2d.y * det_inv, cov2d.x * det_inv);
}

// -----------------------------------------------------------------------------
// 버텍스 셰이더 — 헬퍼
// -----------------------------------------------------------------------------

// 카메라 파라미터에서 픽셀 단위 초점 거리를 계산한다.
fn focal_lengths() -> vec2<f32> {
    let W = camera.viewport.x;
    let H = camera.viewport.y;
    // projection[0][0] = 2f/W 이므로 역산
    return vec2<f32>(
        camera.projection[0][0] * W * 0.5,
        camera.projection[1][1] * H * 0.5,
    );
}

// 화면 공간 오프셋을 clip space 오프셋으로 변환한다.
fn screen_to_clip_offset(offset_screen: vec2<f32>, clip_w: f32) -> vec2<f32> {
    let W = camera.viewport.x;
    let H = camera.viewport.y;
    let offset_ndc = offset_screen / vec2<f32>(W * 0.5, H * 0.5);
    // NDC offset을 clip space로 역변환. clip_pos.z/w는 그대로 유지한다.
    return offset_ndc * clip_w;
}

// 퇴화 vertex — 카메라 뒤쪽 가우시안을 컬링할 때 반환한다.
fn degenerate_vertex() -> VertexOutput {
    return VertexOutput(
        vec4<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
        vec2<f32>(0.0),
        vec3<f32>(0.0),
    );
}

// -----------------------------------------------------------------------------
// 버텍스 셰이더
// -----------------------------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let real_idx = sorted_indices[idx / 6u];
    let g = gaussians[real_idx];
    let uv = corner_uv(idx % 6u);

    let pos_view = camera.view * vec4<f32>(g.pos, 1.0);
    // wgpu/glam right-hand 좌표계에서 카메라 앞쪽은 -Z.
    // tz > 0 이면 카메라 뒤쪽이므로 퇴화 vertex로 컬링한다.
    if pos_view.z > 0.0 {
        return degenerate_vertex();
    }

    let f = focal_lengths();
    let cov2d = project_cov3d(g, pos_view.xyz, f.x, f.y);

    // 3σ 범위를 쿼드 반경으로 사용해 가우시안을 완전히 포함시킨다.
    let extent = vec2<f32>(ceil(3.0 * sqrt(cov2d.x)), ceil(3.0 * sqrt(cov2d.z)));
    let offset_screen = uv * extent;

    let clip_pos = camera.projection * vec4<f32>(pos_view.xyz, 1.0);
    let offset_clip = screen_to_clip_offset(offset_screen, clip_pos.w);

    var out: VertexOutput;
    out.clip_pos      = vec4<f32>(clip_pos.xy + offset_clip, clip_pos.z, clip_pos.w);
    out.color         = compute_sh(g, normalize(g.pos - camera.camera_pos.xyz));
    out.alpha         = sigmoid(g.opacity);
    out.delta_screen  = offset_screen;
    out.conic         = cov2d_to_conic(cov2d);
    return out;
}

// -----------------------------------------------------------------------------
// 프래그먼트 셰이더
// -----------------------------------------------------------------------------

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let d = input.delta_screen;
    // 타원형 2D 가우시안: exp(-0.5 * (A·x² + B·xy + C·y²))
    let power = -0.5 * (input.conic.x * d.x * d.x
                      + input.conic.y * d.x * d.y
                      + input.conic.z * d.y * d.y);
    // power > 0 은 공분산 역행렬이 퇴화한 경우에만 발생하며, 수치 오차를 방어한다.
    if power > 0.0 { discard; }

    let alpha = input.alpha * exp(power);
    // 8-bit 알파 하한 미만은 블렌딩 비용만 발생시키므로 버린다.
    if alpha < 1.0 / 255.0 { discard; }

    return vec4<f32>(input.color, alpha);
}
