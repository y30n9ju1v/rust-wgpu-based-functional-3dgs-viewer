// =============================================================================
// render.wgsl — 3D Gaussian Splatting 빌보드 렌더링 셰이더
//
// 동작 개요:
//   1. 가우시안 1개당 삼각형 2개(= 6 vertex)로 이루어진 빌보드 쿼드를 생성한다.
//   2. 버텍스 버퍼 없이 vertex_index만으로 쿼드 위치를 인라인 계산한다.
//   3. 프래그먼트에서 가우시안 감쇠(가중치)를 적용해 알파 블렌딩한다.
// =============================================================================

// -----------------------------------------------------------------------------
// Uniform / Storage 바인딩
// -----------------------------------------------------------------------------

/// 카메라 뷰·투영 행렬 (binding=0, Uniform Buffer)
struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

/// GPU 업로드용 가우시안 구조체 (binding=1, Storage Buffer)
///
/// Rust의 `GaussianGpu`와 메모리 레이아웃이 1:1로 일치해야 한다.
/// vec3<f32> 뒤에는 WGSL이 자동으로 4바이트 패딩을 삽입하므로
/// Rust 쪽에서도 `_pad` 필드로 맞춰준다.
struct Gaussian {
    pos: vec3<f32>,
    opacity: f32,       // logit 공간 — sigmoid 적용 후 0~1
    color_dc: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,   // 로그 스케일 — exp 적용 후 실제 크기
    _pad1: f32,
    rot: vec4<f32>,
    // f_rest: 45개 f32를 12개 vec4(= 48 f32)로 패딩 포함해 저장
    f_rest: array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;

// -----------------------------------------------------------------------------
// 버텍스 셰이더 출력
// -----------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    /// 쿼드 로컬 UV — (-1,-1) ~ (+1,+1), 원 마스킹과 감쇠 계산에 사용
    @location(2) uv: vec2<f32>,
}

// -----------------------------------------------------------------------------
// 헬퍼 함수
// -----------------------------------------------------------------------------

/// logit → 확률 변환 (opacity 활성화 함수)
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

/// 쿼드의 6개 꼭짓점 UV를 반환한다 (삼각형 2개, CCW).
///
/// naga는 런타임 값으로 배열 리터럴을 인덱싱하는 것을 허용하지 않아
/// switch로 각 코너를 직접 반환한다.
///
/// 코너 배치 (쿼드를 정면에서 봤을 때):
///
///   2(-1,+1) ─── 5(+1,+1)
///      │     ╲       │
///      │      ╲      │
///   0(-1,-1) ─── 1(+1,-1)
///
///   삼각형 1: 0→1→2  /  삼각형 2: 3→4→5 (3=2, 4=1)
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

// -----------------------------------------------------------------------------
// 버텍스 셰이더
// -----------------------------------------------------------------------------

/// 가우시안 하나를 빌보드 쿼드(6 vertex)로 확장한다.
///
/// draw(0..gaussian_count * 6, 0..1) 호출을 가정한다.
/// - gaussian_idx = vertex_index / 6  → 어떤 가우시안인지
/// - corner_idx   = vertex_index % 6  → 쿼드의 어느 꼭짓점인지
///
/// 투영 방식 (간략화된 2D billboard):
///   1. 가우시안 중심을 뷰 공간으로 변환
///   2. scale 크기만큼 뷰 공간에서 오프셋을 더해 쿼드 꼭짓점 위치 결정
///   3. 투영 행렬 적용
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let gaussian_idx = idx / 6u;
    let corner_idx = idx % 6u;

    // naga는 런타임 인덱스로 배열 리터럴 접근을 허용하지 않으므로 switch로 분기한다.
    let uv = corner_uv(corner_idx);

    let g = gaussians[gaussian_idx];

    // 가우시안 중심을 뷰 공간으로 이동
    let pos_view = camera.view * vec4<f32>(g.pos, 1.0);

    // 뷰 공간에서 UV 방향으로 오프셋을 더해 쿼드 꼭짓점 위치 결정
    // scale의 최대값을 반지름으로 사용 (간략화 — 정확한 2D 투영은 공분산 행렬 필요)
    let screen_offset = vec4<f32>(uv * max(g.scale.x, g.scale.y) * 0.5, 0.0, 0.0);
    let pos_proj = camera.projection * (pos_view + screen_offset);

    var output: VertexOutput;
    output.clip_pos = pos_proj;
    output.color = g.color_dc; // DC SH 계수 = 기본 색상
    output.alpha = sigmoid(g.opacity);
    output.uv = uv;

    return output;
}

// -----------------------------------------------------------------------------
// 프래그먼트 셰이더
// -----------------------------------------------------------------------------

/// 가우시안 감쇠를 적용해 최종 색상을 출력한다.
///
/// UV 기준으로 원형 마스크(dist > 1.0 버림)와
/// 가우시안 감쇠(exp(-0.5 * dist² * 4))를 적용한다.
///
/// 최종 알파 = opacity(sigmoid) × 가우시안 감쇠
/// → 알파 블렌딩(`BlendState::ALPHA_BLENDING`)과 결합해 반투명 합성된다.
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dist2 = dot(input.uv, input.uv);

    // 단위원 밖은 그리지 않아 빌보드를 원형으로 마스킹한다
    if dist2 > 1.0 {
        discard;
    }

    // 가우시안 감쇠: 중심(dist=0)에서 1, 가장자리(dist=1)에서 exp(-2) ≈ 0.135
    let weight = exp(-0.5 * dist2 * 4.0);

    return vec4<f32>(input.color, input.alpha * weight);
}
