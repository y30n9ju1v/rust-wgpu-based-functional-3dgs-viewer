use crate::data::gaussian::Gaussian;
use wgpu::util::DeviceExt;
use wgpu::*;

/// GPU 업로드 전용 Gaussian 구조체.
///
/// WGSL `struct Gaussian`과 메모리 레이아웃이 1:1로 일치해야 한다.
/// WGSL은 vec3<f32> 뒤에 4바이트 패딩을 자동 삽입하므로,
/// Rust 측에서도 `_pad` 필드로 명시적으로 맞춰준다.
///
/// 레이아웃 (모든 필드는 16바이트 경계에 배치):
/// - pos(12) + opacity(4) = 16
/// - color_dc(12) + _pad0(4) = 16
/// - scale(12) + _pad1(4) = 16
/// - rot(16) = 16
/// - f_rest(180) + _pad2(12) = 192 (12 vec4)
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GaussianGpu {
    pub pos: [f32; 3],
    pub opacity: f32,
    pub color_dc: [f32; 3],
    pub _pad0: f32,
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rot: [f32; 4],
    pub f_rest: [f32; 45],
    pub _pad2: [f32; 3], // 45 → 48개로 맞춰 WGSL array<vec4<f32>, 12>에 대응
}

/// CPU 측 `Gaussian`을 GPU 업로드용 `GaussianGpu`로 변환한다.
impl From<&Gaussian> for GaussianGpu {
    fn from(g: &Gaussian) -> Self {
        let mut f_rest = [0.0f32; 45];
        f_rest.copy_from_slice(&g.f_rest);
        Self {
            pos: [g.x, g.y, g.z],
            opacity: g.opacity,
            color_dc: [g.f_dc_0, g.f_dc_1, g.f_dc_2],
            _pad0: 0.0,
            scale: [g.scale_0, g.scale_1, g.scale_2],
            _pad1: 0.0,
            rot: [g.rot_0, g.rot_1, g.rot_2, g.rot_3],
            f_rest,
            _pad2: [0.0; 3],
        }
    }
}

/// GPU Storage Buffer에 가우시안 데이터를 초기 업로드한다.
///
/// `COPY_DST`를 추가해 이후 `queue.write_buffer`로 내용을 갱신할 수 있다.
pub fn create_gaussian_buffer(device: &Device, gaussians: &[Gaussian]) -> Buffer {
    let gpu_gaussians: Vec<GaussianGpu> = gaussians.iter().map(|g| g.into()).collect();

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gaussian Buffer"),
        contents: bytemuck::cast_slice(&gpu_gaussians),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// 셰이더에 전달되는 카메라 uniform 데이터.
///
/// WGSL `struct CameraUniform { view: mat4x4<f32>; projection: mat4x4<f32>; }`와
/// 레이아웃이 일치해야 한다. glam의 `to_cols_array()`는 column-major 순서로
/// WGSL의 mat4x4 메모리 순서와 동일하다.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct CameraUniform {
    pub view: [f32; 16],
    pub projection: [f32; 16],
}

/// GPU Uniform Buffer에 카메라 행렬을 초기 업로드한다.
pub fn create_camera_buffer(
    device: &Device,
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
) -> Buffer {
    let camera = CameraUniform {
        view: view_matrix.to_cols_array(),
        projection: proj_matrix.to_cols_array(),
    };

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

/// WGSL 소스 문자열을 컴파일해 GPU ShaderModule을 생성한다.
pub fn create_shader_module(device: &Device, source: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    })
}

/// GPU 버퍼의 내용을 새 데이터로 덮어쓴다 (offset=0부터 전체 교체).
///
/// `T`는 `bytemuck::Pod`를 구현해야 한다 — 임의의 바이트 패턴이 유효한 타입이어야 함.
pub fn update_buffer<T: bytemuck::Pod>(queue: &Queue, buffer: &Buffer, data: &[T]) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}
