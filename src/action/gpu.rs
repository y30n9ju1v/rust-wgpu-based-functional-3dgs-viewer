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
    pub camera_pos: [f32; 4],
    pub viewport: [f32; 4],
}

/// 카메라 상태로부터 `CameraUniform`을 조립한다.
///
/// 순수 함수 — GPU 리소스 없이 호출 가능하며 `update_buffer`와 분리된다.
pub fn build_camera_uniform(
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
    camera_pos: glam::Vec3,
    viewport_size: [f32; 2],
) -> CameraUniform {
    CameraUniform {
        view: view_matrix.to_cols_array(),
        projection: proj_matrix.to_cols_array(),
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
        viewport: [viewport_size[0], viewport_size[1], 0.0, 0.0],
    }
}

/// GPU Uniform Buffer에 카메라 행렬을 초기 업로드한다.
pub fn create_camera_buffer(
    device: &Device,
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
    camera_pos: glam::Vec3,
    viewport_size: [f32; 2],
) -> Buffer {
    let camera = build_camera_uniform(view_matrix, proj_matrix, camera_pos, viewport_size);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_uniform() -> CameraUniform {
        build_camera_uniform(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
            [800.0, 600.0],
        )
    }

    #[test]
    fn test_build_camera_uniform_view_is_identity() {
        let u = identity_uniform();
        assert_eq!(u.view, glam::Mat4::IDENTITY.to_cols_array());
    }

    #[test]
    fn test_build_camera_uniform_projection_is_identity() {
        let u = identity_uniform();
        assert_eq!(u.projection, glam::Mat4::IDENTITY.to_cols_array());
    }

    #[test]
    fn test_build_camera_uniform_camera_pos_w_is_one() {
        // camera_pos는 항상 w=1.0인 동차 좌표로 패킹된다
        let u = build_camera_uniform(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::new(1.0, 2.0, 3.0),
            [800.0, 600.0],
        );
        assert!((u.camera_pos[0] - 1.0).abs() < 1e-6);
        assert!((u.camera_pos[1] - 2.0).abs() < 1e-6);
        assert!((u.camera_pos[2] - 3.0).abs() < 1e-6);
        assert!((u.camera_pos[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_camera_uniform_viewport_xy() {
        let u = identity_uniform();
        assert!((u.viewport[0] - 800.0).abs() < 1e-6);
        assert!((u.viewport[1] - 600.0).abs() < 1e-6);
        // zw는 미사용 패딩이므로 0이어야 함
        assert_eq!(u.viewport[2], 0.0);
        assert_eq!(u.viewport[3], 0.0);
    }

    #[test]
    fn test_build_camera_uniform_view_reflects_translation() {
        // 뷰 행렬로 평행이동 행렬을 넘기면 uniform에 그대로 반영된다
        let t = glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0));
        let u = build_camera_uniform(t, glam::Mat4::IDENTITY, glam::Vec3::ZERO, [1.0, 1.0]);
        assert_eq!(u.view, t.to_cols_array());
    }
}
