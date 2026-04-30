use crate::data::camera::CameraState;
use glam::Vec3;
use std::f32::consts::PI;

/// 마우스 드래그 델타를 카메라 각도 변화로 변환한다.
///
/// - `delta_x`: 화면 오른쪽이 양수, theta(수평각) 증가
/// - `delta_y`: 화면 아래쪽이 양수, phi(수직각) 증가
/// - phi는 (0.1, π-0.1) 범위로 클램프해 카메라가 천정/바닥을 뒤집지 않도록 막는다.
pub fn update_camera_angles(state: CameraState, delta_x: f32, delta_y: f32) -> CameraState {
    let new_theta = state.theta + delta_x * 0.01;
    let new_phi = (state.phi + delta_y * 0.01).clamp(0.1, PI - 0.1);

    CameraState {
        theta: new_theta,
        phi: new_phi,
        radius: state.radius,
    }
}

/// 구면 좌표 (theta, phi, radius)를 데카르트 좌표 Vec3로 변환한다.
///
/// 수식: x = r·sin(φ)·cos(θ),  y = r·cos(φ),  z = r·sin(φ)·sin(θ)
pub fn compute_camera_position(state: CameraState) -> Vec3 {
    let x = state.radius * state.phi.sin() * state.theta.cos();
    let y = state.radius * state.phi.cos();
    let z = state.radius * state.phi.sin() * state.theta.sin();

    Vec3::new(x, y, z)
}

/// 카메라 상태를 오른손 좌표계 뷰 행렬(look-at)로 변환한다.
///
/// 항상 원점(0,0,0)을 바라보며 Y축이 위쪽이다.
pub fn camera_to_view_matrix(state: CameraState) -> glam::Mat4 {
    let position = compute_camera_position(state);
    let target = Vec3::ZERO;
    let up = Vec3::Y;

    glam::Mat4::look_at_rh(position, target, up)
}

/// 스크롤 델타로 카메라 반지름(줌)을 조정한다.
///
/// 지수 스케일을 사용해 가까울수록 세밀하게, 멀수록 크게 이동한다.
/// radius는 (0.5, 10.0) 범위로 클램프된다.
///
/// `delta`는 winit의 `LineDelta` y 값 — 위로 스크롤하면 양수.
pub fn zoom_camera(state: CameraState, delta: f32) -> CameraState {
    let new_radius = (state.radius * (-delta * 0.1).exp()).clamp(0.5, 10.0);

    CameraState {
        radius: new_radius,
        ..state
    }
}
