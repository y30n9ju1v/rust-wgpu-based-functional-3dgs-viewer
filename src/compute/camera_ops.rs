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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_camera() -> CameraState {
        CameraState::default()
    }

    // --- update_camera_angles ---

    #[test]
    fn test_mouse_right_increases_theta() {
        let cam = default_camera();
        let result = update_camera_angles(cam, 100.0, 0.0);
        assert!(result.theta > cam.theta);
    }

    #[test]
    fn test_mouse_down_increases_phi() {
        let cam = default_camera();
        let result = update_camera_angles(cam, 0.0, 100.0);
        assert!(result.phi > cam.phi);
    }

    #[test]
    fn test_radius_unchanged_after_angle_update() {
        let cam = default_camera();
        let result = update_camera_angles(cam, 50.0, 30.0);
        assert_eq!(result.radius, cam.radius);
    }

    #[test]
    fn test_phi_clamp_upper_bound() {
        // 극단적인 아래 드래그 — phi가 π-0.1을 넘지 않아야 함
        let cam = default_camera();
        let result = update_camera_angles(cam, 0.0, 100_000.0);
        assert!(result.phi <= PI - 0.1);
    }

    #[test]
    fn test_phi_clamp_lower_bound() {
        // 극단적인 위 드래그 — phi가 0.1 아래로 내려가지 않아야 함
        let cam = default_camera();
        let result = update_camera_angles(cam, 0.0, -100_000.0);
        assert!(result.phi >= 0.1);
    }

    // --- compute_camera_position ---

    #[test]
    fn test_camera_position_distance_equals_radius() {
        let cam = default_camera();
        let pos = compute_camera_position(cam);
        // 원점으로부터의 거리가 radius와 일치해야 함
        assert!((pos.length() - cam.radius).abs() < 1e-5);
    }

    #[test]
    fn test_theta_zero_phi_half_pi_points_along_x() {
        // phi=π/2, theta=0 → 카메라가 +X 축 위에 위치
        let cam = CameraState {
            theta: 0.0,
            phi: PI / 2.0,
            radius: 1.0,
        };
        let pos = compute_camera_position(cam);
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!(pos.y.abs() < 1e-5);
        assert!(pos.z.abs() < 1e-5);
    }

    #[test]
    fn test_phi_zero_points_along_y() {
        // phi=0 → 카메라가 +Y 축 위에 위치 (천정)
        let cam = CameraState {
            theta: 0.0,
            phi: 0.0,
            radius: 3.0,
        };
        let pos = compute_camera_position(cam);
        assert!((pos.y - 3.0).abs() < 1e-5);
        assert!(pos.x.abs() < 1e-5);
        assert!(pos.z.abs() < 1e-5);
    }

    // --- camera_to_view_matrix ---

    #[test]
    fn test_view_matrix_transforms_origin_to_nonzero() {
        // 뷰 행렬은 월드 원점을 카메라 로컬 공간으로 이동시켜야 함
        let cam = default_camera();
        let view = camera_to_view_matrix(cam);
        let origin_in_view = view * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        // 원점이 카메라 앞쪽 (-z)에 있어야 함 (오른손 좌표계)
        assert!(origin_in_view.z < 0.0);
    }

    // --- zoom_camera ---

    #[test]
    fn test_scroll_up_zooms_in() {
        let cam = default_camera();
        let zoomed = zoom_camera(cam, 1.0); // 위 스크롤 = 양수 delta
        assert!(zoomed.radius < cam.radius);
    }

    #[test]
    fn test_scroll_down_zooms_out() {
        let cam = default_camera();
        let zoomed = zoom_camera(cam, -1.0);
        assert!(zoomed.radius > cam.radius);
    }

    #[test]
    fn test_zoom_clamp_min() {
        let cam = default_camera();
        let zoomed = zoom_camera(cam, 1_000.0);
        assert!(zoomed.radius >= 0.5);
    }

    #[test]
    fn test_zoom_clamp_max() {
        let cam = default_camera();
        let zoomed = zoom_camera(cam, -1_000.0);
        assert!(zoomed.radius <= 10.0);
    }

    #[test]
    fn test_zoom_preserves_angles() {
        let cam = default_camera();
        let zoomed = zoom_camera(cam, 1.0);
        assert_eq!(zoomed.theta, cam.theta);
        assert_eq!(zoomed.phi, cam.phi);
    }
}
