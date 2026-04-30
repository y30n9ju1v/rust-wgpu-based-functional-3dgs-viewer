use crate::data::camera::CameraState;
use glam::Vec3;
use std::f32::consts::PI;

pub fn update_camera_angles(state: CameraState, delta_x: f32, delta_y: f32) -> CameraState {
    let new_theta = state.theta + delta_x * 0.01;
    let new_phi = (state.phi + delta_y * 0.01).clamp(0.1, PI - 0.1);

    CameraState {
        theta: new_theta,
        phi: new_phi,
        radius: state.radius,
    }
}

pub fn compute_camera_position(state: CameraState) -> Vec3 {
    let x = state.radius * state.phi.sin() * state.theta.cos();
    let y = state.radius * state.phi.cos();
    let z = state.radius * state.phi.sin() * state.theta.sin();

    Vec3::new(x, y, z)
}

pub fn camera_to_view_matrix(state: CameraState) -> glam::Mat4 {
    let position = compute_camera_position(state);
    let target = Vec3::ZERO;
    let up = Vec3::Y;

    glam::Mat4::look_at_rh(position, target, up)
}

pub fn zoom_camera(state: CameraState, delta: f32) -> CameraState {
    let new_radius = (state.radius * (-delta * 0.1).exp()).clamp(0.5, 10.0);

    CameraState {
        radius: new_radius,
        ..state
    }
}
