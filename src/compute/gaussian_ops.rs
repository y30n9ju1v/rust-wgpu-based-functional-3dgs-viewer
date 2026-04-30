use crate::data::gaussian::Gaussian;
use glam::{Vec3, Quat};
use rayon::prelude::*;

pub fn quat_to_mat3(q: &[f32; 4]) -> glam::Mat3 {
    let quat = Quat::from_array(*q);
    glam::Mat3::from_quat(quat)
}

pub fn compute_covariance(gaussian: &Gaussian) -> glam::Mat3 {
    let scale = Vec3::new(gaussian.scale_0, gaussian.scale_1, gaussian.scale_2);
    let rot = quat_to_mat3(&gaussian.rotation());

    // Σ = R · S · Sᵀ · Rᵀ  (S는 대각 스케일 행렬)
    let scale_mat = glam::Mat3::from_diagonal(scale);
    let rs = rot * scale_mat;
    rs * rs.transpose()
}

pub fn sort_gaussians_by_depth(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    
    indices.sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });
    
    indices
}

pub fn sort_gaussians_by_depth_parallel(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    
    indices.par_sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });
    
    indices
}

pub fn filter_gaussians_by_lod(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
    lod_level: u32,
) -> Vec<usize> {
    let skip_rate = (1usize) << lod_level;
    gaussians
        .iter()
        .enumerate()
        .filter(|(i, g)| is_visible_at_lod(*i, g, camera_pos, skip_rate))
        .map(|(i, _)| i)
        .collect()
}

fn is_visible_at_lod(index: usize, g: &Gaussian, camera_pos: Vec3, skip_rate: usize) -> bool {
    index % skip_rate == 0 && (g.position() - camera_pos).length() < 50.0
}

pub fn transform_gaussians_batch(
    gaussians: &[Gaussian],
    view_matrix: glam::Mat4,
) -> Vec<Vec3> {
    gaussians
        .iter()
        .map(|g| {
            let pos = Vec3::new(g.x, g.y, g.z);
            (view_matrix * pos.extend(1.0)).truncate()
        })
        .collect()
}

pub fn compute_distances_batch(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<f32> {
    gaussians
        .iter()
        .map(|g| (g.position() - camera_pos).length_squared())
        .collect()
}
