use crate::data::gaussian::Gaussian;
use glam::{Quat, Vec3};
use rayon::prelude::*;

/// 쿼터니언 배열 [w, x, y, z]를 3×3 회전 행렬로 변환한다.
pub fn quat_to_mat3(q: &[f32; 4]) -> glam::Mat3 {
    let quat = Quat::from_array(*q);
    glam::Mat3::from_quat(quat)
}

/// 3DGS 공분산 행렬 Σ = R·S·Sᵀ·Rᵀ 를 계산한다.
///
/// S는 scale을 대각 원소로 하는 행렬, R은 rotation 쿼터니언에서 얻은 회전 행렬.
/// 결과는 타원체 모양을 결정하는 3×3 대칭 행렬이다.
pub fn compute_covariance(gaussian: &Gaussian) -> glam::Mat3 {
    let scale = Vec3::new(gaussian.scale_0, gaussian.scale_1, gaussian.scale_2);
    let rot = quat_to_mat3(&gaussian.rotation());

    // Σ = R · S · Sᵀ · Rᵀ  (S는 대각 스케일 행렬)
    let scale_mat = glam::Mat3::from_diagonal(scale);
    let rs = rot * scale_mat;
    rs * rs.transpose()
}

/// 가우시안들을 카메라로부터 먼 순서(back-to-front)로 정렬한 인덱스 배열을 반환한다.
///
/// 알파 블렌딩은 뒤에서 앞 순서로 그려야 올바른 결과가 나온다.
/// 단일 스레드 버전 — 가우시안 수가 적을 때 적합하다.
pub fn sort_gaussians_by_depth(gaussians: &[Gaussian], camera_pos: Vec3) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();

    // 제곱 거리 비교로 sqrt 연산을 생략해 성능을 높인다
    indices.sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });

    indices
}

/// `sort_gaussians_by_depth`의 rayon 병렬 버전.
///
/// 수십만 개 이상의 가우시안에서 멀티코어를 활용해 정렬 속도를 높인다.
pub fn sort_gaussians_by_depth_parallel(gaussians: &[Gaussian], camera_pos: Vec3) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();

    indices.par_sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });

    indices
}

/// LOD(Level of Detail) 수준에 따라 렌더링할 가우시안의 인덱스를 필터링한다.
///
/// `lod_level`이 0이면 모든 가우시안, 1이면 1/2, 2이면 1/4 씩 선택한다.
/// 추가로 카메라로부터 50.0 이상 떨어진 가우시안은 제외한다.
pub fn filter_gaussians_by_lod(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
    lod_level: u32,
) -> Vec<usize> {
    // lod_level=0 → skip_rate=1(전부), lod_level=1 → 2(절반), lod_level=2 → 4(1/4)
    let skip_rate = (1usize) << lod_level;
    gaussians
        .iter()
        .enumerate()
        .filter(|(i, g)| is_visible_at_lod(*i, g, camera_pos, skip_rate))
        .map(|(i, _)| i)
        .collect()
}

/// LOD와 거리 조건을 모두 만족하는지 확인하는 술어 함수.
fn is_visible_at_lod(index: usize, g: &Gaussian, camera_pos: Vec3, skip_rate: usize) -> bool {
    index.is_multiple_of(skip_rate) && (g.position() - camera_pos).length() < 50.0
}

/// 가우시안 위치 전체를 뷰 공간으로 일괄 변환한다.
///
/// 컴파일러가 루프를 SIMD로 자동 벡터화할 수 있도록 순수 iterator 형태로 작성됐다.
pub fn transform_gaussians_batch(gaussians: &[Gaussian], view_matrix: glam::Mat4) -> Vec<Vec3> {
    gaussians
        .iter()
        .map(|g| {
            let pos = Vec3::new(g.x, g.y, g.z);
            // w=1.0 을 추가해 위치 벡터로 변환 후 다시 Vec3로 잘라낸다
            (view_matrix * pos.extend(1.0)).truncate()
        })
        .collect()
}

/// 가우시안 전체의 카메라까지의 제곱 거리를 일괄 계산한다.
///
/// 제곱 거리를 사용해 sqrt 비용을 절약한다 (순위 비교에만 사용 시 충분).
pub fn compute_distances_batch(gaussians: &[Gaussian], camera_pos: Vec3) -> Vec<f32> {
    gaussians
        .iter()
        .map(|g| (g.position() - camera_pos).length_squared())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    /// 지정한 위치만 설정하고 나머지는 0으로 채운 Gaussian을 생성한다.
    fn gaussian_at(x: f32, y: f32, z: f32) -> Gaussian {
        Gaussian {
            x,
            y,
            z,
            ..Gaussian::zeroed()
        }
    }

    // --- quat_to_mat3 ---

    #[test]
    fn test_identity_quat_gives_identity_mat3() {
        // glam::Quat::from_array는 [x, y, z, w] 순서
        // 단위 쿼터니언: x=y=z=0, w=1
        let mat = quat_to_mat3(&[0.0, 0.0, 0.0, 1.0]);
        assert!((mat - glam::Mat3::IDENTITY).abs_diff_eq(glam::Mat3::ZERO, 1e-5));
    }

    // --- compute_covariance ---

    #[test]
    fn test_covariance_is_symmetric() {
        // 공분산 행렬은 항상 대칭(Σ = Σᵀ)이어야 함
        let g = Gaussian {
            scale_0: 1.0,
            scale_1: 2.0,
            scale_2: 0.5,
            rot_0: 1.0, // 단위 쿼터니언
            ..Gaussian::zeroed()
        };
        let cov = compute_covariance(&g);
        assert!((cov - cov.transpose()).abs_diff_eq(glam::Mat3::ZERO, 1e-5));
    }

    #[test]
    fn test_covariance_identity_rotation_equals_scale_squared() {
        // 회전 없이(단위 쿼터니언) scale=(s0,s1,s2)이면 Σ = diag(s0², s1², s2²)
        let g = Gaussian {
            scale_0: 2.0,
            scale_1: 3.0,
            scale_2: 4.0,
            rot_0: 1.0, // 단위 쿼터니언 w=1
            ..Gaussian::zeroed()
        };
        let cov = compute_covariance(&g);
        assert!((cov.x_axis.x - 4.0).abs() < 1e-5); // 2² = 4
        assert!((cov.y_axis.y - 9.0).abs() < 1e-5); // 3² = 9
        assert!((cov.z_axis.z - 16.0).abs() < 1e-5); // 4² = 16
    }

    // --- sort_gaussians_by_depth ---

    #[test]
    fn test_sort_back_to_front_order() {
        // 카메라가 원점에 있을 때, z=10이 z=1보다 더 멀므로 앞에 와야 함
        let gaussians = vec![gaussian_at(0.0, 0.0, 1.0), gaussian_at(0.0, 0.0, 10.0)];
        let indices = sort_gaussians_by_depth(&gaussians, Vec3::ZERO);
        assert_eq!(indices[0], 1); // 더 먼 것(index=1, z=10)이 먼저
        assert_eq!(indices[1], 0);
    }

    #[test]
    fn test_sort_single_gaussian() {
        let gaussians = vec![gaussian_at(1.0, 2.0, 3.0)];
        let indices = sort_gaussians_by_depth(&gaussians, Vec3::ZERO);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_sort_empty_gaussians() {
        let indices = sort_gaussians_by_depth(&[], Vec3::ZERO);
        assert!(indices.is_empty());
    }

    // --- sort_gaussians_by_depth_parallel ---

    #[test]
    fn test_parallel_sort_same_as_sequential() {
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 5.0),
            gaussian_at(0.0, 0.0, 1.0),
            gaussian_at(0.0, 0.0, 8.0),
        ];
        let camera_pos = Vec3::ZERO;
        let seq = sort_gaussians_by_depth(&gaussians, camera_pos);
        let par = sort_gaussians_by_depth_parallel(&gaussians, camera_pos);
        assert_eq!(seq, par);
    }

    // --- filter_gaussians_by_lod ---

    #[test]
    fn test_lod_0_returns_all_within_range() {
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 1.0),
            gaussian_at(0.0, 0.0, 2.0),
            gaussian_at(0.0, 0.0, 3.0),
        ];
        let indices = filter_gaussians_by_lod(&gaussians, Vec3::ZERO, 0);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_lod_1_returns_every_other() {
        // lod_level=1 → skip_rate=2 → 짝수 인덱스만
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 1.0), // index 0 → 포함
            gaussian_at(0.0, 0.0, 2.0), // index 1 → 제외
            gaussian_at(0.0, 0.0, 3.0), // index 2 → 포함
            gaussian_at(0.0, 0.0, 4.0), // index 3 → 제외
        ];
        let indices = filter_gaussians_by_lod(&gaussians, Vec3::ZERO, 1);
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_lod_excludes_far_gaussians() {
        // 거리 50.0 이상은 제외
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 1.0),  // 가까움 → 포함
            gaussian_at(0.0, 0.0, 60.0), // 멀음 → 제외
        ];
        let indices = filter_gaussians_by_lod(&gaussians, Vec3::ZERO, 0);
        assert_eq!(indices, vec![0]);
    }

    // --- transform_gaussians_batch ---

    #[test]
    fn test_transform_with_identity_unchanged() {
        let gaussians = vec![gaussian_at(1.0, 2.0, 3.0)];
        let result = transform_gaussians_batch(&gaussians, glam::Mat4::IDENTITY);
        assert!((result[0] - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn test_transform_translation() {
        let gaussians = vec![gaussian_at(0.0, 0.0, 0.0)];
        let t = glam::Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let result = transform_gaussians_batch(&gaussians, t);
        assert!((result[0] - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }

    // --- compute_distances_batch ---

    #[test]
    fn test_distances_batch_values() {
        let gaussians = vec![
            gaussian_at(3.0, 0.0, 0.0), // 원점에서 거리 3 → 제곱 9
            gaussian_at(0.0, 4.0, 0.0), // 원점에서 거리 4 → 제곱 16
        ];
        let dists = compute_distances_batch(&gaussians, Vec3::ZERO);
        assert!((dists[0] - 9.0).abs() < 1e-5);
        assert!((dists[1] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_distances_batch_empty() {
        let dists = compute_distances_batch(&[], Vec3::ZERO);
        assert!(dists.is_empty());
    }
}
