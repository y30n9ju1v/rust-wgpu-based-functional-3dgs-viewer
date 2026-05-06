use crate::data::gaussian::Gaussian;
use glam::Vec3;
use rayon::prelude::*;

/// 가우시안들을 카메라로부터 먼 순서(back-to-front)로 정렬한 인덱스 배열을 반환한다.
///
/// 알파 블렌딩은 뒤에서 앞 순서로 그려야 올바른 결과가 나온다.
/// rayon 병렬 정렬로 멀티코어를 활용한다.
/// 제곱 거리 비교로 sqrt 연산을 생략해 성능을 높인다.
pub fn sort_gaussians_by_depth(gaussians: &[Gaussian], camera_pos: Vec3) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();

    indices.par_sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });

    indices
}

/// 가우시안을 back-to-front 정렬한 뒤 `out` 버퍼에 GPU 업로드 형식으로 채운다.
///
/// `out`을 재사용해 매 프레임 대용량 `Vec` 할당을 방지한다.
/// `clear`는 길이만 0으로 만들고 capacity는 유지하므로, 두 번째 호출부터는
/// 재할당 없이 기존 메모리를 덮어쓴다.
pub fn prepare_sorted_indices(gaussians: &[Gaussian], camera_pos: Vec3, out: &mut Vec<u32>) {
    let indices = sort_gaussians_by_depth(gaussians, camera_pos);
    out.clear();
    out.extend(indices.into_iter().map(|i| i as u32));
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

    // --- sort_gaussians_by_depth (추가 케이스) ---

    #[test]
    fn test_sort_with_non_origin_camera() {
        // 카메라가 원점이 아닐 때도 거리 기준 정렬이 올바른지 확인
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 3.0), // 카메라(z=5)에서 거리 2
            gaussian_at(0.0, 0.0, 1.0), // 카메라(z=5)에서 거리 4
        ];
        let camera_pos = Vec3::new(0.0, 0.0, 5.0);
        let indices = sort_gaussians_by_depth(&gaussians, camera_pos);
        // index=1 (z=1)이 더 멀므로 먼저 와야 함
        assert_eq!(indices[0], 1);
        assert_eq!(indices[1], 0);
    }

    #[test]
    fn test_sort_equidistant_gaussians_returns_all() {
        // 동일 거리 가우시안이 있어도 결과 길이는 유지된다
        let gaussians = vec![
            gaussian_at(1.0, 0.0, 0.0),
            gaussian_at(-1.0, 0.0, 0.0),
            gaussian_at(0.0, 1.0, 0.0),
        ];
        let indices = sort_gaussians_by_depth(&gaussians, Vec3::ZERO);
        assert_eq!(indices.len(), 3);
    }

    // --- prepare_sorted_indices ---

    #[test]
    fn test_prepare_sorted_indices_count() {
        let gaussians = vec![
            gaussian_at(0.0, 0.0, 1.0),
            gaussian_at(0.0, 0.0, 5.0),
            gaussian_at(0.0, 0.0, 3.0),
        ];
        let mut out = Vec::new();
        prepare_sorted_indices(&gaussians, Vec3::ZERO, &mut out);
        assert_eq!(out.len(), gaussians.len());
    }

    #[test]
    fn test_prepare_sorted_indices_order() {
        // back-to-front 순서 확인: z=5가 z=1보다 먼저 와야 함
        let gaussians = vec![gaussian_at(0.0, 0.0, 1.0), gaussian_at(0.0, 0.0, 5.0)];
        let mut out = Vec::new();
        prepare_sorted_indices(&gaussians, Vec3::ZERO, &mut out);
        assert_eq!(out[0], 1); // index 1 (z=5)
        assert_eq!(out[1], 0); // index 0 (z=1)
    }

    #[test]
    fn test_prepare_sorted_indices_empty() {
        let mut out = Vec::new();
        prepare_sorted_indices(&[], Vec3::ZERO, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_prepare_sorted_indices_reuses_buffer() {
        // 재사용 시 이전 내용이 clear되고 새 데이터로 채워지는지 확인
        let gaussians = vec![gaussian_at(0.0, 0.0, 1.0), gaussian_at(0.0, 0.0, 2.0)];
        let mut out = Vec::new();
        prepare_sorted_indices(&gaussians, Vec3::ZERO, &mut out);
        assert_eq!(out.len(), 2);

        let gaussians2 = vec![gaussian_at(1.0, 0.0, 0.0)];
        prepare_sorted_indices(&gaussians2, Vec3::ZERO, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0);
    }
}
