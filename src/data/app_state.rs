use crate::data::camera::CameraState;
use crate::data::gaussian::Gaussian;
use std::sync::Arc;

/// 앱 전체 상태를 담는 불변 값 타입.
///
/// FP 원칙에 따라 상태를 직접 변경하지 않고 `with_*` 메서드로
/// 변경된 필드만 교체한 새 상태를 반환한다.
///
/// `gaussians`는 `Arc`로 공유해 카메라 업데이트 시 수십만 개의 clone을 방지한다.
#[derive(Clone)]
pub struct AppState {
    pub camera: CameraState,
    pub gaussians: Arc<Vec<Gaussian>>,
}

impl AppState {
    pub fn new(gaussians: Vec<Gaussian>) -> Self {
        Self {
            camera: CameraState::new(),
            gaussians: Arc::new(gaussians),
        }
    }

    /// 카메라만 교체한 새 `AppState`를 반환한다.
    /// `gaussians`는 `Arc` 포인터만 복사되므로 O(1)이다.
    pub fn with_camera(&self, camera: CameraState) -> Self {
        AppState {
            camera,
            gaussians: Arc::clone(&self.gaussians),
        }
    }
}
