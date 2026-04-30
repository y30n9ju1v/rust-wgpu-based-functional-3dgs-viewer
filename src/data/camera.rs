/// 구면 좌표계로 표현된 카메라 상태 (불변 값 타입).
///
/// 카메라는 항상 원점(0,0,0)을 바라보며 반지름 `radius` 거리의 구면 위에 위치한다.
/// `theta`(수평각)와 `phi`(수직각)로 구면 위의 위치를 결정한다.
///
/// 이 구조체는 순수 데이터이므로 메서드가 상태를 변경하지 않는다.
/// 변환은 `compute::camera_ops`의 순수 함수가 담당한다.
#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    /// 수평 회전각 (라디안) — XZ 평면에서의 방위각
    pub theta: f32,
    /// 수직 회전각 (라디안) — Y축 기준 천정각, (0, π) 범위로 클램프됨
    pub phi: f32,
    /// 원점까지의 거리
    pub radius: f32,
}

impl CameraState {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CameraState {
    fn default() -> Self {
        use std::f32::consts::PI;
        Self {
            theta: 0.0,
            phi: PI / 4.0, // 45도 — 약간 위에서 내려다보는 초기 시점
            radius: 2.0,
        }
    }
}
