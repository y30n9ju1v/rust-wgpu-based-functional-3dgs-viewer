#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    pub theta: f32,
    pub phi: f32,
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
            phi: PI / 4.0,
            radius: 2.0,
        }
    }
}
