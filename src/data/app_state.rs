use crate::data::camera::CameraState;
use crate::data::gaussian::Gaussian;

#[derive(Clone)]
pub struct AppState {
    pub camera: CameraState,
    pub gaussians: Vec<Gaussian>,
}

impl AppState {
    pub fn new(gaussians: Vec<Gaussian>) -> Self {
        Self {
            camera: CameraState::new(),
            gaussians,
        }
    }

    pub fn with_camera(&self, camera: CameraState) -> Self {
        AppState {
            camera,
            gaussians: self.gaussians.clone(),
        }
    }
}
