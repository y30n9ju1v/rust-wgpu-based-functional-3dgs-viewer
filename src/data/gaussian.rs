use glam::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
pub struct Gaussian {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub f_dc_0: f32,
    pub f_dc_1: f32,
    pub f_dc_2: f32,
    pub f_rest: [f32; 45],
    pub opacity: f32,
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
}

impl Gaussian {
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    
    pub fn color(&self) -> Vec3 {
        Vec3::new(self.f_dc_0, self.f_dc_1, self.f_dc_2)
    }
    
    pub fn scale(&self) -> Vec3 {
        Vec3::new(self.scale_0, self.scale_1, self.scale_2)
    }
    
    pub fn rotation(&self) -> [f32; 4] {
        [self.rot_0, self.rot_1, self.rot_2, self.rot_3]
    }
}
