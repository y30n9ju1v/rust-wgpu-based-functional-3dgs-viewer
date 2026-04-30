use wgpu::*;
use wgpu::util::DeviceExt;
use crate::data::gaussian::Gaussian;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GaussianGpu {
    pub pos: [f32; 3],
    pub opacity: f32,
    pub color_dc: [f32; 3],
    pub _pad0: f32,
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rot: [f32; 4],
    pub f_rest: [f32; 45],
    pub _pad2: [f32; 3],
}

impl From<&Gaussian> for GaussianGpu {
    fn from(g: &Gaussian) -> Self {
        let mut f_rest = [0.0f32; 45];
        f_rest.copy_from_slice(&g.f_rest);
        Self {
            pos: [g.x, g.y, g.z],
            opacity: g.opacity,
            color_dc: [g.f_dc_0, g.f_dc_1, g.f_dc_2],
            _pad0: 0.0,
            scale: [g.scale_0, g.scale_1, g.scale_2],
            _pad1: 0.0,
            rot: [g.rot_0, g.rot_1, g.rot_2, g.rot_3],
            f_rest,
            _pad2: [0.0; 3],
        }
    }
}

pub fn create_gaussian_buffer(
    device: &Device,
    gaussians: &[Gaussian],
) -> Buffer {
    let gpu_gaussians: Vec<GaussianGpu> = gaussians.iter().map(|g| g.into()).collect();
    
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gaussian Buffer"),
        contents: bytemuck::cast_slice(&gpu_gaussians),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct CameraUniform {
    pub view: [f32; 16],
    pub projection: [f32; 16],
}

pub fn create_camera_buffer(
    device: &Device,
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
) -> Buffer {
    let camera = CameraUniform {
        view: view_matrix.to_cols_array(),
        projection: proj_matrix.to_cols_array(),
    };
    
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

pub fn create_shader_module(
    device: &Device,
    source: &str,
) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    })
}

pub fn update_buffer<T: bytemuck::Pod>(
    queue: &Queue,
    buffer: &Buffer,
    data: &[T],
) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}
