use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use wgpu::*;

pub mod data;
pub mod compute;
pub mod action;

use data::app_state::AppState;
use data::gaussian::Gaussian;
use compute::camera_ops;
use action::{io, gpu, render};
use action::gpu::GaussianGpu;

struct AppContext {
    state: AppState,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    #[allow(dead_code)] // TODO: Resize 이벤트 처리 시 proj_matrix 갱신에 사용
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    gaussian_buffer: Buffer,
    camera_buffer: Buffer,
    proj_matrix: glam::Mat4,
}

impl AppContext {
    async fn new(window: Arc<winit::window::Window>, gaussians: Vec<Gaussian>) -> Self {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Failed to create surface");
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to request adapter");
        
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .expect("Failed to request device");
        
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        
        surface.configure(&device, &config);
        
        let state = AppState::new(gaussians);

        let view_matrix = camera_ops::camera_to_view_matrix(state.camera);
        let proj_matrix = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            config.width as f32 / config.height as f32,
            0.1,
            100.0,
        );
        
        let gaussian_buffer = gpu::create_gaussian_buffer(&device, &state.gaussians);
        let camera_buffer = gpu::create_camera_buffer(&device, view_matrix, proj_matrix);
        let shader = gpu::create_shader_module(&device, include_str!("shaders/render.wgsl"));
        
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Main"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Main"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: gaussian_buffer.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            multiview: None,
        });
        
        Self {
            state,
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group,
            gaussian_buffer,
            camera_buffer,
            proj_matrix,
        }
    }

    fn update(&mut self, input: InputEvent) {
        let new_camera = match input {
            InputEvent::MouseMove(dx, dy) => {
                camera_ops::update_camera_angles(self.state.camera, dx, dy)
            }
            InputEvent::Zoom(delta) => {
                camera_ops::zoom_camera(self.state.camera, delta)
            }
        };

        self.state.camera = new_camera;

        let view_matrix = camera_ops::camera_to_view_matrix(self.state.camera);
        let camera_data = gpu::CameraUniform {
            view: view_matrix.to_cols_array(),
            projection: self.proj_matrix.to_cols_array(),
        };

        gpu::update_buffer(&self.queue, &self.camera_buffer, &[camera_data]);
    }
    
    fn render(&self) {
        // 알파 블렌딩을 위해 back-to-front 정렬
        let camera_pos = compute::camera_ops::compute_camera_position(self.state.camera);
        let sorted_indices = compute::gaussian_ops::sort_gaussians_by_depth_parallel(
            &self.state.gaussians,
            camera_pos,
        );
        let sorted_gpu: Vec<GaussianGpu> = sorted_indices
            .iter()
            .map(|&i| GaussianGpu::from(&self.state.gaussians[i]))
            .collect();
        gpu::update_buffer(&self.queue, &self.gaussian_buffer, &sorted_gpu);

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(_) => return,
        };

        let view = output.texture.create_view(&TextureViewDescriptor::default());

        render::render_frame(
            &self.device,
            &self.queue,
            &view,
            &self.pipeline,
            &self.bind_group,
            self.state.gaussians.len() as u32,
        );

        output.present();
    }
}

enum InputEvent {
    MouseMove(f32, f32),
    Zoom(f32),
}

fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("3DGS Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0))
            .build(&event_loop)
            .unwrap()
    );
    
    let gaussians = io::load_ply_file("assets/example.ply").unwrap_or_else(|e| {
        log::warn!("Failed to load assets/example.ply: {e}");
        Vec::new()
    });

    log::info!("Loaded {} gaussians", gaussians.len());
    
    let mut app = pollster::block_on(AppContext::new(Arc::clone(&window), gaussians));
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f32, f32)> = None;

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested => {
                    app.render();
                }
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == winit::event::MouseButton::Left {
                        mouse_pressed = state == winit::event::ElementState::Pressed;
                        if !mouse_pressed {
                            last_mouse_pos = None;
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let cur = (position.x as f32, position.y as f32);
                    if mouse_pressed {
                        if let Some((lx, ly)) = last_mouse_pos {
                            app.update(InputEvent::MouseMove(cur.0 - lx, cur.1 - ly));
                        }
                    }
                    last_mouse_pos = Some(cur);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            app.update(InputEvent::Zoom(y));
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            app.update(InputEvent::Zoom(pos.y as f32 * 0.01));
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        elwt.set_control_flow(ControlFlow::Poll);
    });
}
