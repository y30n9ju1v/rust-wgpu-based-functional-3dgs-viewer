use std::sync::Arc;
use wgpu::*;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub mod action;
pub mod compute;
pub mod data;

use action::gpu::GaussianGpu;
use action::{gpu, io, render};
use compute::camera_ops;
use data::app_state::AppState;
use data::gaussian::Gaussian;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

enum InputEvent {
    MouseMove(f32, f32),
    Zoom(f32),
}

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

// ---------------------------------------------------------------------------
// AppContext
// ---------------------------------------------------------------------------

impl AppContext {
    async fn new(window: Arc<winit::window::Window>, gaussians: Vec<Gaussian>) -> Self {
        let (device, queue, surface, config) = init_gpu(window).await;
        let state = AppState::new(gaussians);
        let proj_matrix = make_proj_matrix(&config);
        let view_matrix = camera_ops::camera_to_view_matrix(state.camera);

        let gaussian_buffer = gpu::create_gaussian_buffer(&device, &state.gaussians);
        let camera_buffer = gpu::create_camera_buffer(&device, view_matrix, proj_matrix);
        let shader = gpu::create_shader_module(&device, include_str!("shaders/render.wgsl"));

        let bind_group_layout = make_bind_group_layout(&device);
        let bind_group = make_bind_group(
            &device,
            &bind_group_layout,
            &camera_buffer,
            &gaussian_buffer,
        );
        let pipeline = make_render_pipeline(&device, &shader, &bind_group_layout, config.format);

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
        self.state.camera = apply_input(self.state.camera, input);

        let view_matrix = camera_ops::camera_to_view_matrix(self.state.camera);
        let camera_data = gpu::CameraUniform {
            view: view_matrix.to_cols_array(),
            projection: self.proj_matrix.to_cols_array(),
        };
        gpu::update_buffer(&self.queue, &self.camera_buffer, &[camera_data]);
    }

    fn render(&self) {
        self.upload_sorted_gaussians();

        let Ok(output) = self.surface.get_current_texture() else {
            return;
        };
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

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

    fn upload_sorted_gaussians(&self) {
        let camera_pos = camera_ops::compute_camera_position(self.state.camera);
        let sorted: Vec<GaussianGpu> = compute::gaussian_ops::sort_gaussians_by_depth_parallel(
            &self.state.gaussians,
            camera_pos,
        )
        .iter()
        .map(|&i| GaussianGpu::from(&self.state.gaussians[i]))
        .collect();
        gpu::update_buffer(&self.queue, &self.gaussian_buffer, &sorted);
    }
}

// ---------------------------------------------------------------------------
// GPU initialisation helpers
// ---------------------------------------------------------------------------

async fn init_gpu(
    window: Arc<winit::window::Window>,
) -> (Device, Queue, Surface<'static>, SurfaceConfiguration) {
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
        .expect("Failed to find adapter");

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");

    let config = make_surface_config(&surface, &adapter, &window);
    surface.configure(&device, &config);

    (device, queue, surface, config)
}

fn make_surface_config(
    surface: &Surface,
    adapter: &Adapter,
    window: &winit::window::Window,
) -> SurfaceConfiguration {
    let size = window.inner_size();
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_capabilities(adapter).formats[0],
        width: size.width,
        height: size.height,
        present_mode: PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![],
    }
}

fn make_proj_matrix(config: &SurfaceConfiguration) -> glam::Mat4 {
    glam::Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        config.width as f32 / config.height as f32,
        0.1,
        100.0,
    )
}

fn make_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Main"),
        entries: &[
            camera_binding_layout_entry(),
            gaussian_binding_layout_entry(),
        ],
    })
}

fn camera_binding_layout_entry() -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn gaussian_binding_layout_entry() -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    camera_buffer: &Buffer,
    gaussian_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Main"),
        layout,
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
    })
}

fn make_render_pipeline(
    device: &Device,
    shader: &ShaderModule,
    bind_group_layout: &BindGroupLayout,
    format: TextureFormat,
) -> RenderPipeline {
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Render"),
        layout: Some(&layout),
        vertex: VertexState {
            module: shader,
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
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: PipelineCompilationOptions::default(),
        }),
        multiview: None,
    })
}

// ---------------------------------------------------------------------------
// Pure input helpers
// ---------------------------------------------------------------------------

fn apply_input(camera: data::camera::CameraState, input: InputEvent) -> data::camera::CameraState {
    match input {
        InputEvent::MouseMove(dx, dy) => camera_ops::update_camera_angles(camera, dx, dy),
        InputEvent::Zoom(delta) => camera_ops::zoom_camera(camera, delta),
    }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

fn handle_mouse_input(
    state: winit::event::ElementState,
    button: winit::event::MouseButton,
    mouse_pressed: &mut bool,
    last_mouse_pos: &mut Option<(f32, f32)>,
) {
    if button != winit::event::MouseButton::Left {
        return;
    }
    *mouse_pressed = state == winit::event::ElementState::Pressed;
    if !*mouse_pressed {
        *last_mouse_pos = None;
    }
}

fn handle_cursor_moved(
    position: winit::dpi::PhysicalPosition<f64>,
    mouse_pressed: bool,
    last_mouse_pos: &mut Option<(f32, f32)>,
    app: &mut AppContext,
) {
    let cur = (position.x as f32, position.y as f32);
    if mouse_pressed {
        if let Some((lx, ly)) = *last_mouse_pos {
            app.update(InputEvent::MouseMove(cur.0 - lx, cur.1 - ly));
        }
    }
    *last_mouse_pos = Some(cur);
}

fn handle_mouse_wheel(delta: winit::event::MouseScrollDelta, app: &mut AppContext) {
    let zoom = match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
    };
    app.update(InputEvent::Zoom(zoom));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("3DGS Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0))
            .build(&event_loop)
            .unwrap(),
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
        handle_event(
            event,
            elwt,
            &window,
            &mut app,
            &mut mouse_pressed,
            &mut last_mouse_pos,
        );
        elwt.set_control_flow(ControlFlow::Poll);
    });
}

fn handle_event(
    event: Event<()>,
    elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    window: &Arc<winit::window::Window>,
    app: &mut AppContext,
    mouse_pressed: &mut bool,
    last_mouse_pos: &mut Option<(f32, f32)>,
) {
    match event {
        Event::AboutToWait => window.request_redraw(),
        Event::WindowEvent { event, .. } => {
            handle_window_event(event, elwt, app, mouse_pressed, last_mouse_pos);
        }
        _ => {}
    }
}

fn handle_window_event(
    event: WindowEvent,
    elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    app: &mut AppContext,
    mouse_pressed: &mut bool,
    last_mouse_pos: &mut Option<(f32, f32)>,
) {
    match event {
        WindowEvent::RedrawRequested => app.render(),
        WindowEvent::CloseRequested => elwt.exit(),
        WindowEvent::MouseInput { state, button, .. } => {
            handle_mouse_input(state, button, mouse_pressed, last_mouse_pos);
        }
        WindowEvent::CursorMoved { position, .. } => {
            handle_cursor_moved(position, *mouse_pressed, last_mouse_pos, app);
        }
        WindowEvent::MouseWheel { delta, .. } => handle_mouse_wheel(delta, app),
        _ => {}
    }
}
