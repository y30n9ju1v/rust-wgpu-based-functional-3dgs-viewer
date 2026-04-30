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

use action::{gpu, io, render};
use compute::camera_ops;
use data::app_state::AppState;
use data::gaussian::Gaussian;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// 사용자 입력을 추상화한 열거형.
///
/// winit 이벤트를 직접 전달하는 대신 이 타입으로 변환해
/// `AppContext::update`가 winit에 직접 의존하지 않도록 분리한다.
enum InputEvent {
    /// 마우스 드래그 델타 (dx, dy) — 픽셀 단위
    MouseMove(f32, f32),
    /// 스크롤 줌 델타 — 양수이면 줌인
    Zoom(f32),
}

/// GPU 리소스와 앱 상태를 함께 보유하는 컨텍스트.
///
/// GPU 리소스(device, queue, surface, pipeline 등)는 Side Effect를 일으키므로
/// `action` 레이어에서 생성하고 여기에 보관한다.
/// 순수 앱 상태(`state`)는 `data` 레이어 타입을 사용한다.
struct AppContext {
    state: AppState,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    #[allow(dead_code)] // TODO: Resize 이벤트 처리 시 proj_matrix 갱신에 사용
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    /// 가우시안 데이터를 담는 GPU Storage Buffer — 매 프레임 깊이 정렬 후 갱신됨
    gaussian_buffer: Buffer,
    /// 카메라 뷰/프로젝션 행렬을 담는 GPU Uniform Buffer
    camera_buffer: Buffer,
    /// 창 크기가 바뀌지 않는 한 고정되는 투영 행렬 (캐싱)
    proj_matrix: glam::Mat4,
}

// ---------------------------------------------------------------------------
// AppContext
// ---------------------------------------------------------------------------

impl AppContext {
    /// GPU를 초기화하고 모든 리소스를 생성해 `AppContext`를 반환한다.
    ///
    /// `pollster::block_on`으로 async를 블로킹 실행한다.
    /// winit 이벤트 루프가 메인 스레드를 점유하므로 tokio와 함께 쓸 수 없다.
    async fn new(window: Arc<winit::window::Window>, gaussians: Vec<Gaussian>) -> Self {
        let (device, queue, surface, config) = init_gpu(window).await;
        let state = AppState::new(gaussians);
        let proj_matrix = make_proj_matrix(&config);

        let gaussian_buffer = gpu::create_gaussian_buffer(&device, &state.gaussians);
        let camera_buffer = make_initial_camera_buffer(&device, state.camera, proj_matrix, &config);
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

    /// 입력 이벤트를 받아 카메라 상태와 GPU 카메라 버퍼를 갱신한다.
    ///
    /// 순서: 순수 계산(apply_input → build_camera_uniform) → GPU 버퍼 쓰기(Side Effect)
    fn update(&mut self, input: InputEvent) {
        self.state = self
            .state
            .with_camera(apply_input(self.state.camera, input));

        let viewport_size = [self.config.width as f32, self.config.height as f32];
        let camera_data = gpu::build_camera_uniform(
            camera_ops::camera_to_view_matrix(self.state.camera),
            self.proj_matrix,
            camera_ops::compute_camera_position(self.state.camera),
            viewport_size,
        );
        gpu::update_buffer(&self.queue, &self.camera_buffer, &[camera_data]);
    }

    /// 한 프레임을 렌더링한다.
    ///
    /// 1. 가우시안을 back-to-front 정렬 후 GPU 버퍼 갱신
    /// 2. 스왑체인에서 출력 텍스처 획득
    /// 3. 렌더 패스 실행
    /// 4. 텍스처를 화면에 출력
    fn render(&self) {
        self.upload_sorted_gaussians();

        let Ok(output) = self.surface.get_current_texture() else {
            return; // 창이 최소화됐거나 surface가 유효하지 않으면 건너뜀
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

    /// 현재 카메라 위치 기준으로 가우시안을 back-to-front 정렬해 GPU 버퍼를 갱신한다.
    ///
    /// 알파 블렌딩의 정확성을 위해 매 프레임 호출된다.
    /// 정렬+변환(compute)은 `prepare_sorted_gaussians`가, GPU 업로드(action)는 이 함수가 담당한다.
    fn upload_sorted_gaussians(&self) {
        let camera_pos = camera_ops::compute_camera_position(self.state.camera);
        let sorted =
            compute::gaussian_ops::prepare_sorted_gaussians(&self.state.gaussians, camera_pos);
        gpu::update_buffer(&self.queue, &self.gaussian_buffer, &sorted);
    }
}

// ---------------------------------------------------------------------------
// GPU initialisation helpers
// ---------------------------------------------------------------------------

/// wgpu 인스턴스 → 어댑터 → 디바이스/큐 → 서피스 설정까지 GPU 초기화를 수행한다.
///
/// `Backends::all()`로 플랫폼에 따라 Metal / Vulkan / DX12 중 최적을 자동 선택한다.
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

/// 서피스 포맷과 창 크기를 기반으로 `SurfaceConfiguration`을 생성한다.
///
/// `formats[0]`은 어댑터가 지원하는 기본(권장) 포맷이다.
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
        present_mode: PresentMode::Fifo, // VSync 활성화
        desired_maximum_frame_latency: 2,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![],
    }
}

/// 45° FOV, 현재 창 비율, near=0.1, far=100.0의 투영 행렬을 생성한다.
fn make_proj_matrix(config: &SurfaceConfiguration) -> glam::Mat4 {
    glam::Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4, // 45° vertical FOV
        config.width as f32 / config.height as f32,
        0.1,
        100.0,
    )
}

/// 초기 카메라 상태로 GPU Uniform Buffer를 생성한다.
fn make_initial_camera_buffer(
    device: &Device,
    camera: data::camera::CameraState,
    proj_matrix: glam::Mat4,
    config: &SurfaceConfiguration,
) -> Buffer {
    let view_matrix = camera_ops::camera_to_view_matrix(camera);
    let camera_pos = camera_ops::compute_camera_position(camera);
    let viewport_size = [config.width as f32, config.height as f32];
    gpu::create_camera_buffer(device, view_matrix, proj_matrix, camera_pos, viewport_size)
}

/// 셰이더가 접근할 버퍼의 종류와 바인딩 번호를 정의하는 레이아웃을 생성한다.
///
/// - binding 0: CameraUniform (Uniform Buffer, vertex shader)
/// - binding 1: Gaussian Storage (Storage Buffer read-only, vertex shader)
fn make_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Main"),
        entries: &[
            camera_binding_layout_entry(),
            gaussian_binding_layout_entry(),
        ],
    })
}

/// binding=0, vertex shader에서 읽는 Uniform Buffer 레이아웃 항목
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

/// binding=1, vertex shader에서 읽는 read-only Storage Buffer 레이아웃 항목
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

/// 실제 GPU 버퍼를 레이아웃에 바인딩하는 `BindGroup`을 생성한다.
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

/// 셰이더, 레이아웃, 포맷을 조합해 렌더 파이프라인을 생성한다.
///
/// - 버텍스 버퍼 없음: vertex_index로 쿼드를 인라인 생성
/// - 깊이 버퍼 없음: 깊이 정렬은 CPU에서 back-to-front로 처리
/// - 알파 블렌딩: `BlendState::ALPHA_BLENDING`으로 가우시안 반투명 합성
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
        vertex: make_vertex_state(shader),
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

/// 버텍스 버퍼 없이 vertex_index로만 위치를 계산하는 VertexState를 생성한다.
fn make_vertex_state(shader: &ShaderModule) -> VertexState<'_> {
    VertexState {
        module: shader,
        entry_point: "vs_main",
        buffers: &[],
        compilation_options: PipelineCompilationOptions::default(),
    }
}

// ---------------------------------------------------------------------------
// Pure input helpers
// ---------------------------------------------------------------------------

/// 입력 이벤트를 카메라 상태 변환 순수 함수로 라우팅한다.
///
/// Side Effect 없이 새 `CameraState`만 반환한다.
fn apply_input(camera: data::camera::CameraState, input: InputEvent) -> data::camera::CameraState {
    match input {
        InputEvent::MouseMove(dx, dy) => camera_ops::update_camera_angles(camera, dx, dy),
        InputEvent::Zoom(delta) => camera_ops::zoom_camera(camera, delta),
    }
}

/// 스크롤 델타를 줌 스칼라로 정규화한다.
///
/// `LineDelta`(마우스 휠)와 `PixelDelta`(트랙패드)를 동일한 단위로 변환한다.
fn scroll_delta_to_zoom(delta: winit::event::MouseScrollDelta) -> f32 {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
    }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

/// 마우스 버튼 이벤트를 처리한다.
///
/// 왼쪽 버튼 Press/Release만 추적하고, Release 시 드래그 시작점을 초기화한다.
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

/// 드래그 중이라면 이전 위치와의 델타로 카메라를 회전시킨다.
fn apply_drag_if_active(cur: (f32, f32), last_mouse_pos: Option<(f32, f32)>, app: &mut AppContext) {
    if let Some((lx, ly)) = last_mouse_pos {
        app.update(InputEvent::MouseMove(cur.0 - lx, cur.1 - ly));
    }
}

/// 마우스 이동 이벤트를 처리한다.
///
/// 왼쪽 버튼이 눌린 상태에서만 델타를 계산해 카메라를 회전시킨다.
fn handle_cursor_moved(
    position: winit::dpi::PhysicalPosition<f64>,
    mouse_pressed: bool,
    last_mouse_pos: &mut Option<(f32, f32)>,
    app: &mut AppContext,
) {
    let cur = (position.x as f32, position.y as f32);
    if mouse_pressed {
        apply_drag_if_active(cur, *last_mouse_pos, app);
    }
    *last_mouse_pos = Some(cur);
}

/// 마우스 휠 이벤트를 처리한다.
fn handle_mouse_wheel(delta: winit::event::MouseScrollDelta, app: &mut AppContext) {
    app.update(InputEvent::Zoom(scroll_delta_to_zoom(delta)));
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

    // pollster로 async GPU 초기화를 블로킹 실행
    let mut app = pollster::block_on(AppContext::new(Arc::clone(&window), gaussians));
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f32, f32)> = None;

    // ControlFlow::Poll — 이벤트가 없어도 계속 루프를 돌며 매 프레임 렌더링
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

/// 최상위 이벤트 디스패처.
///
/// `AboutToWait`(이벤트 큐 소진)마다 redraw를 요청해 게임 루프처럼 동작하게 한다.
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

/// 윈도우 이벤트를 종류별로 분기해 적절한 핸들러에 위임한다.
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
