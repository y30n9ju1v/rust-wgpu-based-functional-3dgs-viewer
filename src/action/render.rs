use wgpu::*;

/// 한 프레임을 렌더링한다.
///
/// 가우시안 하나당 빌보드 쿼드(삼각형 2개 = 6 vertex)를 그린다.
/// 버텍스 버퍼 없이 vertex_index만으로 쿼드를 생성하는 방식이므로 `buffers: &[]`.
pub fn render_frame(
    device: &Device,
    queue: &Queue,
    view: &TextureView,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    gaussian_count: u32,
) {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });

    record_render_pass(&mut encoder, view, pipeline, bind_group, gaussian_count);

    queue.submit(std::iter::once(encoder.finish()));
}

/// RenderPass를 encoder에 기록한다.
///
/// 수명 제약(rpass가 encoder를 borrow)으로 별도 함수로 분리해
/// `render_frame`의 중첩 블록 없이 encoder를 finish할 수 있게 한다.
fn record_render_pass(
    encoder: &mut CommandEncoder,
    view: &TextureView,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    gaussian_count: u32,
) {
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });

    rpass.set_pipeline(pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    // 가우시안 1개 = 쿼드 2삼각형 = 6 vertex
    rpass.draw(0..gaussian_count * 6, 0..1);
}
