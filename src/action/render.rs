use wgpu::*;

/// 한 프레임을 렌더링한다.
///
/// 가우시안 하나당 빌보드 쿼드(삼각형 2개 = 6 vertex)를 그린다.
/// 버텍스 버퍼 없이 vertex_index만으로 쿼드를 생성하는 방식이므로 `buffers: &[]`.
///
/// 렌더 순서:
/// 1. CommandEncoder 생성
/// 2. RenderPass 시작 (화면을 검정으로 클리어)
/// 3. 파이프라인 + bind group 설정
/// 4. draw 호출 (`gaussian_count * 6` vertex)
/// 5. 커맨드 큐에 제출
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

    {
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
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

    queue.submit(std::iter::once(encoder.finish()));
}
