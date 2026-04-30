use crate::data::gaussian::Gaussian;

/// 바이너리 바이트 슬라이스 하나를 `Gaussian` 구조체 하나로 파싱한다.
///
/// PLY 바이너리 포맷은 little-endian f32의 연속이며, 오프셋은 property 선언 순서와 일치한다.
/// - 0..12   : pos (x, y, z)
/// - 12..24  : normal (nx, ny, nz)
/// - 24..36  : f_dc (0, 1, 2)
/// - 36..216 : f_rest[0..45]  (45 × 4 = 180 bytes)
/// - 216     : opacity
/// - 220..232: scale (0, 1, 2)
/// - 232..248: rot (0, 1, 2, 3)
pub fn parse_gaussian_from_bytes(data: &[u8]) -> Result<Gaussian, String> {
    let required = std::mem::size_of::<Gaussian>();
    if data.len() < required {
        return Err(format!(
            "insufficient data: got {} bytes, need {}",
            data.len(),
            required
        ));
    }

    // 클로저로 오프셋 → f32 변환을 재사용한다 (little-endian 4바이트)
    let read_f32 = |offset: usize| -> f32 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    };

    let mut f_rest = [0.0f32; 45];
    for (j, val) in f_rest.iter_mut().enumerate() {
        *val = read_f32(36 + j * 4);
    }

    Ok(Gaussian {
        x: read_f32(0),
        y: read_f32(4),
        z: read_f32(8),
        nx: read_f32(12),
        ny: read_f32(16),
        nz: read_f32(20),
        f_dc_0: read_f32(24),
        f_dc_1: read_f32(28),
        f_dc_2: read_f32(32),
        f_rest,
        opacity: read_f32(216),
        scale_0: read_f32(220),
        scale_1: read_f32(224),
        scale_2: read_f32(228),
        rot_0: read_f32(232),
        rot_1: read_f32(236),
        rot_2: read_f32(240),
        rot_3: read_f32(244),
    })
}

/// 바이너리 데이터 전체를 `stride` 간격으로 나눠 `count`개의 `Gaussian`으로 파싱한다.
///
/// 각 가우시안은 `stride` 바이트를 차지하며, 에러가 하나라도 있으면 전체가 실패한다.
pub fn parse_gaussians(data: &[u8], stride: usize, count: usize) -> Result<Vec<Gaussian>, String> {
    (0..count)
        .map(|i| {
            let offset = i * stride;
            parse_gaussian_from_bytes(&data[offset..offset + stride])
        })
        .collect()
}
