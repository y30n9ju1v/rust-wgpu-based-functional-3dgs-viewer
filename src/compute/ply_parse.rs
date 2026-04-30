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

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: usize = std::mem::size_of::<Gaussian>();

    /// f32 값을 little-endian 4바이트로 직렬화한다.
    fn f32_le(v: f32) -> [u8; 4] {
        v.to_le_bytes()
    }

    /// 지정된 필드 값들로 채운 248바이트 PLY 레코드를 생성한다.
    /// 나머지 필드는 0.0으로 채운다.
    fn make_record(
        pos: [f32; 3],
        normal: [f32; 3],
        f_dc: [f32; 3],
        f_rest: [f32; 45],
        opacity: f32,
        scale: [f32; 3],
        rot: [f32; 4],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(STRIDE);
        for v in pos {
            buf.extend(f32_le(v));
        }
        for v in normal {
            buf.extend(f32_le(v));
        }
        for v in f_dc {
            buf.extend(f32_le(v));
        }
        for v in f_rest {
            buf.extend(f32_le(v));
        }
        buf.extend(f32_le(opacity));
        for v in scale {
            buf.extend(f32_le(v));
        }
        for v in rot {
            buf.extend(f32_le(v));
        }
        assert_eq!(buf.len(), STRIDE);
        buf
    }

    fn default_record() -> Vec<u8> {
        make_record(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.6, 0.7],
            [0.1; 45],
            0.8,
            [0.2, 0.3, 0.4],
            [1.0, 0.0, 0.0, 0.0],
        )
    }

    // --- parse_gaussian_from_bytes ---

    #[test]
    fn test_parse_position() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.x - 1.0).abs() < 1e-6);
        assert!((g.y - 2.0).abs() < 1e-6);
        assert!((g.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_normal() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.nx - 0.0).abs() < 1e-6);
        assert!((g.ny - 1.0).abs() < 1e-6);
        assert!((g.nz - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_color_dc() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.f_dc_0 - 0.5).abs() < 1e-6);
        assert!((g.f_dc_1 - 0.6).abs() < 1e-6);
        assert!((g.f_dc_2 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_parse_f_rest() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        for v in g.f_rest {
            assert!((v - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parse_opacity() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.opacity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_parse_scale() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.scale_0 - 0.2).abs() < 1e-6);
        assert!((g.scale_1 - 0.3).abs() < 1e-6);
        assert!((g.scale_2 - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_parse_rotation() {
        let data = default_record();
        let g = parse_gaussian_from_bytes(&data).unwrap();
        assert!((g.rot_0 - 1.0).abs() < 1e-6);
        assert!((g.rot_1 - 0.0).abs() < 1e-6);
        assert!((g.rot_2 - 0.0).abs() < 1e-6);
        assert!((g.rot_3 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_insufficient_data_returns_error() {
        let short = vec![0u8; STRIDE - 1];
        assert!(parse_gaussian_from_bytes(&short).is_err());
    }

    #[test]
    fn test_parse_error_message_contains_byte_counts() {
        let short = vec![0u8; 10];
        let err = parse_gaussian_from_bytes(&short).unwrap_err();
        assert!(err.contains("10"));
        assert!(err.contains(&STRIDE.to_string()));
    }

    // --- parse_gaussians ---

    #[test]
    fn test_parse_two_gaussians() {
        let r1 = make_record(
            [1.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0.0; 45],
            0.0,
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
        );
        let r2 = make_record(
            [2.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0.0; 45],
            0.0,
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
        );
        let data = [r1, r2].concat();
        let gaussians = parse_gaussians(&data, STRIDE, 2).unwrap();
        assert_eq!(gaussians.len(), 2);
        assert!((gaussians[0].x - 1.0).abs() < 1e-6);
        assert!((gaussians[1].x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_zero_count_returns_empty() {
        let gaussians = parse_gaussians(&[], STRIDE, 0).unwrap();
        assert!(gaussians.is_empty());
    }
}
