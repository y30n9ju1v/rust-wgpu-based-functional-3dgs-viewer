use crate::data::gaussian::Gaussian;

// PLY property 오프셋 상수 (bytes). 단위: bytes = field_index × 4.
const OFF_X: usize = 0;
const OFF_Y: usize = 4;
const OFF_Z: usize = 8;
// 법선 — 3DGS 렌더링에는 쓰이지 않지만 PLY 포맷 호환을 위해 오프셋을 보존한다
const OFF_NX: usize = 12;
const OFF_NY: usize = 16;
const OFF_NZ: usize = 20;
const OFF_F_DC_0: usize = 24;
const OFF_F_DC_1: usize = 28;
const OFF_F_DC_2: usize = 32;
const OFF_F_REST: usize = 36; // 45 × 4 = 180 bytes
const OFF_OPACITY: usize = 216;
const OFF_SCALE_0: usize = 220;
const OFF_SCALE_1: usize = 224;
const OFF_SCALE_2: usize = 228;
const OFF_ROT_0: usize = 232;
const OFF_ROT_1: usize = 236;
const OFF_ROT_2: usize = 240;
const OFF_ROT_3: usize = 244;

/// little-endian f32 4바이트를 읽는다.
fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// f_rest 45개 계수를 읽는다.
fn read_f_rest(data: &[u8]) -> [f32; 45] {
    let mut f_rest = [0.0f32; 45];
    for (j, val) in f_rest.iter_mut().enumerate() {
        *val = read_f32(data, OFF_F_REST + j * 4);
    }
    f_rest
}

/// 바이너리 바이트 슬라이스 하나를 `Gaussian` 구조체 하나로 파싱한다.
///
/// PLY 바이너리 포맷은 little-endian f32의 연속이며, 오프셋은 property 선언 순서와 일치한다.
pub fn parse_gaussian_from_bytes(data: &[u8]) -> Result<Gaussian, String> {
    let required = std::mem::size_of::<Gaussian>();
    if data.len() < required {
        return Err(format!(
            "insufficient data: got {} bytes, need {}",
            data.len(),
            required
        ));
    }

    Ok(Gaussian {
        x: read_f32(data, OFF_X),
        y: read_f32(data, OFF_Y),
        z: read_f32(data, OFF_Z),
        nx: read_f32(data, OFF_NX),
        ny: read_f32(data, OFF_NY),
        nz: read_f32(data, OFF_NZ),
        f_dc_0: read_f32(data, OFF_F_DC_0),
        f_dc_1: read_f32(data, OFF_F_DC_1),
        f_dc_2: read_f32(data, OFF_F_DC_2),
        f_rest: read_f_rest(data),
        opacity: read_f32(data, OFF_OPACITY),
        scale_0: read_f32(data, OFF_SCALE_0),
        scale_1: read_f32(data, OFF_SCALE_1),
        scale_2: read_f32(data, OFF_SCALE_2),
        rot_0: read_f32(data, OFF_ROT_0),
        rot_1: read_f32(data, OFF_ROT_1),
        rot_2: read_f32(data, OFF_ROT_2),
        rot_3: read_f32(data, OFF_ROT_3),
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

    // --- read_f32 ---

    #[test]
    fn test_read_f32_zero() {
        let data = 0.0f32.to_le_bytes();
        assert_eq!(read_f32(&data, 0), 0.0);
    }

    #[test]
    fn test_read_f32_one() {
        let data = 1.0f32.to_le_bytes();
        assert_eq!(read_f32(&data, 0), 1.0);
    }

    #[test]
    fn test_read_f32_with_offset() {
        let mut data = vec![0u8; 8];
        data[4..8].copy_from_slice(&42.0f32.to_le_bytes());
        assert_eq!(read_f32(&data, 4), 42.0);
    }

    #[test]
    fn test_read_f32_negative() {
        let data = (-3.14f32).to_le_bytes();
        let v = read_f32(&data, 0);
        assert!((v - (-3.14f32)).abs() < 1e-6);
    }

    // --- read_f_rest ---

    #[test]
    fn test_read_f_rest_all_zeros() {
        let data = vec![0u8; STRIDE];
        let f_rest = read_f_rest(&data);
        assert!(f_rest.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_read_f_rest_values() {
        let mut data = vec![0u8; STRIDE];
        for i in 0..45usize {
            let val = (i as f32) * 0.1;
            let bytes = val.to_le_bytes();
            let off = OFF_F_REST + i * 4;
            data[off..off + 4].copy_from_slice(&bytes);
        }
        let f_rest = read_f_rest(&data);
        for (i, &v) in f_rest.iter().enumerate() {
            assert!((v - i as f32 * 0.1).abs() < 1e-6, "index {i} mismatch");
        }
    }

    #[test]
    fn test_read_f_rest_length() {
        let data = vec![0u8; STRIDE];
        let f_rest = read_f_rest(&data);
        assert_eq!(f_rest.len(), 45);
    }
}
