use crate::compute::ply_parse;
use crate::data::gaussian::Gaussian;
use std::fs::File;
use std::io::Read;

/// PLY 파일을 읽어 `Gaussian` 벡터로 반환한다.
///
/// 헤더에서 vertex 수를 추출하고, 헤더 이후 바이너리 데이터를 파싱한다.
/// 3DGS SH degree=3 포맷(62 floats/vertex)만 지원한다.
pub fn load_ply_file(path: &str) -> anyhow::Result<Vec<Gaussian>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (header_str, data_start) = parse_ply_header(&buffer)?;
    let vertex_count = extract_vertex_count(&header_str)?;

    // 3DGS SH degree=3 고정: 62개 f32 = 248 bytes
    const PLY_STRIDE: usize = 62 * 4;
    // Gaussian 레이아웃이 PLY stride와 다르면 잘못 파싱되므로 컴파일 타임에 차단한다
    const _: () = assert!(PLY_STRIDE == std::mem::size_of::<Gaussian>());
    let stride = PLY_STRIDE;

    ply_parse::parse_gaussians(&buffer[data_start..], stride, vertex_count)
        .map_err(|e| anyhow::anyhow!(e))
}

/// PLY 헤더 끝(`end_header\n`)을 찾아 헤더 문자열과 바이너리 데이터 시작 오프셋을 반환한다.
fn parse_ply_header(buffer: &[u8]) -> anyhow::Result<(String, usize)> {
    const END_HEADER: &[u8] = b"end_header\n";

    let end_pos = buffer
        .windows(END_HEADER.len())
        .position(|w| w == END_HEADER)
        .ok_or_else(|| anyhow::anyhow!("end_header not found in PLY file"))?;

    let header_str = std::str::from_utf8(&buffer[..end_pos])
        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in PLY header: {}", e))?
        .to_string();

    Ok((header_str, end_pos + END_HEADER.len()))
}

/// 헤더 문자열에서 `element vertex <count>` 행을 찾아 vertex 수를 반환한다.
fn extract_vertex_count(header: &str) -> anyhow::Result<usize> {
    header
        .lines()
        .find_map(parse_vertex_count_line)
        .ok_or_else(|| anyhow::anyhow!("element vertex count not found in PLY header"))
}

/// `element vertex <n>` 형식의 줄이면 n을 파싱해 반환하고, 아니면 None을 반환한다.
fn parse_vertex_count_line(line: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let is_vertex_element = parts.len() == 3 && parts[0] == "element" && parts[1] == "vertex";
    if is_vertex_element {
        parts[2].parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ply_buffer(header_extra: &str, data: &[u8]) -> Vec<u8> {
        let header = format!("ply\nformat binary_little_endian 1.0\n{header_extra}end_header\n");
        let mut buf = header.into_bytes();
        buf.extend_from_slice(data);
        buf
    }

    // --- parse_vertex_count_line ---

    #[test]
    fn test_vertex_count_line_valid() {
        assert_eq!(parse_vertex_count_line("element vertex 42"), Some(42));
    }

    #[test]
    fn test_vertex_count_line_not_vertex() {
        assert_eq!(parse_vertex_count_line("element face 10"), None);
    }

    #[test]
    fn test_vertex_count_line_wrong_keyword() {
        assert_eq!(parse_vertex_count_line("property float x"), None);
    }

    #[test]
    fn test_vertex_count_line_invalid_number() {
        assert_eq!(parse_vertex_count_line("element vertex abc"), None);
    }

    #[test]
    fn test_vertex_count_line_empty() {
        assert_eq!(parse_vertex_count_line(""), None);
    }

    // --- extract_vertex_count ---

    #[test]
    fn test_extract_vertex_count_found() {
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 100\n";
        assert_eq!(extract_vertex_count(header).unwrap(), 100);
    }

    #[test]
    fn test_extract_vertex_count_not_found() {
        let header = "ply\nformat binary_little_endian 1.0\n";
        assert!(extract_vertex_count(header).is_err());
    }

    #[test]
    fn test_extract_vertex_count_picks_first() {
        let header = "element vertex 5\nelement vertex 99\n";
        assert_eq!(extract_vertex_count(header).unwrap(), 5);
    }

    // --- parse_ply_header ---

    #[test]
    fn test_parse_ply_header_valid() {
        let buf = make_ply_buffer("element vertex 3\n", &[1, 2, 3]);
        let (header, offset) = parse_ply_header(&buf).unwrap();
        assert!(header.contains("element vertex 3"));
        assert_eq!(&buf[offset..], &[1, 2, 3]);
    }

    #[test]
    fn test_parse_ply_header_missing_end_header() {
        let buf = b"ply\nformat binary_little_endian 1.0\n".to_vec();
        assert!(parse_ply_header(&buf).is_err());
    }

    #[test]
    fn test_parse_ply_header_empty_data_section() {
        let buf = make_ply_buffer("element vertex 0\n", &[]);
        let (_, offset) = parse_ply_header(&buf).unwrap();
        assert_eq!(offset, buf.len());
    }
}
