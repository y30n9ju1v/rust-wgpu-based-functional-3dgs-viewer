use crate::data::gaussian::Gaussian;
use crate::compute::ply_parse;
use std::fs::File;
use std::io::Read;

pub fn load_ply_file(path: &str) -> anyhow::Result<Vec<Gaussian>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let (header_str, data_start) = parse_ply_header(&buffer)?;
    let vertex_count = extract_vertex_count(&header_str)?;
    
    // 3DGS SH degree=3 고정: 62개 f32 = 248 bytes
    const PLY_STRIDE: usize = 62 * 4;
    const _: () = assert!(PLY_STRIDE == std::mem::size_of::<Gaussian>());
    let stride = PLY_STRIDE;
    
    ply_parse::parse_gaussians(&buffer[data_start..], stride, vertex_count)
        .map_err(|e| anyhow::anyhow!(e))
}

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

fn extract_vertex_count(header: &str) -> anyhow::Result<usize> {
    header
        .lines()
        .find_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "element" && parts[1] == "vertex" {
                parts[2].parse().ok()
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow::anyhow!("element vertex count not found in PLY header"))
}
