use crate::data::gaussian::Gaussian;

pub fn parse_gaussian_from_bytes(data: &[u8]) -> Result<Gaussian, String> {
    let required = std::mem::size_of::<Gaussian>();
    if data.len() < required {
        return Err(format!(
            "insufficient data: got {} bytes, need {}",
            data.len(),
            required
        ));
    }

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

pub fn parse_gaussians(data: &[u8], stride: usize, count: usize) -> Result<Vec<Gaussian>, String> {
    (0..count)
        .map(|i| {
            let offset = i * stride;
            parse_gaussian_from_bytes(&data[offset..offset + stride])
        })
        .collect()
}
