use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// 하나의 3D Gaussian을 표현하는 CPU 측 데이터 구조체.
///
/// 3DGS PLY 파일의 property 순서와 메모리 레이아웃이 1:1로 일치한다.
/// SH(Spherical Harmonics) degree=3 기준으로 62개 f32 = 248 bytes.
///
/// `#[repr(C)]` 로 C ABI 레이아웃을 강제해 bytemuck으로 안전하게 캐스팅할 수 있다.
#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
pub struct Gaussian {
    /// 월드 공간 위치 (x, y, z)
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// 법선 벡터 — 3DGS 렌더링에는 사용되지 않으나 PLY 포맷 호환을 위해 유지
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    /// DC(직류) Spherical Harmonics 계수 — 시점 독립적 기본 색상 (RGB)
    pub f_dc_0: f32,
    pub f_dc_1: f32,
    pub f_dc_2: f32,
    /// 고차 SH 계수 (degree 1~3) — 시점에 따른 색상 변화 표현, 45 = (3+5+7)×3채널
    pub f_rest: [f32; 45],
    /// 불투명도 (logit 공간 — sigmoid 적용 후 0~1)
    pub opacity: f32,
    /// 로그 스케일 크기 (exp 적용 후 실제 크기)
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    /// 회전을 나타내는 쿼터니언 (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    /// WGSL quat_to_mat의 r.x=w, r.y=x, r.z=y, r.w=z 매핑과 일치한다
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
}

impl Gaussian {
    /// 월드 공간 위치를 Vec3로 반환
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// DC SH 계수를 RGB 색상 Vec3로 반환
    pub fn color(&self) -> Vec3 {
        Vec3::new(self.f_dc_0, self.f_dc_1, self.f_dc_2)
    }

    /// 로그 스케일 크기를 Vec3로 반환
    pub fn scale(&self) -> Vec3 {
        Vec3::new(self.scale_0, self.scale_1, self.scale_2)
    }

    /// 쿼터니언을 [rot_0, rot_1, rot_2, rot_3] 배열로 반환
    pub fn rotation(&self) -> [f32; 4] {
        [self.rot_0, self.rot_1, self.rot_2, self.rot_3]
    }
}
