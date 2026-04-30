# Functional 3DGS Viewer

A cross-platform 3D Gaussian Splatting viewer built with Rust and wgpu, following functional programming principles (Data / Compute / Action separation).

## Features

- Cross-platform: macOS (Metal), Linux (Vulkan), Windows (DX12) — same codebase
- Functional architecture based on Eric Normand's data/calculation/action model
- Binary PLY file parsing (3DGS SH degree=3)
- Per-frame back-to-front depth sorting with Rayon parallel sort
- Alpha blending billboard rendering via WGSL shader
- Mouse drag orbit + scroll zoom camera control

## Architecture

```
src/
├── data/          # Immutable data structures (Gaussian, CameraState, AppState)
├── compute/       # Pure functions — no side effects, fully testable
│   ├── camera_ops.rs
│   ├── gaussian_ops.rs
│   └── ply_parse.rs
├── action/        # Side-effecting functions (I/O, GPU, rendering)
│   ├── io.rs
│   ├── gpu.rs
│   └── render.rs
├── shaders/
│   └── render.wgsl
└── main.rs        # Event loop and action orchestration
```

## Requirements

- Rust 1.75+
- A GPU with Metal / Vulkan / DX12 support

## Getting Started

### 1. Clone

```bash
git clone https://github.com/y30n9ju1v/rust-wgpu-based-functional-3dgs-viewer.git
cd rust-wgpu-based-functional-3dgs-viewer
```

### 2. Generate a dummy PLY (optional)

```bash
cd assets
python3 generate_dummy_ply.py   # outputs example.ply
cd ..
```

### 3. Run

```bash
cargo run --release
```

The viewer loads `assets/example.ply` on startup. Replace it with any 3DGS PLY file (SH degree=3, 62 floats per vertex).

## Controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit camera |
| Scroll wheel | Zoom in / out |
| Close window | Exit |

## Dependencies

| Crate | Purpose |
|---|---|
| `wgpu` | Cross-platform GPU API (Metal / Vulkan / DX12 / WebGPU) |
| `winit` | Window and input handling |
| `glam` | Linear algebra (Vec3, Mat4, Quat) |
| `bytemuck` | Safe GPU buffer casting |
| `rayon` | Parallel depth sorting |
| `pollster` | Blocking async executor (avoids tokio conflict with winit) |
| `anyhow` | Error handling |

## PLY Format

The viewer expects binary little-endian PLY with the following 62 float properties per vertex:

```
x y z nx ny nz
f_dc_0 f_dc_1 f_dc_2
f_rest_0 … f_rest_44
opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
```

This matches the output of the [official 3DGS training code](https://github.com/graphdeco-inria/gaussian-splatting).

## License
