import struct
import random

def generate_dummy_ply(filepath, num_gaussians=1000):
    with open(filepath, 'wb') as f:
        # 1. Write Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_gaussians}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
        for i in range(45):
            header += f"property float f_rest_{i}\n"
            
        header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        f.write(header.encode('ascii'))
        
        # 2. Write Binary Data
        for _ in range(num_gaussians):
            # pos (x, y, z): Random position in [-2, 2]
            pos = [random.uniform(-2, 2) for _ in range(3)]
            # normal (nx, ny, nz)
            normal = [0.0, 0.0, 0.0]
            # color dc (RGB) - randomized
            f_dc = [random.uniform(-1, 1) for _ in range(3)]
            # f_rest (45)
            f_rest = [0.0] * 45
            # opacity
            opacity = [2.0]
            # scale
            scale = [-3.0, -3.0, -3.0]
            # rotation (quaternion)
            rot = [1.0, 0.0, 0.0, 0.0]
            
            data = pos + normal + f_dc + f_rest + opacity + scale + rot
            # Pack as 62 floats (stride = 248 bytes)
            packed = struct.pack(f"<{len(data)}f", *data)
            f.write(packed)

if __name__ == '__main__':
    generate_dummy_ply('example.ply', 1000)
    print("Generated dummy PLY file at assets/example.ply")
