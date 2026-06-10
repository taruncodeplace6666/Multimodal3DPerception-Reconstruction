#!/usr/bin/env python3
"""
Convert KITTI LiDAR TUM trajectory (from FAST-LIVO2 /aft_mapped_to_init)
into a Nerfstudio transforms.json file, with correct OpenGL convention:
  X right, Y up, Z backward (out of image plane)
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ======================= CONFIGURATION =======================
TUM_FILE      = "trajectory.tum"      # input TUM file (8 cols)
OUTPUT_JSON   = "transforms.json"          # output for Nerfstudio
IMAGE_FOLDER  = "images"                   # folder containing images
IMAGE_PREFIX  = "frame"                    # e.g., frame0000.jpg
IMAGE_FORMAT  = "jpg"                      # image extension
# =============================================================

# ---------- 1. LiDAR → cam00 (unrectified) from calib_velo_to_cam.txt ----------
R_velo_to_cam = np.array([
    [ 7.533745e-03, -9.999714e-01, -6.166020e-04],
    [ 1.480249e-02,  7.280733e-04, -9.998902e-01],
    [ 9.998621e-01,  7.523790e-03,  1.480755e-02]
])
t_velo_to_cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

T_velo_to_cam0 = np.eye(4)
T_velo_to_cam0[:3, :3] = R_velo_to_cam
T_velo_to_cam0[:3, 3]  = t_velo_to_cam

# ---------- 2. Rectification for cam00 (R_rect_00) ----------
R_rect_00 = np.array([
    [ 9.999239e-01,  9.837760e-03, -7.445048e-03],
    [-9.869795e-03,  9.999421e-01, -4.278459e-03],
    [ 7.402527e-03,  4.351614e-03,  9.999631e-01]
])
T_rect_00 = np.eye(4)
T_rect_00[:3, :3] = R_rect_00

# ---------- 3. cam00_unrect → cam02_unrect (R_02, T_02) ----------
R_02 = np.array([
    [ 9.999758e-01, -5.267463e-03, -4.552439e-03],
    [ 5.251945e-03,  9.999804e-01, -3.413835e-03],
    [ 4.570332e-03,  3.389843e-03,  9.999838e-01]
])
T_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])

T_cam0_to_cam2 = np.eye(4)
T_cam0_to_cam2[:3, :3] = R_02
T_cam0_to_cam2[:3, 3]  = T_02

# ---------- 4. Rectification for cam02 (R_rect_02) ----------
R_rect_02 = np.array([
    [ 9.998817e-01,  1.511453e-02, -2.841595e-03],
    [-1.511724e-02,  9.998853e-01, -9.338510e-04],
    [ 2.827154e-03,  9.766976e-04,  9.999955e-01]
])
T_rect_02 = np.eye(4)
T_rect_02[:3, :3] = R_rect_02

# ---------- 5. Build full LiDAR → cam02_rect transformation ----------
T_lidar_to_cam2_rect = T_rect_02 @ T_cam0_to_cam2 @ T_rect_00 @ T_velo_to_cam0

print("LiDAR → cam02_rect (camera frame) transformation:")
print(np.array2string(T_lidar_to_cam2_rect, precision=6, suppress_small=True))

# Verify rotation part is near‑orthonormal
R_check = T_lidar_to_cam2_rect[:3, :3]
ortho_err = np.max(np.abs(R_check.T @ R_check - np.eye(3)))
print(f"Rotation orthonormality error: {ortho_err:.2e}")

# ---------- 6. Camera intrinsics from P_rect_02 ----------
fx     = 721.5377
fy     = 721.5377
cx     = 609.5593
cy     = 172.854
width  = 1242
height = 375
print(f"\nIntrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, w={width}, h={height}")

# ---------- 7. TUM world (X‑fwd, Y‑left, Z‑up) → OpenGL world (X‑right, Y‑up, Z‑backward) ----------
M_tum_to_gl = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1]
], dtype=float)

# Additional rotation to flip camera's Y and Z axes (because KITTI camera has Y‑down, Z‑forward)
R_flip = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])

# ---------- 8. Main conversion function ----------
def tum_to_nerfstudio():
    frames = []
    frame_idx = 0

    with open(TUM_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue

            ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts)

            # LiDAR pose in TUM world frame (X‑fwd, Y‑left, Z‑up)
            T_world_lidar = np.eye(4)
            T_world_lidar[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_world_lidar[:3, 3]  = [tx, ty, tz]

            # Camera pose in TUM world frame
            T_world_cam2_tum = T_world_lidar @ np.linalg.inv(T_lidar_to_cam2_rect)

            # Convert to OpenGL world
            T_world_cam2_gl = M_tum_to_gl @ T_world_cam2_tum

            # Flip camera's Y and Z axes (180° about X) to meet OpenGL convention
            T_world_cam2_gl[:3, :3] = T_world_cam2_gl[:3, :3] @ R_flip

            img_name = f"{IMAGE_PREFIX}{frame_idx:04d}.{IMAGE_FORMAT}"
            frames.append({
                "file_path": f"{IMAGE_FOLDER}/{img_name}",
                "transform_matrix": T_world_cam2_gl.tolist(),
                "colmap_im_id": frame_idx
            })
            frame_idx += 1

    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "frames": frames
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(transforms, f, indent=4)

    print(f"\n✅ Saved {OUTPUT_JSON} with {len(frames)} frames.")

    # ---------- Sanity checks for the first frame ----------
    if frames:
        T0 = np.array(frames[0]["transform_matrix"])
        R0 = T0[:3, :3]
        view_dir = R0[:, 2]
        print(f"Viewing direction (cam Z in world): {np.round(view_dir, 4)}")
        print("✅ Looks backward (+Z)" if view_dir[2] > 0 else "⚠️  Warning: camera may be looking forward")

        cam_y = R0[:, 1]
        print(f"Camera Y‑axis in world:             {np.round(cam_y, 4)}")
        print("✅ Points up (+Y)" if cam_y[1] > 0 else "❌ ERROR: Y does not point up")

        ortho_err = np.max(np.abs(R0.T @ R0 - np.eye(3)))
        print(f"Rotation orthonormality error:      {ortho_err:.2e}")

        t0 = T0[:3, 3]
        print(f"First frame translation:            {np.round(t0, 4)}")


if __name__ == "__main__":
    tum_to_nerfstudio()
