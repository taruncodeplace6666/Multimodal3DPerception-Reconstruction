import pykitti
import numpy as np
from scipy.spatial.transform import Rotation as R

BASEDIR = "./data"
DATE = "2011_09_26"
DRIVE = "0023"

dataset = pykitti.raw(BASEDIR, DATE, DRIVE)

# LiDAR (Velodyne) to IMU extrinsic from KITTI calib
T_imu_to_velo = dataset.calib.T_velo_imu          # velo-from-imu
T_velo_to_imu = np.linalg.inv(T_imu_to_velo)      # imu-from-velo

with open("ground_truth.tum", "w") as f:
    f.write("# timestamp tx ty tz qx qy qz qw\n")

    for i, oxts in enumerate(dataset.oxts):
        T_w_imu = oxts.T_w_imu

        # World-to-LiDAR: go world→imu→velo
        T_w_velo = T_w_imu @ T_velo_to_imu

        t = T_w_velo[:3, 3]
        q = R.from_matrix(T_w_velo[:3, :3]).as_quat()  # qx qy qz qw
        timestamp = dataset.timestamps[i].timestamp()

        f.write(f"{timestamp:.6f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")

print(f"Saved {i+1} poses to ground_truth.tum")