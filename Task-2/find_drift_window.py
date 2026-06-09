#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.linalg import svd

def load_tum(filename):
    """Load TUM trajectory file (timestamp x y z qx qy qz qw)"""
    df = pd.read_csv(filename, sep=' ', header=None,
                     names=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    return df

def umeyama_sim3(src, dst):
    """
    Compute Sim(3) transformation (sR, t) from src to dst.
    src, dst: Nx3 arrays of points.
    Returns: s (scale), R (3x3 rotation), t (3x1 translation)
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = svd(H)   # S is 1D array of singular values
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale: sum of singular values divided by trace(src_centered^T src_centered)
    src_var = np.sum(src_centered ** 2)   # equivalent to np.trace(src_centered.T @ src_centered)
    if src_var > 0:
        s = np.sum(S) / src_var
    else:
        s = 1.0

    t = dst_mean - s * (R @ src_mean)
    return s, R, t

def apply_sim3(points, s, R, t):
    """Apply Sim(3) transformation to Nx3 points."""
    return s * (R @ points.T).T + t

def main():
    # Load trajectories
    gt_df = load_tum("ground_truth.tum")
    est_df = load_tum("trajectory.tum")

    # Sort by timestamp
    gt_df = gt_df.sort_values('t')
    est_df = est_df.sort_values('t')

    gt_t = gt_df['t'].values
    gt_xyz = gt_df[['x','y','z']].values
    est_t = est_df['t'].values
    est_xyz = est_df[['x','y','z']].values

    # Associate timestamps (nearest neighbor within 0.01s)
    common_t = []
    est_at_common = []
    gt_at_common = []
    for i, t_gt in enumerate(gt_t):
        idx = np.argmin(np.abs(est_t - t_gt))
        if np.abs(est_t[idx] - t_gt) < 0.05:
            common_t.append(t_gt)
            est_at_common.append(est_xyz[idx])
            gt_at_common.append(gt_xyz[i])

    common_t = np.array(common_t)
    est_points = np.array(est_at_common)
    gt_points = np.array(gt_at_common)

    if len(common_t) < 5:
        print("ERROR: Fewer than 3 matching timestamps. Try increasing max_diff.")
        return

    # Align estimated to ground truth using Sim(3)
    s, R, t = umeyama_sim3(est_points, gt_points)
    est_aligned = apply_sim3(est_points, s, R, t)

    # Per‑pose translational error
    ate = np.linalg.norm(est_aligned - gt_points, axis=1)

    # Find 2‑second window with largest error increase
    max_increase = 0.0
    drift_start_idx = 0
    drift_end_idx = 0

    for i in range(len(common_t)):
        end_time = common_t[i] + 2.0
        j = i
        while j < len(common_t) and common_t[j] <= end_time:
            j += 1
        if j > i + 1:
            increase = ate[j-1] - ate[i]
            if increase > max_increase:
                max_increase = increase
                drift_start_idx = i
                drift_end_idx = j - 1

    drift_start = common_t[drift_start_idx]
    drift_end   = common_t[drift_end_idx]
    duration = drift_end - drift_start

    print(f"Drift window: {drift_start:.2f} – {drift_end:.2f} seconds")
    print(f"Error increased by {max_increase:.2f} m in {duration:.2f} s")
    print(f"Drift rate: {max_increase / duration:.2f} m/s")
    print(f"Scale factor from alignment: {s:.3f}")

if __name__ == "__main__":
    main()
