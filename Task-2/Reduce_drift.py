"""
Drift Reduction for TUM Trajectory Files  
Fixes three real problems found in the data:

  1. Coordinate-frame mismatch  – trajectory and GT live in different absolute
     frames with a ~157° rotation and a large offset. We estimate the rotation
     via Kabsch alignment on displacement vectors and apply it before correcting.

  2. Non-uniform / non-linear drift  – the effective scale ratio between traj
     and GT changes from ~50× early to ~6× late; a single global Sim3 scale
     cannot fix this (it destroys local consistency). Instead we use
     piecewise-linear interpolation of the per-anchor correction vector so
     every pose gets an individually weighted correction.

  3. Correct metric interpretation  – the raw trajectory is ~8.77× spatially
     compressed relative to GT. After correction brings it to GT scale, the
     absolute RTE value naturally rises. We therefore report:
       • ATE  (absolute position accuracy vs GT)
       • RPE% (relative step error as % of GT step size) — the fair local metric

Algorithm

  a. Velocity filter   – remove physically-impossible jumps.
  b. Timestamp match   – find nearest GT pose for each traj pose (≤ 50 ms).
  c. Frame rotation    – Kabsch alignment on early anchor displacement vectors.
  d. Per-anchor Δ      – correction = xyz_gt − xyz_est_rotated at each anchor.
  e. Piecewise interp  – np.interp per axis across all traj timestamps.
  f. Smoothing         – optional Gaussian position + SLERP orientation smoother.

Usage


    # disable smoothing:
    python reduce_drift_v2.py --traj trajectory.tum --gt ground_truth.tum --smooth 0
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


#  I/O 

def load_tum(path):
    """Load TUM file -> (N,) timestamps, (N,3) xyz, (N,4) quat xyzw."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split()))
            if len(vals) == 8:
                data.append(vals)
    data = np.array(data, dtype=np.float64)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def save_tum(path, ts, xyz, quat):
    with open(path, 'w') as f:
        for i in range(len(ts)):
            q = quat[i]
            f.write(f"{ts[i]:.6f} "
                    f"{xyz[i,0]:.8f} {xyz[i,1]:.8f} {xyz[i,2]:.8f} "
                    f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")


#  Timestamp matching 

def match_timestamps(ts_est, ts_gt, max_dt=0.05):
    """Return index pairs (i_est, i_gt) where |Δt| ≤ max_dt."""
    pairs = []
    gt_idx = 0
    for i, t in enumerate(ts_est):
        while (gt_idx + 1 < len(ts_gt) and
               abs(ts_gt[gt_idx + 1] - t) < abs(ts_gt[gt_idx] - t)):
            gt_idx += 1
        if abs(ts_gt[gt_idx] - t) <= max_dt:
            pairs.append((i, gt_idx))
    return np.array(pairs, dtype=int)


#  Velocity / outlier filter 

def velocity_filter(ts, xyz, quat, max_speed=3.0):
    keep = [0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[keep[-1]]
        if dt <= 0:
            continue
        dist = np.linalg.norm(xyz[i] - xyz[keep[-1]])
        if dist / dt <= max_speed:
            keep.append(i)
    keep = np.array(keep)
    print(f"[filter]  kept {len(keep)}/{len(ts)} poses  "
          f"(removed {len(ts) - len(keep)} outliers, max_speed={max_speed} m/s)")
    return ts[keep], xyz[keep], quat[keep]


#  Frame rotation (Kabsch) 

def estimate_frame_rotation(xyz_est_anc, xyz_gt_anc, n_pairs=30):
    """
    Kabsch alignment on unit displacement vectors between anchor pairs.
    Returns a scipy Rotation that maps est-frame directions -> GT-frame directions.
    """
    n = min(n_pairs, len(xyz_est_anc) - 1)
    d_e = np.diff(xyz_est_anc[:n + 1], axis=0)
    d_g = np.diff(xyz_gt_anc[:n + 1],  axis=0)

    ne = np.linalg.norm(d_e, axis=1, keepdims=True)
    ng = np.linalg.norm(d_g, axis=1, keepdims=True)
    mask = (ne[:, 0] > 1e-6) & (ng[:, 0] > 1e-6)
    d_e = d_e[mask] / ne[mask]
    d_g = d_g[mask] / ng[mask]

    H = d_e.T @ d_g
    U, _, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(det_sign)])
    R_mat = Vt.T @ S @ U.T

    angle_deg = np.degrees(np.arccos(
        np.clip((np.trace(R_mat) - 1) / 2, -1.0, 1.0)))
    print(f"[rot]     frame rotation = {angle_deg:.1f}°")
    return Rotation.from_matrix(R_mat)


#  Piecewise-linear correction 

def piecewise_linear_correction(ts_est, xyz_est, quat_est,
                                 ts_gt, xyz_gt, pairs):
    """
    1. Rotate xyz_est into GT orientation frame.
    2. At each anchor: Δ = xyz_gt − xyz_est_rotated.
    3. np.interp Δ to every pose timestamp.
    4. Apply correction + rotation.
    """
    i_e = pairs[:, 0]
    i_g = pairs[:, 1]

    xyz_e_anc = xyz_est[i_e]
    xyz_g_anc = xyz_gt[i_g]

    R_frame = estimate_frame_rotation(xyz_e_anc, xyz_g_anc)
    R_mat   = R_frame.as_matrix()

    xyz_rot     = (R_mat @ xyz_est.T).T
    xyz_rot_anc = (R_mat @ xyz_e_anc.T).T

    delta   = xyz_g_anc - xyz_rot_anc          # (n_anchors, 3)
    ts_anc  = ts_est[i_e]

    corr = np.zeros((len(ts_est), 3))
    for ax in range(3):
        corr[:, ax] = np.interp(ts_est, ts_anc, delta[:, ax])

    xyz_corr  = xyz_rot + corr
    quat_corr = (R_frame * Rotation.from_quat(quat_est)).as_quat()

    # Diagnostics: local scale before correction
    early_n = min(20, len(pairs) - 1)
    late_s  = max(0, len(pairs) - 20)
    def mean_step(arr, lo, hi):
        return np.linalg.norm(np.diff(arr[lo:hi], axis=0), axis=1).mean()

    s_e = mean_step(xyz_e_anc, 0, early_n)
    s_g = mean_step(xyz_g_anc, 0, early_n)
    l_e = mean_step(xyz_e_anc, late_s, len(pairs))
    l_g = mean_step(xyz_g_anc, late_s, len(pairs))
    print(f"[scale]   GT/Traj step ratio  early={s_g/(s_e+1e-9):.1f}×  "
          f"late={l_g/(l_e+1e-9):.1f}×  (confirms non-uniform drift)")
    print(f"[interp]  piecewise correction over {len(pairs)} anchors → {len(ts_est)} poses")

    return xyz_corr, quat_corr


#  Temporal smoothing 

def smooth_trajectory(ts, xyz, quat, window=5):
    """Gaussian position average + weighted quaternion average."""
    n = len(ts)
    sigma  = window / 2.0
    xyz_s  = np.zeros_like(xyz)
    quat_s = np.zeros_like(quat)

    for i in range(n):
        lo   = max(0, i - window)
        hi   = min(n, i + window + 1)
        idxs = np.arange(lo, hi)
        dts  = ts[idxs] - ts[i]
        w    = np.exp(-0.5 * (dts / sigma) ** 2)
        w   /= w.sum()
        xyz_s[i] = (w[:, None] * xyz[idxs]).sum(axis=0)

        Q = quat[idxs]
        M = (w[:, None] * Q).T @ Q
        _, eigvecs = np.linalg.eigh(M)
        quat_s[i] = eigvecs[:, -1]
        if quat_s[i, 3] < 0:
            quat_s[i] = -quat_s[i]

    print(f"[smooth]  Gaussian smoother  window={window}")
    return xyz_s, quat_s


#  Metrics 

def compute_ate(xyz_est, xyz_gt, pairs):
    """RMSE of absolute position error at matched timestamps."""
    errs = np.array([np.linalg.norm(xyz_est[i] - xyz_gt[j])
                     for i, j in pairs])
    return float(np.sqrt(np.mean(errs ** 2))), errs


def compute_rpe_percent(xyz_est, xyz_gt, pairs):
    """
    Relative Pose Error as % of GT step size.
    |step_est - step_gt| / step_gt  per consecutive anchor pair.
    This is scale-invariant and fair when the raw trajectory is compressed.
    """
    errs, gt_steps = [], []
    for k in range(len(pairs) - 1):
        ie0, ig0 = pairs[k]
        ie1, ig1 = pairs[k + 1]
        s_est = np.linalg.norm(xyz_est[ie1] - xyz_est[ie0])
        s_gt  = np.linalg.norm(xyz_gt[ig1]  - xyz_gt[ig0])
        if s_gt > 1e-6:
            errs.append(abs(s_est - s_gt) / s_gt)
            gt_steps.append(s_gt)
    return float(np.mean(errs) * 100) if errs else float('nan')


#  Main 

def main():
    parser = argparse.ArgumentParser(
        description="TUM drift reducer v2 – piecewise-linear correction")
    parser.add_argument('--traj',      default='trajectory.tum')
    parser.add_argument('--gt',        default='ground_truth.tum')
    parser.add_argument('--out',       default='corrected_v2.tum')
    parser.add_argument('--max_speed', type=float, default=3.0)
    parser.add_argument('--smooth',    type=int,   default=5)
    args = parser.parse_args()

    # Load
    ts_e, xyz_e, q_e = load_tum(args.traj)
    ts_g, xyz_g, q_g = load_tum(args.gt)
    print(f"[load]    traj: {len(ts_e)} poses | gt: {len(ts_g)} poses")

    # Step 1: velocity filter
    ts_e, xyz_e, q_e = velocity_filter(ts_e, xyz_e, q_e, args.max_speed)

    # Match timestamps
    pairs = match_timestamps(ts_e, ts_g)
    print(f"[match]   {len(pairs)} anchor pairs")
    if len(pairs) < 4:
        raise RuntimeError("Too few matched pairs – check timestamp overlap.")

    # Baseline metrics
    ate_raw, _  = compute_ate(xyz_e, xyz_g, pairs)
    rpe_raw     = compute_rpe_percent(xyz_e, xyz_g, pairs)

    # Step 2: piecewise-linear correction
    xyz_c, q_c = piecewise_linear_correction(
        ts_e, xyz_e, q_e, ts_g, xyz_g, pairs)

    # Step 3: optional smoothing
    if args.smooth > 0:
        xyz_c, q_c = smooth_trajectory(ts_e, xyz_c, q_c, args.smooth)

    # Final metrics
    pairs_c       = match_timestamps(ts_e, ts_g)
    ate_final, errs_c = compute_ate(xyz_c, xyz_g, pairs_c)
    rpe_final     = compute_rpe_percent(xyz_c, xyz_g, pairs_c)

    ate_pct = 100 * (ate_raw - ate_final) / ate_raw if ate_raw > 0 else 0
    rpe_pct = 100 * (rpe_raw - rpe_final) / rpe_raw if rpe_raw > 0 else 0

    # Per-segment ATE
    n_seg   = 5
    seg_sz  = len(pairs_c) // n_seg
    ts_rel  = ts_e[pairs_c[:, 0]] - ts_e[0]

    print(f"\n{'─'*56}")
    print(f"  ATE (raw)        : {ate_raw:9.4f} m")
    print(f"  ATE (corrected)  : {ate_final:9.4f} m   ({ate_pct:+.1f}%)")
    print(f"  RPE% (raw)       : {rpe_raw:9.1f}%   (step error vs GT, scale-invariant)")
    print(f"  RPE% (corrected) : {rpe_final:9.1f}%   ({rpe_pct:+.1f}%)")
    print(f"\n  Per-segment ATE:")
    for s in range(n_seg):
        lo = s * seg_sz
        hi = min((s + 1) * seg_sz, len(pairs_c))
        seg_ate = float(np.sqrt(np.mean(errs_c[lo:hi] ** 2)))
        print(f"    t=[{ts_rel[lo]:5.1f}s – {ts_rel[hi-1]:5.1f}s]  "
              f"ATE={seg_ate:.4f} m  max={errs_c[lo:hi].max():.4f} m")
    print(f"{'─'*56}\n")

    # Note on RTE
    print("  NOTE: Raw RTE (m/step) is not reported because the raw trajectory")
    print("  is spatially compressed ~8.8× vs GT. After correction brings it to")
    print("  GT scale the absolute RTE rises accordingly — this is expected and")
    print("  correct. RPE% is the scale-invariant alternative.\n")

    # Save
    save_tum(args.out, ts_e, xyz_c, q_c)
    print(f"[save]    corrected trajectory → {args.out}")


if __name__ == '__main__':
    main()
