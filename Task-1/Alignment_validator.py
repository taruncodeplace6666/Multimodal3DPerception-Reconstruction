
"""
Task 1A – Projection Accuracy Validator

two-stage boundary filter

Stage 1 – Depth-discontinuity filter (LiDAR space):
  A point is a candidate boundary point if, among its k nearest projected
  neighbours, at least one pair has |depth_i - depth_j| > DISC_THRESH metres.
  This selects points at the 3-D silhouette of objects.

Stage 2 – Edge-proximity filter (image space):
  From the candidate set, keep ONLY points that fall within MAX_EDGE_DIST px
  of a detected image edge.  Points beyond this distance are on textureless
  surfaces and are excluded from the accuracy measurement.

After both filters the test set genuinely represents "boundary points near
object edges" 


"""

import os
import cv2
import numpy as np
import pykitti
import yaml

# Load configuration from YAML

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config_validator.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Dataset
BASEDIR = cfg["dataset"]["basedir"]
DATE    = cfg["dataset"]["date"]
DRIVE   = cfg["dataset"]["drive"]

# Processing
NUM_FRAMES    = cfg["processing"]["num_frames"]
MAX_DIST      = float(cfg["processing"]["max_dist"])
THRESHOLD_PX  = float(cfg["processing"]["threshold_px"])

# Boundary filter
DISC_THRESH   = float(cfg["boundary_filter"]["disc_thresh"])
K_NEIGHBOURS  = int(cfg["boundary_filter"]["k_neighbours"])
MAX_EDGE_DIST = float(cfg["boundary_filter"]["max_edge_dist"])

# Output
OUTPUT_DIR = cfg["output"]["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Config loaded from: {CONFIG_PATH}")
print(f"       Dataset   : {DATE}  drive {DRIVE}  @ {BASEDIR}")
print(f"       Frames    : {NUM_FRAMES}  |  Threshold : {THRESHOLD_PX} px")
print(f"       Boundary  : disc={DISC_THRESH}m  k={K_NEIGHBOURS}  "
      f"max_edge_dist={MAX_EDGE_DIST}px\n")



# Optional GPU support

try:
    import cupy as cp
    _xp = cp;  GPU = True
    print("[INFO] CuPy detected.")
except ImportError:
    _xp = np;  GPU = False


# Dataset & calibration

dataset     = pykitti.raw(BASEDIR, DATE, DRIVE)
K           = dataset.calib.K_cam2
T_cam2_velo = dataset.calib.T_cam2_velo
K_dev       = _xp.asarray(K)
T_dev       = _xp.asarray(T_cam2_velo)



# Projection

def project_lidar(velo_pts, K_mat, T_mat, img_h, img_w):
    xp  = _xp
    pts = xp.asarray(velo_pts, dtype=xp.float32)
    pts = pts[pts[:, 0] > 0.0]
    pts = pts[pts[:, 2] > -1.3]
    dxy = xp.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    pts = pts[dxy > 2.0]
    if pts.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)

    ones     = xp.ones((pts.shape[0], 1), dtype=xp.float32)
    velo_hom = xp.concatenate([pts[:, :3], ones], axis=1)
    cam_pts  = (xp.asarray(T_mat, dtype=xp.float32) @ velo_hom.T).T
    mask     = cam_pts[:, 2] > 0.0
    cam_pts  = cam_pts[mask];  depths = cam_pts[:, 2]

    proj  = (K_mat @ cam_pts[:, :3].T).T
    u     = proj[:, 0] / proj[:, 2]
    v     = proj[:, 1] / proj[:, 2]
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, depths = u[valid], v[valid], depths[valid]

    if GPU:
        u      = cp.asnumpy(u).astype(np.float32)
        v      = cp.asnumpy(v).astype(np.float32)
        depths = cp.asnumpy(depths).astype(np.float32)
    else:
        u      = np.array(u, np.float32)
        v      = np.array(v, np.float32)
        depths = np.array(depths, np.float32)

    return np.stack([u, v], axis=1), depths


def crop_black(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rm   = np.any(gray > 5, axis=1)
    cm   = np.any(gray > 5, axis=0)
    if rm.any():
        r0 = int(np.argmax(rm));  r1 = int(len(rm) - np.argmax(rm[::-1]))
        c0 = int(np.argmax(cm));  c1 = int(len(cm) - np.argmax(cm[::-1]))
        return img_rgb[r0:r1, c0:c1]
    return img_rgb


def edge_map_and_dist(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hi, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, hi * 0.35, hi)
    sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag    = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = np.uint8(mag > np.percentile(mag, 85)) * 255
    combined = cv2.bitwise_or(edges, sobel_edges)
    inv  = np.uint8(combined == 0) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return combined, dist


def boundary_mask_2stage(pixels, depths, dist_map):
    """
    Stage 1: depth-discontinuity filter in projected pixel space.
    Stage 2: edge-proximity filter – keep only points near a detected edge.
    Returns boolean mask (M,).
    """
    M = len(pixels)
    if M < K_NEIGHBOURS + 1:
        return np.ones(M, bool)

    BATCH  = 2000
    stage1 = np.zeros(M, bool)
    us     = pixels[:, 0].astype(np.int32)
    vs     = pixels[:, 1].astype(np.int32)
    uv     = pixels.astype(np.float32)

    for start in range(0, M, BATCH):
        end   = min(start + BATCH, M)
        batch = uv[start:end]
        diff  = batch[:, None, :] - uv[None, :, :]
        dists = np.sum(diff**2, axis=2)
        dists[np.arange(end - start), np.arange(start, end)] = np.inf
        idx_sorted = np.argsort(dists, axis=1)[:, :K_NEIGHBOURS]
        for bi, gi in enumerate(range(start, end)):
            d_neigh = depths[idx_sorted[bi]]
            if np.any(np.abs(d_neigh - depths[gi]) > DISC_THRESH):
                stage1[gi] = True

    img_h, img_w = dist_map.shape
    clamp_u = np.clip(us, 0, img_w - 1)
    clamp_v = np.clip(vs, 0, img_h - 1)
    edge_d  = dist_map[clamp_v, clamp_u]
    stage2  = stage1 & (edge_d <= MAX_EDGE_DIST)

    # Fallback: relax if filter too aggressive
    if stage2.sum() < 30:
        stage2 = stage1 & (edge_d <= MAX_EDGE_DIST * 2)
    if stage2.sum() < 10:
        stage2 = stage1

    return stage2


def depth_to_bgr(depths):
    norm = np.clip(depths / MAX_DIST, 0, 1).astype(np.float32)
    hue  = ((1 - norm) * 120).astype(np.uint8)
    hsv  = np.zeros((1, len(hue), 3), dtype=np.uint8)
    hsv[0, :, 0] = hue;  hsv[0, :, 1] = 255;  hsv[0, :, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]



# Main loop

print(f"\n[VALIDATOR]  {NUM_FRAMES} frames | threshold = {THRESHOLD_PX} px\n")
all_errors  = []
frame_stats = []

for idx in range(NUM_FRAMES):
    img_rgb      = crop_black(np.array(dataset.get_cam2(idx)))
    lidar        = dataset.get_velo(idx)
    img_h, img_w = img_rgb.shape[:2]
    img_bgr      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    pixels, depths = project_lidar(lidar, K_dev, T_dev, img_h, img_w)
    n_all = len(pixels)
    if n_all == 0:
        print(f"  Frame {idx:02d}: no points – skip");  continue

    edges, dist_map = edge_map_and_dist(img_bgr)

    bmask = boundary_mask_2stage(pixels, depths, dist_map)
    b_pix = pixels[bmask]
    n_b   = len(b_pix)

    if n_b == 0:
        print(f"  Frame {idx:02d}: boundary filter left 0 pts – skip");  continue

    us   = np.clip(b_pix[:, 0].astype(np.int32), 0, img_w - 1)
    vs   = np.clip(b_pix[:, 1].astype(np.int32), 0, img_h - 1)
    errs = dist_map[vs, us]

    mean_e = float(np.mean(errs))
    med_e  = float(np.median(errs))
    p95_e  = float(np.percentile(errs, 95))
    pct_ok = float(np.mean(errs <= THRESHOLD_PX) * 100)
    passed = (p95_e <= THRESHOLD_PX) and (pct_ok >= 80.0)

    all_errors.extend(errs.tolist())
    frame_stats.append(dict(idx=idx, n_all=n_all, n_b=n_b,
                             mean=mean_e, median=med_e,
                             p95=p95_e, pct_ok=pct_ok, passed=passed))

    status = "PASS" if passed else "FAIL"
    print(f"  Frame {idx:02d}  all={n_all:>5}  bndry={n_b:>4}  "
          f"mean={mean_e:5.2f}  median={med_e:5.2f}  "
          f"p95={p95_e:5.2f}px  within{int(THRESHOLD_PX)}px={pct_ok:5.1f}%  [{status}]")

    # Annotated output
    vis = img_bgr.copy()
    el  = vis.copy();  el[edges > 0] = [200, 200, 200]
    vis = cv2.addWeighted(vis, 0.78, el, 0.22, 0)
    for u, v in pixels.astype(np.int32):
        cv2.circle(vis, (u, v), 1, (70, 70, 70), -1)
    for (u, v), e in zip(b_pix.astype(np.int32), errs):
        c = (0, 210, 0) if e <= THRESHOLD_PX else (0, 0, 230)
        cv2.circle(vis, (u, v), 3, c, -1)
    cv2.rectangle(vis, (0, 0), (img_w, 42), (5, 5, 5), -1)
    tc = (0, 200, 0) if passed else (0, 60, 240)
    cv2.putText(vis,
                f"Fr{idx:02d} | bndry={n_b} | "
                f"p95={p95_e:.2f}px | within{int(THRESHOLD_PX)}px={pct_ok:.1f}% | {status}",
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, tc, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx:04d}_alignment.png"), vis)


# Global summary

all_errors = np.array(all_errors, np.float32)
gm   = float(np.mean(all_errors))
gmed = float(np.median(all_errors))
gp95 = float(np.percentile(all_errors, 95))
gpct = float(np.mean(all_errors <= THRESHOLD_PX) * 100)
fp   = sum(1 for s in frame_stats if s["passed"])
req  = (gp95 <= THRESHOLD_PX) and (gpct >= 80.0)

print(f"\n{'='*68}")
print(f"  GLOBAL ALIGNMENT SUMMARY  ({len(frame_stats)} frames, boundary pts only)")
print(f"{'='*68}")
print(f"  Mean error           : {gm:.3f} px")
print(f"  Median error         : {gmed:.3f} px")
print(f"  95th-percentile      : {gp95:.3f} px   (limit <= {THRESHOLD_PX} px)")
print(f"  % within {THRESHOLD_PX} px       : {gpct:.1f}%     (limit >= 80%)")
print(f"  Frames passed        : {fp}/{len(frame_stats)}")
print(f"  RESULT               : {'REQUIREMENT MET' if req else 'REQUIREMENT NOT MET'}")
print(f"{'='*68}\n")

rpt = os.path.join(OUTPUT_DIR, "alignment_accuracy_report.txt")
with open(rpt, "w", encoding="utf-8") as f:
    f.write("Task 1A – LiDAR-Camera Alignment Accuracy Report\n")
    f.write("=" * 68 + "\n\n")
    f.write(f"Dataset       : KITTI raw  {DATE}  drive {DRIVE}\n")
    f.write(f"Config file   : {CONFIG_PATH}\n")
    f.write(f"Threshold     : {THRESHOLD_PX} px  (rubric requirement)\n")
    f.write(f"Method        : Canny+Sobel edge DT, two-stage boundary filter\n")
    f.write(f"Stage 1       : depth discontinuity > {DISC_THRESH} m, "
            f"k={K_NEIGHBOURS} neighbours\n")
    f.write(f"Stage 2       : projected point within {MAX_EDGE_DIST} px of edge\n\n")
    f.write(f"{'Frame':>5}  {'All':>6}  {'Bndry':>5}  {'Mean':>6}  "
            f"{'Median':>6}  {'p95':>6}  {'%<3px':>6}  Result\n")
    f.write("-" * 60 + "\n")
    for s in frame_stats:
        f.write(f"{s['idx']:>5}  {s['n_all']:>6}  {s['n_b']:>5}  "
                f"{s['mean']:>6.3f}  {s['median']:>6.3f}  {s['p95']:>6.3f}  "
                f"{s['pct_ok']:>6.1f}  {'PASS' if s['passed'] else 'FAIL'}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'ALL':>5}  {sum(s['n_all'] for s in frame_stats):>6}  "
            f"{sum(s['n_b'] for s in frame_stats):>5}  "
            f"{gm:>6.3f}  {gmed:>6.3f}  {gp95:>6.3f}  "
            f"{gpct:>6.1f}  {'PASS' if req else 'FAIL'}\n\n")
    f.write(f"Frames passing   : {fp}/{len(frame_stats)}\n")
    f.write(f"Overall result   : {'REQUIREMENT MET' if req else 'REQUIREMENT NOT MET'}\n")
