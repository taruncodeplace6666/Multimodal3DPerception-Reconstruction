
"""
Task 1B – Anomaly Detection & Dynamic Calibration Recovery


Design

DETECTION  – four independent checks fire in parallel per frame:
  1. Projection-ratio drop : < min_proj_ratio of raw scan projects into image
  2. Centroid shift        : mean (u,v) deviates > max_centroid_shift px from
                             rolling baseline (computed from good frames only)
  3. KL-divergence         : depth histogram diverges > max_kl_div from
                             rolling baseline
  4. Edge-p95              : 95th-pct distance to nearest image edge > max_edge_p95

Any single check triggering = frame flagged as anomalous.

RECOVERY   – fully automatic, zero manual steps:
  - A deque of the last buffer_size clean-frame T matrices is maintained.
  - On anomaly: T_recovered = element-wise median of the buffer.
  - Frame is re-projected with T_recovered.
  - Buffer and rolling stats are NOT updated on anomaly frames.

ANOMALY INJECTION – to prove 100% detection:
  - Frames in anomaly_frames receive a synthetic yaw_error_rad rotation error.
  - Detection rate = (caught anomalies) / (injected anomalies) × 100.
"""

import os
import cv2
import numpy as np
import pykitti
import yaml
from collections import deque



# Load configuration from YAML

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config_anomaly.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Dataset
BASEDIR = cfg["dataset"]["basedir"]
DATE    = cfg["dataset"]["date"]
DRIVE   = cfg["dataset"]["drive"]

# Processing
NUM_FRAMES = cfg["processing"]["num_frames"]
MAX_DIST   = float(cfg["processing"]["max_dist"])

# Detection thresholds
MIN_PROJ_RATIO     = float(cfg["detection"]["min_proj_ratio"])
MAX_CENTROID_SHIFT = float(cfg["detection"]["max_centroid_shift"])
MAX_KL_DIV         = float(cfg["detection"]["max_kl_div"])
MAX_EDGE_P95       = float(cfg["detection"]["max_edge_p95"])

# Recovery
BUFFER_SIZE = int(cfg["recovery"]["buffer_size"])

# Injection
ANOMALY_FRAMES = set(cfg["injection"]["anomaly_frames"])
YAW_ERROR_RAD  = float(cfg["injection"]["yaw_error_rad"])

# Output
OUTPUT_DIR = cfg["output"]["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Config loaded from: {CONFIG_PATH}")
print(f"       Dataset   : {DATE}  drive {DRIVE}  @ {BASEDIR}")
print(f"       Frames    : {NUM_FRAMES}  |  Max depth : {MAX_DIST} m")
print(f"       Thresholds: proj_ratio={MIN_PROJ_RATIO}  centroid={MAX_CENTROID_SHIFT}px  "
      f"KL={MAX_KL_DIV}  edge_p95={MAX_EDGE_P95}px")
print(f"       Injection : frames {sorted(ANOMALY_FRAMES)}  "
      f"yaw={YAW_ERROR_RAD} rad ({np.degrees(YAW_ERROR_RAD):.1f}°)\n")



# Optional GPU support

try:
    import cupy as cp
    _xp = cp
    GPU = True
    print("[INFO] CuPy detected – GPU acceleration enabled.")
except ImportError:
    _xp = np
    GPU = False



# Dataset & calibration

dataset   = pykitti.raw(BASEDIR, DATE, DRIVE)
K         = dataset.calib.K_cam2
T_nominal = dataset.calib.T_cam2_velo.copy()
K_dev     = _xp.asarray(K)



# Helpers

def make_corrupted_T(T_base, yaw_rad=YAW_ERROR_RAD):
    """Apply a synthetic yaw rotation error to T."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_err = np.array([[ c, -s, 0, 0],
                      [ s,  c, 0, 0],
                      [ 0,  0, 1, 0],
                      [ 0,  0, 0, 1]], dtype=np.float64)
    return R_err @ T_base


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
    T_xp     = xp.asarray(T_mat, dtype=xp.float32)
    cam_pts  = (T_xp @ velo_hom.T).T
    mask     = cam_pts[:, 2] > 0.0
    cam_pts  = cam_pts[mask]
    depths   = cam_pts[:, 2]

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


def edge_distance_transform(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    hi, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, hi * 0.4, hi)
    inv   = np.uint8(edges == 0) * 255
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5)


def depth_histogram(depths, bins=20):
    hist, _ = np.histogram(depths, bins=bins, range=(0, MAX_DIST), density=True)
    return hist.astype(np.float32) + 1e-9


def kl_divergence(p, q):
    p = p / p.sum();  q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def depth_to_bgr(depths):
    norm = np.clip(depths / MAX_DIST, 0.0, 1.0).astype(np.float32)
    hue  = ((1.0 - norm) * 120.0).astype(np.uint8)
    hsv  = np.zeros((1, len(hue), 3), dtype=np.uint8)
    hsv[0, :, 0] = hue;  hsv[0, :, 1] = 255;  hsv[0, :, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]



# Rolling state

T_buffer = deque(maxlen=BUFFER_SIZE)
c_buffer = deque(maxlen=BUFFER_SIZE)   # centroid history
h_buffer = deque(maxlen=BUFFER_SIZE)   # depth-hist history

# Pre-seed with nominal calibration so checks work from frame 0
for _ in range(BUFFER_SIZE):
    T_buffer.append(T_nominal.copy())


def recover_calibration():
    """Element-wise median of buffered good-frame T matrices."""
    return np.median(np.stack(list(T_buffer), axis=0), axis=0)



# Main processing loop

print(f"\n[ANOMALY DETECTOR]  {NUM_FRAMES} frames")
print(f"  Injected anomaly frames : {sorted(ANOMALY_FRAMES)}")
print(f"  Detectors: proj_ratio | centroid_shift | KL_depth | edge_p95\n")

frame_log = []

for idx in range(NUM_FRAMES):
    img_rgb      = crop_black(np.array(dataset.get_cam2(idx)))
    lidar        = dataset.get_velo(idx)
    img_h, img_w = img_rgb.shape[:2]
    img_bgr      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    n_raw        = len(lidar)

    # Get transform (nominal or corrupted)
    T_frame = (make_corrupted_T(T_nominal)
               if idx in ANOMALY_FRAMES
               else T_nominal.copy())

    # Project with (possibly bad) T
    pixels, depths = project_lidar(lidar, K_dev, T_frame, img_h, img_w)
    n_proj = len(pixels)

    # Run all four detectors
    anomalies = []

    # 1. Projection ratio
    proj_ratio = n_proj / n_raw if n_raw > 0 else 0.0
    if proj_ratio < MIN_PROJ_RATIO:
        anomalies.append(f"proj_ratio={proj_ratio:.4f}<{MIN_PROJ_RATIO}")

    if n_proj == 0:
        anomalies.append("zero_points")
    else:
        centroid = np.array([pixels[:, 0].mean(), pixels[:, 1].mean()])

        # 2. Centroid shift
        if len(c_buffer) >= 2:
            ref   = np.mean(list(c_buffer), axis=0)
            shift = float(np.linalg.norm(centroid - ref))
            if shift > MAX_CENTROID_SHIFT:
                anomalies.append(f"centroid_shift={shift:.1f}>{MAX_CENTROID_SHIFT}px")

        # 3. KL depth divergence
        if len(h_buffer) >= 2:
            hist_now = depth_histogram(depths)
            ref_hist = np.mean(list(h_buffer), axis=0)
            kl = kl_divergence(hist_now, ref_hist)
            if kl > MAX_KL_DIV:
                anomalies.append(f"KL={kl:.3f}>{MAX_KL_DIV}")

        # 4. Edge p95
        dist_map = edge_distance_transform(img_bgr)
        us       = np.clip(pixels[:, 0].astype(np.int32), 0, img_w - 1)
        vs       = np.clip(pixels[:, 1].astype(np.int32), 0, img_h - 1)
        p95_edge = float(np.percentile(dist_map[vs, us], 95))
        if p95_edge > MAX_EDGE_P95:
            anomalies.append(f"edge_p95={p95_edge:.2f}>{MAX_EDGE_P95}px")

    is_anomaly = len(anomalies) > 0

    # Recovery (automatic)
    if is_anomaly:
        T_used         = recover_calibration()
        action         = "AUTO-RECOVERED (buffer median)"
        pixels, depths = project_lidar(lidar, K_dev, T_used, img_h, img_w)
        n_proj         = len(pixels)
    else:
        T_used = T_frame
        action = "nominal"
        T_buffer.append(T_frame.copy())
        if n_proj > 0:
            c_buffer.append(np.array([pixels[:, 0].mean(), pixels[:, 1].mean()]))
            h_buffer.append(depth_histogram(depths))

    # Console output
    flag    = "ANOMALY" if is_anomaly else "OK     "
    reasons = " | ".join(anomalies) if anomalies else "-"
    print(f"  Frame {idx:02d}  [{flag}]  proj={n_proj:>5}  {action}")
    if is_anomaly:
        print(f"           Triggers: {reasons}")

    # Save annotated frame
    overlay = img_bgr.copy()
    if n_proj > 0:
        colors = depth_to_bgr(depths)
        for (u, v), col in zip(pixels.astype(np.int32), colors):
            cv2.circle(overlay, (u, v), 2,
                       (int(col[0]), int(col[1]), int(col[2])), -1)

    bar_col = (0, 0, 180) if is_anomaly else (0, 150, 0)
    tag_txt = "ANOMALY  AUTO-RECOVERED" if is_anomaly else "OK – nominal calibration"
    cv2.rectangle(overlay, (0, 0), (img_w, 42), (10, 10, 10), -1)
    cv2.putText(overlay,
                f"Frame {idx:02d}  [{tag_txt}]  pts={n_proj:,}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_col, 2, cv2.LINE_AA)
    cv2.putText(overlay,
                f"Triggers: {reasons[:80]}",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 120, 255) if is_anomaly else (100, 180, 100), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx:04d}_anomaly.png"), overlay)

    frame_log.append(dict(idx=idx, is_anomaly=is_anomaly,
                          injected=idx in ANOMALY_FRAMES,
                          triggers=reasons, action=action, n_proj=n_proj))



# Detection-rate summary

true_positives  = sum(1 for f in frame_log if f["is_anomaly"] and f["injected"])
false_negatives = sum(1 for f in frame_log if not f["is_anomaly"] and f["injected"])
false_positives = sum(1 for f in frame_log if f["is_anomaly"] and not f["injected"])
n_injected      = len(ANOMALY_FRAMES)
detection_rate  = true_positives / n_injected * 100 if n_injected > 0 else 100.0

print(f"\n{'='*62}")
print(f"  ANOMALY DETECTION SUMMARY")
print(f"{'='*62}")
print(f"  Injected anomaly frames   : {sorted(ANOMALY_FRAMES)}")
print(f"  True positives detected   : {true_positives}/{n_injected}")
print(f"  False negatives (missed)  : {false_negatives}")
print(f"  False positives (spurious): {false_positives}")
print(f"  Detection rate            : {detection_rate:.1f}%")
print(f"  Manual interventions      : 0  (fully automatic)")
result = ("REQUIREMENT MET (100% detection, 0 manual steps)"
          if detection_rate == 100.0 else
          f"REQUIREMENT NOT MET – missed {false_negatives} frame(s)")
print(f"  Result                    : {result}")
print(f"{'='*62}\n")


 
# Write report

rpt = os.path.join(OUTPUT_DIR, "anomaly_detection_report.txt")
with open(rpt, "w", encoding="utf-8") as f:
    f.write("Task 1B – Anomaly Detection & Dynamic Calibration Recovery Report\n")
    f.write("=" * 66 + "\n\n")
    f.write(f"Dataset              : KITTI raw {DATE} drive {DRIVE}\n")
    f.write(f"Config file          : {CONFIG_PATH}\n")
    f.write(f"Injected anomalies   : frames {sorted(ANOMALY_FRAMES)}\n")
    f.write(f"Corruption type      : {YAW_ERROR_RAD} rad yaw error "
            f"({np.degrees(YAW_ERROR_RAD):.1f} deg)\n")
    f.write(f"Recovery strategy    : element-wise median of last "
            f"{BUFFER_SIZE} good-frame T matrices\n\n")
    f.write("Detection Thresholds\n")
    f.write("-" * 40 + "\n")
    f.write(f"  1. Min projection ratio : {MIN_PROJ_RATIO:.0%}\n")
    f.write(f"  2. Max centroid shift   : {MAX_CENTROID_SHIFT} px\n")
    f.write(f"  3. Max KL divergence    : {MAX_KL_DIV}\n")
    f.write(f"  4. Max edge p95         : {MAX_EDGE_P95} px\n\n")
    f.write(f"{'Frame':>5}  {'Injected':>8}  {'Detected':>8}  "
            f"{'Proj pts':>8}  {'Action':<28}  Triggers\n")
    f.write("-" * 100 + "\n")
    for fl in frame_log:
        f.write(f"{fl['idx']:>5}  {'YES' if fl['injected'] else 'no':>8}  "
                f"{'YES' if fl['is_anomaly'] else 'no':>8}  "
                f"{fl['n_proj']:>8}  {fl['action']:<28}  {fl['triggers']}\n")
    f.write("\n")
    f.write(f"Detection rate         : {detection_rate:.1f}%  "
            f"({true_positives}/{n_injected})\n")
    f.write(f"False positives        : {false_positives}\n")
    f.write(f"Manual interventions   : 0\n")
    f.write(f"Overall result         : {result}\n")

print(f"[DETECTOR] Report saved: {rpt}")