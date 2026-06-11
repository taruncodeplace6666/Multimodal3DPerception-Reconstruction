# Task 1 – LiDAR-to-Camera Projection

## Overview
Projects 3D Velodyne LiDAR points onto 2D camera images using KITTI raw
dataset calibration matrices. Outputs colour-coded depth overlays and a
validation report.

---

## Files

| File | Purpose |
|------|---------|
| `Spatial_projection.py` | Main projection script |
| `Alignment_validator.py` | Alignment accuracy validator (rubric: ≤ 3 px error) |
| `Anomaly_detection.py` | Anomaly detection & auto calibration recovery |
| `config.yaml` | Config for `Spatial_projection.py` |
| `config_validator.yaml` | Config for `Alignment_validator` |
| `config_anomaly.yaml` | Config for `Anomaly_detection.py` |

---

## create environment

```bash
python -m venv Kitti_env
```

```bash
kitti_env\Scripts\activate
```



## Requirements

```bash
pip install -r requirements.txt
```

---



Expected folder structure:
```
data/
└── 2011_09_26/
    ├── calib_cam_to_cam.txt
    ├── calib_velo_to_cam.txt
    ├── calib_imu_to_velo.txt
    └── 2011_09_26_drive_0023_sync/
        ├── image_02/data/          
        └── velodyne_points/data/   
```

---

## How to Run

### Task 1 – Projection
```bash
python Spatial_projection.py

```

### Task 1A – Alignment Validator
```bash
python Alignment_validator.py

```

### Task 1B – Anomaly Detector
```bash
python Anomaly_detection.py

```

---

## Dependencies
Visual studio installed




# Task-2 StateEstimation(Odometry)
# KITTI FAST-LIVO2 Trajectory Drift Pipeline

A collection of tools for running FAST-LIVO2 on KITTI bags, fixing point cloud format issues, and analyzing / correcting odometry drift.





## How to Run Task-2

###  Terminal-1
```bash
roscore

```

### Terminal-2
```bash
cd ~/catkin_ws
catkin_make 
source devel/setup.bash
rosparam set use_sim_time true
roslaunch fast_livo kitti_velodyne.launch

```

### Terminal-3
```bash
rosparam set use_sim_time true
rosbag play ~/kitti_2011_09_26_drive_0023_synced.bag --clock

```

### Terminal-4
```bash
rostopic echo -p /aft_mapped_to_init > trajectory.csv

```
---



Usage:
```bash
python Reduce_drift.py \
    --traj  trajectory.tum \
    --gt    ground_truth.tum \
    --out   corrected.tum \
    --max_speed 3.0 \
    --smooth 5 \
    
```

```bash
python find_drift_window.py
#disclaimer you should specifiy the groundtruth.tum and trajectory.tum in the script
    
```

```bash
evo_ape tun ground_truth.tum trajectory.tum --align --correct_scale 

```

```bash
python3 convert.py trajectory.csv
```

---

## Typical Workflow

```
1. Launch FAST-LIVO2 with kitti_velodyne.yaml
2. Play KITTI bag → kitti_cloud_fixer.py fixes the point cloud on-the-fly
3. Record the output trajectory as trajectory.tum or trajectory.csv
4. Run find_drift_window.py to see where drift is worst
5. Run Reduce_drift.py to generate corrected.tum
```

---

## Dependencies

- Python 3.8+: `numpy`, `scipy`, `pandas`
- ROS (Noetic )
- FAST-LIVO2 
- Ubuntu-20.04


# Task-3 Novel View Synthesis (3D Gaussian Splatting)
So the above task was implemented using two approachs : converting a trajectory.tum file to transform.json, and using COLMAP's native files. so we create a two directory

## directory-1 
So the Novel View Synthesis contains all transform.json and `tum_to_nerfstudio.py
```bash
cd Novel View Synthesis
```

## directory-2 
So the gaussian-splatting contains all COLMAP's native files
```bash
cd gaussian-splatting
```

Novel View Synthesis: Contains trajectory.tum and transform.json. we have a trajectory from  (FAST-LIVO2) . the `tum_to_nerfstudio.py` that reads the TUM format, builds the necessary transformation matrices, and writes them to a transform.json file. 

gaussian-splatting : Using Native COLMAP Files. This is the most integrated approach. we will be using colmap GUI Where we extract the features and reconstruct, and  gives standard COLMAP output (like cameras.txt, images.txt, points3D.txt) and generates a transform.json file. Within this process, colmap2nerf.py handles critical transformations, such as converting COLMAP's world-to-camera pose format to the camera-to-world (c2w) format expected by tools like Instant-NGP, and adjusting for different coordinate systems 

## Requirements
Step 1: Set Up a Dedicated Conda Environment

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
```

Step 2: Install Core Libraries

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Step 3: Install Nerfstudio and Dependencies
```bash
# Install nerfstudio
pip install nerfstudio        
# Install the required backend
pip install gsplat            
# Install the nerfstudio CLI tools
ns-install-cli                
```

Step 4: Install COLMAP
```bash
sudo apt install colmap 
```

## Train Splatfacto
```bash
ns-train splatfacto \
    --vis viewer \
    --max-num-iterations 50000 \
    --pipeline.model.num-downscales 1 \
    --pipeline.model.cull-alpha-thresh 0.001 \
    --pipeline.model.densify-grad-thresh 0.0002 \
    --pipeline.datamanager.cache-images cpu \
    colmap \
    --data ~/gaussian-splatting/kitti_3dgs_project
```

## eval Splatfacto
```bash
ns-eval \
  --load-config outputs/unnamed/splatfacto/2026-06-09_174606/config.yml \
  --output-path results_174606.json
```



