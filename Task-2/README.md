
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
- catkin_ws
- FAST-LIVO2 
- Ubuntu-20.04Trajectory Metrics Evaluation: evo (Python Trajectory Evaluation Pack) 
- Trajectory Metrics Evaluation: evo (Python Trajectory Evaluation Pack) 






