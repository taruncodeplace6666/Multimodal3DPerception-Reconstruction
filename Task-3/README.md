
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

```bash
python3 tum_to_nerfstudio.py #to convert trajectory.csv to transform.json
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


## Results
Peak Signal-to-Noise Ratio (PSNR) 19.89 on held-out test frames. 


The Fly-Through Video: A 10-second rendered MP4 video showing a smooth camera
fly-through of  generated 3D scene.

https://drive.google.com/drive/folders/13BDEYxH1l_ngtZLGJpHlGLnzsMXNd__n

## Hardware Configuration
1)CPU: AMD Ryzen AI Processor
2)RAM: 16 GB
3)GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
4)Integrated Graphics: AMD Radeon Graphics
5)Storage: NVMe SSD
6)Operating System: Ubuntu 20.04 (WSL2 on Windows 11)



