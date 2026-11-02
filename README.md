# MeshGS-SLAM — Running on Replica

## Prerequisites

Conda environment `mini_splatting2` must be set up. If not:

```bash
conda env create -f environment.yml
conda activate mini_splatting2
pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/
pip install submodules/diff-gaussian-rasterization_ms/
pip install submodules/fused-ssim/
```

## Data Layout

```
mini-splatting2/
├── data/
│   ├── Replica/
│   │   ├── cam_params.json          # camera intrinsics (required)
│   │   ├── office0/
│   │   │   ├── mav0/rgb2/data/      # frame000000.jpg … frame001999.jpg
│   │   │   ├── mav0/rgb2/data.csv   # timestamp → filename
│   │   │   ├── mesh_kf*.ply         # incremental OKVIS2 meshes
│   │   │   └── okvis2-slam_trajectory.csv
│   │   ├── office1/ … office4/
│   │   └── room0/ … room2/
│   └── TartanAir/
│       ├── cam_params.json
│       └── TartanAir_AbandonedFactory_Easy006/
│           ├── rgb2/data/
│           ├── rgb2/data.csv
│           ├── mesh_kf*.ply
│           └── okvis2-slam_trajectory.csv
```

`cam_params.json` must exist in the dataset root (parent of the scene folder).
Replica example:
```json
{"camera": {"fx": 600.0, "fy": 600.0, "w": 1200, "h": 680, "cx": 599.5, "cy": 339.5}}
```

## Single Scene

```bash
conda run --no-capture-output -n mini_splatting2 python msv2/train_online2.py \
    --scene  data/Replica/office0 \
    --output output/office0 \
    --no_viewer
```

No camera args needed — they are loaded automatically from `cam_params.json`.
RGB paths (`mav0/rgb2/` vs `rgb2/`) are also auto-detected.

## Full Replica Benchmark (all 8 scenes)

```bash
bash run_eval.sh output/eval_v8
```

This trains all scenes sequentially, then runs `final_eval.py` over all scenes and prints
a comparison table vs. GSFusion, LoopSplat, SEGS-SLAM, etc.

Optional second argument overrides the data root (default: `data/Replica`):
```bash
bash run_eval.sh output/eval_v8 data/Replica
```

## Evaluate an Existing Checkpoint

```bash
# Single scene
conda run --no-capture-output -n mini_splatting2 python meshgs_slam/final_eval.py \
    --output output/eval_v8/office0 \
    --scene  data/Replica/office0

# All 8 Replica scenes
conda run --no-capture-output -n mini_splatting2 python meshgs_slam/final_eval.py \
    --output_root output/eval_v8 \
    --data_root   data/Replica \
    --save_json
```

## TartanAir

```bash
conda run --no-capture-output -n mini_splatting2 python msv2/train_online2.py \
    --scene   data/TartanAir/TartanAir_AbandonedFactory_Easy006 \
    --output  output/tartanair/AbandonedFactory_Easy006 \
    --voxel_size 0.05 \
    --blend_max_gaussians 50000 \
    --no_viewer
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iters_per_keyframe` | 3 | Gradient steps per keyframe |
| `--voxel_size` | 0.01 | Mesh voxel downsampling (use 0.05 for outdoor) |
| `--blend_max_gaussians` | 0 | Cap boundary Gaussians for blend (use 50000 for outdoor) |
| `--no_boundary_blend` | off | Ablation: disable boundary blending |
| `--no_mesh_densify` | off | Ablation: disable mesh-error densification |
| `--no_viewer` | off | Disable real-time viser viewer |
| `--traj` | `okvis2-slam_trajectory.csv` | Which trajectory file to use |

## TensorBoard

```bash
tensorboard --logdir output/
```

Logs: `train/psnr`, `train/pixel_loss`, `train/num_gaussians`, `eval/psnr`, `eval/mapping_fps`.

## Expected Results (eval_v8, GSFusion protocol, all frames)

| Scene | PSNR | SSIM | LPIPS | FPS |
|-------|------|------|-------|-----|
| office0 | 39.41 | 0.971 | 0.112 | 38.3 |
| office1 | 38.15 | 0.957 | 0.200 | 35.6 |
| room0 | 30.84 | 0.907 | 0.161 | 33.6 |
| room1 | 31.14 | 0.920 | 0.164 | 31.6 |
| room2 | 33.96 | 0.944 | 0.162 | 33.4 |
| **mean (5 scenes)** | **34.70** | **0.940** | **0.160** | **34.5** |
