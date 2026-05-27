# Scene Graph Building Procedure

## Set up

Build the C++ dynamic libraries for low-level operations:

**You must set the environment variable `CUDA_HOME` before running setup.**

```bash
cd agents/sg
./setup.sh
```

This compiles `builder/builtin/libbuilder.so` and `builder/builtin/libregion.so`.

### Third-party libraries

The following third-party libraries live under `agents/sg/third_party/`:

- `EfficientSAM` — segmentation model (checkpoint auto-downloaded on first use)
- `GroundingDINO` — open-vocabulary object detection (checkpoint auto-downloaded on first use)
- `recognize-anything` — RAM+ image tagging (checkpoint auto-downloaded on first use)
- `open_clip` — CLIP feature extraction
- `segment-anything-2` — SAM 2 (alternative segmentation backend)

Model checkpoints are fetched automatically via `check_download_to` on first initialization; no manual download is needed.

## Introduction

### Object Recognition Pipeline

RAM+ → GroundingDINO → EfficientSAM

1. **RAM+** (`RAMWrapper`) — tags visible object categories from an RGB image.
2. **GroundingDINO** (`DINOWrapper`) — detects bounding boxes for each tagged category.
3. **EfficientSAM** (`SAMWrapper`) — produces per-object segmentation masks from the detected boxes.

### Volume Grid

`VolumeGridBuilder` wraps a C++ backend (`libbuilder.so`) and maintains a voxel representation of the environment. Key capabilities:

- Insert a colored, labeled point cloud frame from an RGB-D observation.
- Query voxel existence and color in $O(1)$.
- Query the surface height at any $(x, y)$ position.
- Get an occupancy map (unknown / obstacle / road) for navigation.
- A\* pathfinding via `navigate(start, goal_hull)` — returns a waypoint path or `None` if unreachable.
- Save / load the point cloud as a pickle file.
- Radius denoising and overlap queries between two grids.

Default configuration (`VolumeGridBuilderConfig`):
| Parameter | Default |
|---|---|
| `voxel_size` | 0.1 m |
| `depth_bound` | 30.0 m |
| `nav_grid_size` | 0.5 m |
| `thread_num` | 1 |

## Usage

### Builder API

```python
from agents.sg.builder.builder import Builder, BuilderConfig

builder = Builder(BuilderConfig(fov=120.0, debug=True))

# Per simulation step:
# rgb: (H, W, 3) uint8
# depth: (H, W) float32
# labels: (H, W) int32  (-1 for unlabeled pixels)
# camera_ext: (4, 4) float32 camera-to-world extrinsics
builder.add_frame(rgb, depth, labels, camera_ext)

# Navigation
path = builder.volume_grid_builder.navigate(start_xy, goal_hull_xy)

# Persistence
builder.volume_grid_builder.save("output/point_cloud.pkl")
builder.volume_grid_builder.load("output/point_cloud.pkl")
```

### Running the example

```bash
python agents/sg/example.py
```

The example initializes a Genesis simulation, loads a city mesh, walks a humanoid controller through the scene while building the volume grid, and saves `output/point_cloud.pkl`.
