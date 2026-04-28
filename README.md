# easyhec-calib

Standalone hand-eye camera calibration around [simple-easyhec](https://github.com/StoneT2000/simple-easyhec).

Given a directory of multi-view images with joint readings, a robot URDF, and an initial camera-pose guess, refine the camera extrinsic by silhouette matching.

## Install

Install PyTorch first, matched to your CUDA toolkit. `nvdiffrast` (pulled in by `easyhec`) JIT-compiles CUDA extensions, so the torch CUDA version must match the local `nvcc`. Example for CUDA 12.8:

```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install `easyhec` (the underlying optimizer) one of three ways:

1. Pip from git (default in `pyproject.toml`):
   ```
   pip install -e .
   ```
2. Editable local checkout (when you want to patch easyhec):
   ```
   pip install -e ../simple-easyhec
   pip install -e . --no-deps
   pip install numpy opencv-python trimesh urchin
   ```
3. Git submodule:
   ```
   git submodule add https://github.com/StoneT2000/simple-easyhec third_party/simple-easyhec
   pip install -e third_party/simple-easyhec
   pip install -e . --no-deps
   ```

```
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

## Data layout

```
data_dir/
├── meta.json
├── frame_000.png      frame_000.qpos.npy   [frame_000.mask.png]
├── frame_001.png      frame_001.qpos.npy   [frame_001.mask.png]
...
```

`meta.json`:
```json
{
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "width": 1280,
  "height": 720,
  "joint_names": ["joint1", "joint2", "..."],
  "gt_cam_in_world": [[...]]
}
```

`gt_cam_in_world` is optional; only used for printing error metrics.

`qpos.npy` is a 1D float vector ordered to match `meta["joint_names"]`. The URDF's actuated-joint subset is selected by name.

## Usage

```
easyhec-calib refine \
  --data_dir   /path/to/data \
  --urdf       /path/to/robot.urdf \
  --init_pose  /path/to/init_cam_in_world.npy \
  [--exclude_link_prefixes finger_,palm,wrist_mount,wrist_base] \
  [--use_sam2 --sam2_model facebook/sam2-hiera-large] \
  [--iterations 2000 --lr 3e-3 --early_stop 200] \
  [--output refined_cam_in_world.npy] \
  [--save_video out.mp4 --no_vis]
```

`--init_pose` must be a `.npy` of a 4×4 cam-in-world matrix.

## Library

```python
import numpy as np
from easyhec_calib import load_dataset, refine

meta, frames = load_dataset("data_dir")
T_init = np.load("init_cam_in_world.npy")
T_refined = refine(
    meta=meta, frames=frames,
    urdf_path="robot.urdf",
    init_cam_in_world=T_init,
    exclude_link_prefixes=("finger_", "palm"),
)
```

## Trouble Shooting

### `nvdiffrast` build fails with `cuda_runtime_api.h: No such file or directory`

PyTorch's JIT extension builder doesn't pass `CXXFLAGS` through to its ninja rules, so the `g++` step misses the conda CUDA toolkit's include path (only the `nvcc` step picks it up). `g++` honors `CPATH`, so set it after activating the env:

```
conda activate <env>
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include
```

To make it persistent:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include' \
  > $CONDA_PREFIX/etc/conda/activate.d/zz-cuda-cpath.sh
```

Re-activate the env, then clear the stale build cache before retrying:

```
rm -rf ~/.cache/torch_extensions/*/nvdiffrast_plugin
```
