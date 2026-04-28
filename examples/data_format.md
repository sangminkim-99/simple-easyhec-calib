# Calibration data format

A dataset directory consumed by `easyhec-calib refine` looks like:

```
data_dir/
├── meta.json
├── frame_000.png          frame_000.qpos.npy   [frame_000.mask.png]
├── frame_001.png          frame_001.qpos.npy   [frame_001.mask.png]
...
```

## meta.json

```json
{
  "K":      [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "width":  1280,
  "height": 720,
  "joint_names": ["joint1", "joint2", "..."],
  "gt_cam_in_world": [[...]]
}
```

Required: `K`, `width`, `height`, `joint_names`.
Optional: `gt_cam_in_world` (4×4) — only used to print error metrics.

`joint_names` defines the ordering of values in each `*.qpos.npy`. The URDF's actuated-joint subset is selected by name (joints not in `joint_names` are not supported).

## frame_NNN.png

RGB image at the same `width`×`height` as `meta.json`. (Loaded with cv2 + BGR→RGB conversion.)

## frame_NNN.qpos.npy

1D float vector matching `meta.joint_names` ordering.

## frame_NNN.mask.png (optional)

Cached robot silhouette: uint8 PNG, threshold 127. Without these you must pass `--use_sam2` to generate masks interactively.

## Initial pose (.npy)

Separate file (path passed via `--init_pose`) containing a 4×4 cam-in-world matrix as a numpy array.
