"""Console entry point: `easyhec-calib refine ...`."""

import argparse
import os
import sys

import numpy as np

from .data import load_dataset
from .overlay import render_overlays
from .refine import refine
from .viz import MaskDiffViz


def _add_refine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_dir", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--init_pose", required=True,
                   help="Path to a 4x4 cam-in-world .npy file (initial guess).")
    p.add_argument("--exclude_link_prefixes", default="",
                   help="Comma-separated link-name prefixes to drop from the URDF mesh set.")
    p.add_argument("--use_sam2", action="store_true",
                   help="Run SAM2 for any frame without a cached mask.")
    p.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    p.add_argument("--resegment", action="store_true",
                   help="Re-run SAM2 even if cached masks exist.")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--early_stop", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--vis_every", type=int, default=5)
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--save_video", default=None,
                   help="Path to save the mask-stack visualization as .mp4.")
    p.add_argument("--video_fps", type=int, default=15)
    p.add_argument("--output", default=None,
                   help="Output .npy path. Defaults to <data_dir>/refined_cam_in_world.npy.")


def _refine_cmd(args: argparse.Namespace) -> int:
    meta, frames = load_dataset(args.data_dir)
    W, H = meta["width"], meta["height"]
    print(f"Loaded {len(frames)} frames from {args.data_dir}  ({W}x{H})")

    if args.use_sam2 or args.resegment or any(f["mask"] is None for f in frames):
        if not args.use_sam2:
            missing = [f["path"] for f in frames if f["mask"] is None]
            raise SystemExit(
                f"Missing mask files for: {missing}. Provide cached *.mask.png "
                f"alongside frames or pass --use_sam2."
            )
        from .segmentation import segment_frames_with_sam2
        segment_frames_with_sam2(frames, args.sam2_model, force=args.resegment)

    for f in frames:
        pct = 100 * f["mask"].mean()
        print(f"  {os.path.basename(f['path'])}: robot pixels {pct:.1f}%")

    init_cam_in_world = np.load(args.init_pose)
    if init_cam_in_world.shape != (4, 4):
        raise SystemExit(f"--init_pose must be a 4x4 matrix; got {init_cam_in_world.shape}.")

    excludes = tuple(s for s in args.exclude_link_prefixes.split(",") if s)

    viz = None
    on_step = None
    if not args.no_vis or args.save_video:
        gt = np.stack([f["mask"].astype(np.float32) for f in frames])
        viz = MaskDiffViz(
            gt_masks=gt,
            total_iterations=args.iterations,
            show_window=not args.no_vis,
            save_video=args.save_video,
            video_fps=args.video_fps,
            every=args.vis_every,
        )
        on_step = viz

    try:
        T_refined = refine(
            meta=meta, frames=frames,
            urdf_path=args.urdf,
            init_cam_in_world=init_cam_in_world,
            exclude_link_prefixes=excludes,
            iterations=args.iterations,
            lr=args.lr,
            early_stop=args.early_stop,
            on_step=on_step,
        )
    finally:
        if viz is not None:
            viz.close()

    out = args.output or os.path.join(args.data_dir, "refined_cam_in_world.npy")
    np.save(out, T_refined)
    print(f"Saved refined extrinsic to {out}")
    return 0


def _add_overlay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_dir", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--pose", required=True,
                   help="Path to a 4x4 cam-in-world .npy file.")
    p.add_argument("--output_dir", default=None,
                   help="Directory to write *.overlay.png. Defaults to writing "
                        "next to each input frame.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Peak overlay opacity in [0, 1].")
    p.add_argument("--exclude_link_prefixes", default="",
                   help="Comma-separated link-name prefixes to drop from the URDF mesh set.")


def _overlay_cmd(args: argparse.Namespace) -> int:
    cam_in_world = np.load(args.pose)
    if cam_in_world.shape != (4, 4):
        raise SystemExit(f"--pose must be a 4x4 matrix; got {cam_in_world.shape}.")
    excludes = tuple(s for s in args.exclude_link_prefixes.split(",") if s)
    written = render_overlays(
        data_dir=args.data_dir,
        urdf_path=args.urdf,
        cam_in_world=cam_in_world,
        output_dir=args.output_dir,
        alpha=args.alpha,
        exclude_link_prefixes=excludes,
    )
    print(f"Wrote {len(written)} overlay(s):")
    for p in written:
        print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="easyhec-calib")
    sub = ap.add_subparsers(dest="cmd", required=True)
    refine_p = sub.add_parser("refine", help="Refine a camera extrinsic.")
    _add_refine_args(refine_p)
    overlay_p = sub.add_parser(
        "overlay",
        help="Render red semi-transparent URDF overlays at a given cam-in-world pose.",
    )
    _add_overlay_args(overlay_p)
    args = ap.parse_args(argv)
    if args.cmd == "refine":
        return _refine_cmd(args)
    if args.cmd == "overlay":
        return _overlay_cmd(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
