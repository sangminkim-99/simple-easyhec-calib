"""Per-iteration mask diff panel + optional mp4 writer.

Builds a stacked grid (one row per frame, columns: GT mask | rendered | diff)
that the CLI uses as an `on_step` callback for `refine()`.
"""

import os
from typing import Optional

import cv2
import numpy as np


class MaskDiffViz:
    """Compose the mask-diff panel; optionally show a window and/or write mp4."""

    def __init__(
        self,
        gt_masks: np.ndarray,                 # (N, H, W) float in [0, 1]
        total_iterations: int,
        show_window: bool = True,
        save_video: Optional[str] = None,
        video_fps: int = 15,
        every: int = 5,
    ):
        self.gt = gt_masks
        self.total = total_iterations
        self.show = show_window
        self.every = every
        self.window_name = "mask: GT | rendered | diff"

        N, H, W = gt_masks.shape
        self.N, self.H, self.W = N, H, W
        if show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.writer = None
        self.save_video = save_video
        if save_video:
            d = os.path.dirname(save_video)
            if d and not os.path.exists(d):
                os.makedirs(d)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(save_video, fourcc, video_fps, (3 * W, N * H))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {save_video}")
            print(f"Writing video to {save_video} at {video_fps} fps, {3*W}x{N*H}.")

    def __call__(
        self, i: int, loss: float, best_loss: float, rendered: np.ndarray
    ) -> bool:
        if not (i % self.every == 0 or i == self.total - 1):
            return False
        rows = []
        for k in range(self.N):
            gt_u8 = (self.gt[k] * 255).astype(np.uint8)
            ren_u8 = (rendered[k] * 255).astype(np.uint8)
            diff_u8 = (np.abs(rendered[k] - self.gt[k]) * 255).astype(np.uint8)
            rows.append(np.concatenate([gt_u8, ren_u8, diff_u8], axis=1))
        panel = cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, f"{i+1}/{self.total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(panel, f"loss={loss:.0f}  best={best_loss:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        if self.writer is not None:
            self.writer.write(panel)
        if self.show:
            cv2.imshow(self.window_name, panel)
            if cv2.waitKey(1) == 27:
                print("ESC — stopping.")
                return True
        return False

    def close(self):
        if self.show:
            cv2.destroyWindow(self.window_name)
        if self.writer is not None:
            self.writer.release()
            print(f"Saved video: {self.save_video}")
