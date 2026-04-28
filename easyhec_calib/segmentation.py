"""Interactive SAM2 segmentation for frames missing cached masks."""

import cv2
import numpy as np


def segment_frames_with_sam2(
    frames: list[dict],
    predictor_id: str = "facebook/sam2-hiera-large",
    force: bool = False,
) -> None:
    """Segment frames in-place; cache PNG masks alongside the RGB images."""
    import torch
    from easyhec.segmentation.interactive import InteractiveSegmentation
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(predictor_id)

    def sam2_segment(image, clicked_points_np):
        pts_xy = clicked_points_np[:, :2]
        labels = np.where(clicked_points_np[:, 2] > 0, 1, 0).astype(np.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, _, _ = predictor.predict(pts_xy, labels, multimask_output=False)
        return masks[0]

    needing = [f for f in frames if force or f["mask"] is None]
    if not needing:
        print("All frames have cached masks; skipping segmentation.")
        return

    print(f"Segmenting {len(needing)} frames with SAM2 — click pos / right-click neg, "
          "'t' to run, 't' again to accept.")
    tool = InteractiveSegmentation(segmentation_model=sam2_segment)
    new_masks = tool.get_segmentation([f["rgb"] for f in needing])
    for f, m in zip(needing, new_masks):
        f["mask"] = m.astype(bool)
        cv2.imwrite(f["stem"] + ".mask.png", (m.astype(np.uint8) * 255))
