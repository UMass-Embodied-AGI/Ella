from typing import List, Tuple, overload

import numpy as np
import torch
import torchvision.transforms as TS
from torchvision.ops import box_convert
from PIL import Image

from groundingdino.util.inference import load_model, annotate as annotate_func
from .utils import check_download_to, get_device_type

def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the area of bounding boxes in xyxy format.

    Args:
        boxes (np.ndarray): Bounding boxes with shape (n, 4) or (4,).

    Returns:
        np.ndarray: Area of each bounding box.
    """
    widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return widths * heights

def box_intersection(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # Expand box1 to match the shape of boxes2
    boxes1 = np.expand_dims(box1, axis=0).repeat(boxes2.shape[0], axis=0)

    # Calculate top-left (max of x1, y1)
    tl = np.maximum(boxes1[:, :2], boxes2[:, :2])

    # Calculate bottom-right (min of x2, y2)
    br = np.minimum(boxes1[:, 2:], boxes2[:, 2:])

    # Concatenate top-left and bottom-right
    return np.concatenate([tl, br], axis=1)


def _remove_overlap_bboxes(bboxes: np.ndarray) -> np.ndarray:
    n = bboxes.shape[0]
    mask = np.ones(n, dtype=bool)  # Initialize mask with True
    areas = box_area(bboxes)  # Calculate areas of all bounding boxes

    for i in range(n):
        # Compute intersection areas between the current box and all others
        intersection_areas = box_area(box_intersection(bboxes[i], bboxes))
        # Custom logic: Keep if the area of the current box is less than or equal to other areas
        not_filtered = (areas[i] <= areas) | (intersection_areas / areas < 0.8)
        # Update the mask
        mask = mask & not_filtered

    return mask


class DINOWrapper:
    def __init__(self, device='cuda', ckpt_path="agents/sg/third_party/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth", box_threshold=0.4):
        self.device = device
        self.ckpt_path = ckpt_path
        check_download_to("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth", self.ckpt_path)
        self.model = load_model("agents/sg/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", self.ckpt_path, device=device).to(device)
        self.transform = TS.Compose([
            TS.Resize((512, 512)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.box_threshold = box_threshold
    
    @overload
    def predict(self, rgb: List[np.ndarray], text: List[List[str]], annotate=False) -> List[Tuple[np.ndarray, List[str], np.ndarray]]:
        ...

    def predict(self, rgb: np.ndarray, text: List[str], annotate=False) -> Tuple[np.ndarray, List[str], np.ndarray]:
        unbatched = False
        if isinstance(rgb, np.ndarray):
            rgb = [rgb]
            text = [text]
            unbatched = True
        text = [" . ".join(t) + " ." for t in text]
        image: torch.Tensor = torch.stack([self.transform(Image.fromarray(img)) for img in rgb]).to(self.device)
        with torch.no_grad():
            with torch.autocast(get_device_type(self.device)):
                outputs = self.model(image, captions=text)
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()
        prediction_boxes = outputs["pred_boxes"].cpu()

        ret = []
        for i in range(len(rgb)):
            mask = prediction_logits[i].max(dim=1)[0] > self.box_threshold

            logits = prediction_logits[i][mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes[i][mask]  # boxes.shape = (n, 4)

            h, w = rgb[i].shape[:2]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            boxes = np.round(boxes).astype(np.int32).clip(0, [w, h, w, h])

            tokenizer = self.model.tokenizer
            tokenized = tokenizer(text[i])
            phrases = [tokenizer.decode([tokenized['input_ids'][logit.argmax().item()]]) for logit in logits]
            
            if not annotate:
                ret.append((boxes, phrases))
            else:
                ret.append((boxes, phrases, annotate_func(rgb[i], prediction_boxes[i][mask], logits.max(dim=1)[0], phrases)))
        if unbatched:
            return ret[0]
        return ret


if __name__ == "__main__":
    import cv2
    import os
    dino = DINOWrapper()
    rgb = cv2.imread("/work/pi_chuangg_umass_edu/icefox/Ella/output/newyork_agents_test_react_num_2/ella/curr_sim/Elon Musk/episodic_memory/img_October 01, 2024, 17:16:00.png")[:, :, ::-1]
    boxes, phrases, annotate = dino.predict(rgb, ['black', 'business suit', 'cocktail dress', 'dress', 'man', 'stand', 'suit', 'tie', 'wear'], annotate=True)
    print(boxes, phrases)
    Image.fromarray(annotate).save(f"_dino.png")