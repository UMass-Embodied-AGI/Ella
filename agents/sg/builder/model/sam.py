import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from .utils import check_download_to, get_device_type

class SAMWrapper:
    def __init__(self, device='cuda', ckpt_path="agents/sg/third_party/EfficientSAM/weights/efficient_sam_vitt.pt"):
        self.device = device
        self.ckpt_path = ckpt_path
        check_download_to("https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt", self.ckpt_path)
        self.model = build_efficient_sam_vitt(self.ckpt_path).to(device)
    
    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, boxes: np.ndarray, annotate=False) -> np.ndarray:
        # boxes: [N, 4] (x0, y0, x1, y1)
        img_tensor = ToTensor()(Image.fromarray(rgb)).to(self.device)
        pts_sampled = np.stack([boxes[:, :2], boxes[:, 2:]], axis=1)
        pts_sampled = torch.tensor(pts_sampled).unsqueeze(0).to(self.device)
        pts_labels = torch.tensor([[2, 3]] * boxes.shape[0]).unsqueeze(0).to(self.device)
        with torch.autocast(get_device_type(self.device)):
            predicted_logits, predicted_iou = self.model(
                img_tensor[None, ...],
                pts_sampled,
                pts_labels,
            )

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        masks = torch.ge(predicted_logits[0, :, 0, :, :], 0).cpu().detach().numpy()
        if not annotate:
            return masks
        else:
            ann = rgb[None].repeat(masks.shape[0], axis=0).astype(np.float64)
            ann[~masks] += (255 - ann[~masks]) / 2
            ann = ann.astype(np.uint8)
            return masks, ann