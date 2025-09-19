from typing import List, overload
import numpy as np
import torch
import torchvision.transforms as TS
from PIL import Image
import time

from ram import inference_ram, inference_tag2text
from ram.models import ram, ram_plus
from .utils import check_download_to, get_device_type

class RAMWrapper:
    def __init__(self, device='cuda', ckpt_path="agents/sg/third_party/recognize-anything/weights/ram_plus_swin_large_14m.pth"):
        self.device = device
        self.ckpt_path = ckpt_path
        check_download_to("https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth", self.ckpt_path)
        self.model = ram_plus(pretrained=self.ckpt_path, image_size=384, vit="swin_l").eval().to(device) # default threshold is 0.68
        self.transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @overload
    def predict(self, rgb: List[np.ndarray]) -> List[List[str]]:
        ...

    def predict(self, rgb: np.ndarray) -> List[str]:
        unbatched = False
        if isinstance(rgb, np.ndarray):
            rgb = [rgb]
            unbatched = True
        
        image = torch.stack([self.transform(Image.fromarray(img)) for img in rgb]).to(self.device)
        with torch.no_grad():
            with torch.autocast(get_device_type(self.device)):
                tags, tags_chinese = self.model.generate_tag(image)
        ret = [[t.strip() for t in tag.split("|")] for tag in tags]
        if unbatched:
            return ret[0]
        return ret

if __name__ == "__main__":
    model = RAMWrapper()
    image_path = "ViCo/assets/imgs/indoor_scenes/dormitory-0_ego.png"
    image = np.array(Image.open(image_path).convert("RGB"))
    print(model.predict(image))