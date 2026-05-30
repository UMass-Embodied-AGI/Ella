from typing import List, overload, Union
import numpy as np
import torch
from PIL import Image
from itertools import chain

class CLIPWrapper:
    def __init__(self, device='cuda', model="ViT-B-32-256", tag="datacomp_s34b_b86k"):
        import open_clip
        self.device = device
        self.model, _, self.transform = open_clip.create_model_and_transforms(model, tag)
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.text_cache = {}
        self.embedding_dim = self.model.visual.proj.shape[1]

    def predict_image(self, rgbs: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]], normalize=True) -> Union[np.ndarray, List[np.ndarray]]:
        # unless input is batched to a list of lists of np.ndarray, return is a single np.ndarray of shape (N, 512)
        single_image = False
        if not isinstance(rgbs, list):
            rgbs = [rgbs]
            single_image = True
        unbatched = False
        if isinstance(rgbs[0], np.ndarray):
            rgbs = [rgbs]
            unbatched = True
        sp = [0]
        for i in range(len(rgbs)):
            sp.append(sp[-1] + len(rgbs[i]))

        preprocessed_image = [self.transform(Image.fromarray(rgb)) for rgb in chain(*rgbs)]
        preprocessed_image = torch.stack(preprocessed_image).to(self.device)
        with torch.no_grad():
            crop_feat = self.model.encode_image(preprocessed_image)
        if normalize:
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        if single_image:
            crop_feat = crop_feat[0]
            return crop_feat.cpu().numpy()
        if unbatched:
            return crop_feat.cpu().numpy()
        return [crop_feat[sp[i]:sp[i+1]].cpu().numpy() for i in range(len(rgbs))]
    
    def predict_text(self, text: List[str], normalize=True) -> Union[np.ndarray, List[np.ndarray]]:
        unbatched = False
        if isinstance(text[0], str):
            text = [text]
            unbatched = True
        sp = [0]
        for i in range(len(text)):
            sp.append(sp[-1] + len(text[i]))
        
        filt_text = [t for t in chain(*text) if t not in self.text_cache]
        if len(filt_text) > 0:
            tokenized_text = self.tokenizer(filt_text).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(tokenized_text)
            text_feat = text_feat.cpu().numpy()
            for t, f in zip(filt_text, text_feat):
                self.text_cache[t] = f
        ret = np.stack([self.text_cache[t] for t in chain(*text)], axis=0)
        if normalize:
            ret /= np.linalg.norm(ret, axis=1, keepdims=True)
        if unbatched:
            return ret
        return [ret[sp[i]:sp[i+1]] for i in range(len(text))]
