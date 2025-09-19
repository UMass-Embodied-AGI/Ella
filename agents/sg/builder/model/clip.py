from typing import List, overload, Union
import numpy as np
import torch
import open_clip
from PIL import Image
from itertools import chain

class CLIPWrapper:
    def __init__(self, device='cuda', model="ViT-B-32-256", tag="datacomp_s34b_b86k"):
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

if __name__ == "__main__":
    # Example usage
    clip_model = CLIPWrapper()
    a1 = np.array(Image.open("/work/pi_chuangg_umass_edu/icefox/Ella/output/CF/IB/DETROIT_agents_num_15/ella/curr_sim/Chad Thompson/semantic_memory/object/obj_000002/appearance.png"))
    a2 = np.array(Image.open("/work/pi_chuangg_umass_edu/icefox/Ella/output/CF/IB/DETROIT_agents_num_15/ella/curr_sim/Chad Thompson/semantic_memory/object/obj_000003/appearance.png"))
    img_features = clip_model.predict_image([a1, a2])
    # text_features = clip_model.predict_text(["person", "road", "bus", "building"])
    similarity = np.dot(img_features, img_features.T)
    Image.fromarray(a1).resize((256, 256), Image.BICUBIC).save("a1.png")
    Image.fromarray(a2).resize((256, 256), Image.BICUBIC).save("a2.png")
    print(f"Image similarity: {similarity}")
    # Plotting for visual comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(a1)
    axes[1].imshow(a2)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("comparison.png")

    # character_name_to_skin_info = json.load(open(f"ViCo/assets/character2skin.json", 'r'))
    # character_name_to_image_features = {}
    # for file in os.listdir("ViCo/assets/imgs"):
    #     if file.endswith(".png"):
    #         rgbs = [np.array(Image.open(f"ViCo/assets/imgs/{file}"))]
    #         image_features = clip_model.predict_image(rgbs)
    #         character_name = file.split(".png")[0].replace("_", " ")
    #         character_name_to_image_features[character_name] = image_features[0]
    #         if character_name not in character_name_to_skin_info:
    #             print(f"Processed {character_name}, not in character2skin.json")
    # for name in character_name_to_skin_info:
    #     if name not in character_name_to_image_features:
    #         print(f"{name} not in image features")
    # with open("ViCo/assets/character_name_to_image_features.pkl", "wb") as f:
    #     pickle.dump(character_name_to_image_features, f)
