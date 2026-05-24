from tqdm import tqdm
import numpy as np
from PIL import Image
import json
import os
import pickle
import sys

current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from agents.sg.builder.model.clip import CLIPWrapper
from vico.tools.constants import ASSETS_PATH


if __name__ == "__main__":
    clip_model = CLIPWrapper()
    character_name_to_skin_info = json.load(open(os.path.join(ASSETS_PATH, "character2skin.json"), 'r'))
    character_name_to_image_features = {}
    features_path = os.path.join(ASSETS_PATH, "character_name_to_image_features.pkl")
    if os.path.exists(features_path):
        character_name_to_image_features = pickle.load(open(features_path, "rb"))
    for char_name, skin_dict in tqdm(character_name_to_skin_info.items(), desc="Processing characters"):
        rgbs = []
        for idx in range(4):
            file = os.path.join(ASSETS_PATH, "imgs", f"{char_name}_{idx}_rgb.png")
            if not os.path.exists(file):
                print(f"{char_name} not in imgs")
            rgbs.append(np.array(Image.open(file)))
        image_features = clip_model.predict_image(rgbs)
        mean_feature = image_features.mean(axis=0)
        character_name_to_image_features[char_name] = mean_feature
    pickle.dump(character_name_to_image_features, open(features_path, "wb"))
