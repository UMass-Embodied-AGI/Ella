import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sg.builder.model import DINOWrapper

image = np.array(Image.open("output/newyork_agents_num_25_with_schedules/ella/ego/Adam Pierce/rgb_000050.png").convert("RGB"))
TEXT_PROMPT = "road . path . building . person . bridge . bin ."
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

model = DINOWrapper()

for i in range(10):
    start_time = time.time()
    with torch.autocast("cuda"):
        model.predict(image, TEXT_PROMPT)
    print("Time for GroundingDINO:", time.time() - start_time)

for i in [2, 4, 8, 16]:
    start_time = time.time()
    with torch.autocast("cuda"):
        model.predict([image] * i, [TEXT_PROMPT] * i)
    print("Time for GroundingDINO:", time.time() - start_time)