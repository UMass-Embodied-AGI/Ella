import time
import numpy as np
import torch
from PIL import Image
from .sg.builder.model import CLIPWrapper

clip = CLIPWrapper()
raw_image = np.array(Image.open("/work/pi_chuangg_umass_edu/icefox/Ella/output/DETROIT_agents_num_15_with_schedules/ella/curr_sim/Mike Tyson/episodic_memory/img_October 01, 2024, 09:00:00.png").convert("RGB"))

res = clip.predict_image(raw_image)

print(type(res), res.shape)

for i in range(10):
    start_time = time.time()
    res = clip.predict_image([raw_image] * 3)
    print(type(res), res.shape)

    break
    print("Time for CLIP:", time.time() - start_time)

for i in [2, 4, 8, 16]:
    start_time = time.time()
    res = clip.predict_image([[raw_image] * 3] * i)
    print(type(res), res[0].shape)
    break
    print("Time for CLIP:", time.time() - start_time)

while True:
    pass