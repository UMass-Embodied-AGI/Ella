from PIL import Image
import torchvision.transforms as TS
import torch
import time
import numpy as np
from sg.builder.model import RAMWrapper

ram = RAMWrapper()

raw_image = np.array(Image.open("output/newyork_agents_num_25_with_schedules/ella/ego/Adam Pierce/rgb_000050.png").convert("RGB"))
for i in range(10):
    start_time = time.time()
    res = ram.predict(raw_image)
    print("Time for RAM:", time.time() - start_time)

for i in [2, 4, 8, 16]:
    start_time = time.time()
    res = ram.predict([raw_image] * i)
    print("Time for RAM:", time.time() - start_time)
print(res)