from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=5)
    )

model = YOLOWorld('yolov8s-world.pt')

model.set_classes(["person", "road", "building", "path"])

image_path = "output/20.png"
results = model(image_path, imgsz=256)

fig, ax = plt.subplots(1, 2, figsize=(30, 30))
image = np.array(Image.open(image_path))
for result in results:
    print(result.boxes.xyxy)
    for box in result.boxes.xyxy:
        show_box(box.cpu(), ax[0])
ax[0].imshow(image)
plt.show()