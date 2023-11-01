import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from detectron2.utils.visualizer import GenericMask
import matplotlib.pyplot as plt

with open("dataset_output_part1.json", "r") as f:
    data = json.load(f)

converted = []
i = 0
for image_id, img_data in enumerate(data):
    segmentation = [s for s in img_data["segmentation"] if len(s) > 0]
    if len(segmentation) <= 0:
        continue
    img = cv2.imread(img_data["file_name"])
    width, height = Image.open(img_data["file_name"]).size
    mask = GenericMask(segmentation, height, width).mask
    os.makedirs("output/convert", exist_ok=True)
    cv2.imwrite(f"output/convert/{image_id}.jpg", mask*255)
    for bbox in tqdm(img_data["bbox"]):
        i += 1
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        sub_mask = np.zeros_like(mask)
        sub_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        polygons = GenericMask(sub_mask, height, width).polygons
        # length = [len(p) for p in polygons]
        area = np.sum([np.sum(cv2.resize(GenericMask([p], height, width).mask[y1:y2, x1:x2], (128, 128))) for p in polygons])
        # fig = plt.figure(figsize=(8,4))
        # ax = fig.subplots(ncols=2, nrows=1)
        # ax[0].imshow(img[y1:y2, x1:x2, ...])
        # ax[1].imshow(mask[y1:y2, x1:x2], cmap="gray")
        # os.makedirs("output/convert", exist_ok=True)
        # fig.savefig(f"output/convert/{i}.png")
        # plt.close(fig)
        converted_img = {
            "id": i,
            "image_id": image_id,
            "segmentation": [p.tolist() for p in polygons],
            "category_id": 4,
            "category_name": "plane",
            "iscrowd": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "file_name": os.path.basename(img_data["file_name"]),
            "area": int(area),
        }
        converted.append(converted_img)

with open("dataset_converted_part1.json", "w", encoding="utf8") as f:
    data = json.dump(converted, f, indent=2)
