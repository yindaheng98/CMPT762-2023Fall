import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from detectron2.utils.visualizer import GenericMask
from detectron2.structures.masks import polygons_to_bitmask

with open("dataset_output_part1.json", "r") as f:
    data = json.load(f)

converted = []
i = 0
for image_id, img_data in enumerate(data):
    segmentation = img_data["segmentation"]
    if len(segmentation) <= 0:
       continue
    width, height = Image.open(img_data["file_name"]).size
    mask = GenericMask([segmentation], height, width).mask
    for bbox in tqdm(img_data["bbox"]):
        i += 1
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        sub_mask = np.zeros_like(mask)
        sub_mask[y1:y2,x1:x2] = mask[y1:y2,x1:x2]
        polygons = GenericMask(sub_mask, height, width).polygons
        length = [len(p) for p in polygons]
        max_polygons = []
        if len(length) > 0 and max(length) >= 200:
            max_polygons = polygons[np.argmax(length)]
        converted_img = {
            "id": i,
            "image_id": image_id,
            "segmentation": [max_polygons],
            "category_id": 4,
            "category_name": "plane",
            "iscrowd": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "file_name": os.path.basename(img_data["file_name"])
        }
        converted.append(converted_img)

with open("dataset_converted_part1.json", "w", encoding="utf8") as f:
    data = json.dump(converted, f, indent=2)
