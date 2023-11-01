import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
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
        sub_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        polygons = GenericMask(sub_mask, height, width).polygons
        # length = [len(p) for p in polygons]
        area = [np.sum(cv2.resize(GenericMask([p], height, width).mask[y1:y2, x1:x2], (128, 128))) for p in polygons]
        max_polygon = []
        if len(area) > 0 and max(area) > 1000:
            max_polygon = polygons[np.argmax(area)].tolist()
            with open("record.txt", "a") as f:
                f.write(f"{int(max(area))}\n")
        converted_img = {
            "id": i,
            "image_id": image_id,
            "segmentation": [max_polygon],
            "category_id": 4,
            "category_name": "plane",
            "iscrowd": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "file_name": os.path.basename(img_data["file_name"]),
            "area": int(max(area)) if len(area) > 0 else 0,
        }
        converted.append(converted_img)

with open("dataset_converted_part1.json", "w", encoding="utf8") as f:
    data = json.dump(converted, f, indent=2)
