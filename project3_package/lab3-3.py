# import some common libraries
from cv2 import imshow
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import datetime
import random
import json
import cv2
import csv
import os

# import some common pytorch utilities
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.layers import nms
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
setup_logger()

'''
# This function should return a list of data samples in which each sample is a dictionary.
# Make sure to select the correct bbox_mode for the data
# For the test data, you only have access to the images, therefore, the annotations should be empty.
# Other values could be obtained from the image files.
# TODO: approx 35 lines
'''
BASE_DIR = './'

COCO_EVALUATOR_OUTPUT = "{}/COCO_output".format(BASE_DIR)
os.makedirs(COCO_EVALUATOR_OUTPUT, exist_ok=True)

IMAGE_DIR = "{}/images_output".format(BASE_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

data_dirs = '{}/data'.format(BASE_DIR)

cropped_dir = os.path.join(data_dirs, "cropped")

if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)


stride = 700
size = 1400

def get_image_dimensions(filename):
    with Image.open(filename) as img:
        return img.size  # returns (width, height)

def adjust_segmentation(segmentation, i, j):
    adjusted_segmentation = []
    for segment in segmentation:
        adjusted_segment = []
        for k in range(0, len(segment), 2):
            x = segment[k] - i
            y = segment[k + 1] - j
            adjusted_segment.extend([x, y])
        adjusted_segmentation.append(adjusted_segment)
    return adjusted_segmentation


def split_image_and_annotations(img, annotations):
    cropped_annotations = []
    data_dirs = '{}/data'.format(BASE_DIR)
    cropped_dir = os.path.join(data_dirs, "cropped")

    width, height = img.size

    index = 1
    for i in range(0, width-stride, stride):
        for j in range(0, height-stride, stride):
            if i + size > width:
                i = width - size
            if j + size > height:
                j = height - size
            
            cropped_filename = os.path.join(cropped_dir, "{}_cropped_{}.png".format(annotations["file_name"][:-4], str(index)))
            if not os.path.exists(cropped_filename):
                cropped_img = img.crop((i, j, i + size, j + size))
                cropped_img.save(cropped_filename,"PNG")
            index +=1

            x, y, w, h = annotations["bbox"]
            if x >= i and (x + w) <= (i + size) and y >= j and (y + h) <= (j + size):
                cropped_anno = annotations.copy()
                cropped_anno["bbox"] = [x - i, y - j, w, h]
                cropped_anno["segmentation"] = adjust_segmentation(annotations["segmentation"],i,j)
                #cropped_anno["segmentation"] =[[0,0]]
                cropped_anno["file_name"] = cropped_filename
                cropped_anno["id"] = annotations["id"]
                cropped_anno["position"] = [i,j]
                cropped_annotations.append(cropped_anno)
                #print(len(cropped_anno["segmentation"]))

    return cropped_annotations

def get_detection_data(set_name):
    
    dataset = []

    # Load the annotations from the JSON file
    with open(os.path.join(data_dirs, "train.json"), 'r') as f:
      data = json.load(f)

    # val_size = int(len(data) * 0.2)

    # train_data = data[:-val_size]
    # val_data = data[-val_size:]

    val_size = 6274

    train_data = data[:val_size]
    val_data = data[val_size:]

    # Determine the path to the JSON file based on the set_name
    if set_name == "train":
        annotations = train_data
    elif set_name == "all":
        annotations = data
    elif set_name == "val":
        annotations = val_data
    elif set_name == "val_test":
        annotations = val_data
        for idx, anno in  enumerate(annotations):
            filename = os.path.join(data_dirs, "train", anno["file_name"])
            width, height = get_image_dimensions(filename)
            # Check if filename already exists in dataset
            existing_record = next((record for record in dataset if record["file_name"] == filename), None)
            if existing_record is None:
                record = {
                    "file_name": filename,
                    "image_id": anno["image_id"],
                    "height": height,
                    "width": width,
                    "annotations": []
                }
                dataset.append(record)
            else:
                record = existing_record

            # If it's the test set, we don't have annotations
            if set_name != "test":
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": anno["segmentation"],
                    "category_id": anno["category_id"],
                    "iscrowd": anno["iscrowd"],
                    "area": anno["area"]
                }

                record["annotations"].append(obj)

        return dataset

    elif set_name == "test":
        test_dir = os.path.join(data_dirs, "test")
        for filename in os.listdir(test_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(test_dir, filename)
                width, height = get_image_dimensions(filepath)
                record = {
                    "file_name": filepath,
                    "image_id": filename,
                    "height": height,
                    "width": width,
                    "annotations": []
                }
                dataset.append(record)
        return dataset
    else:
        raise ValueError("Unknown set_name: {}".format(set_name))


    for anno in annotations:
        if set_name == "all" or set_name == "val":
            filename = os.path.join(data_dirs, "train", anno["file_name"])
        else:
            filename = os.path.join(data_dirs, set_name, anno["file_name"])

        img = Image.open(filename)
        cropped_anno = split_image_and_annotations(img, anno)
        
        for crop_anno in cropped_anno:
            filename_crop = crop_anno["file_name"]
            width, height = get_image_dimensions(filename_crop)
            
            # Check if filename already exists in dataset
            existing_record = next((record for record in dataset if record["file_name"] == filename_crop), None)

            if existing_record is None:
                record = {
                    "file_name": filename_crop,
                    "image_id": crop_anno["image_id"],
                    "height": height,
                    "width": width,
                    "position": crop_anno["position"],
                    "annotations": []
                }
                dataset.append(record)
            else:
                record = existing_record

            # If it's the test set, we don't have annotations
            if set_name != "test":
                obj = {
                    "bbox": crop_anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": crop_anno["segmentation"],
                    "category_id": crop_anno["category_id"],
                    "iscrowd": crop_anno["iscrowd"],
                    "area": crop_anno["area"]
                }

                record["annotations"].append(obj)

    return dataset


'''
# Remember to add your dataset to DatasetCatalog and MetadataCatalog
# Consdier "data_detection_train" and "data_detection_test" for registration
# You can also add an optional "data_detection_val" for your validation by spliting the training data
# TODO: approx 5 lines
'''
for d in ["train", "test", "val", "val_test", "all"]:
    DatasetCatalog.register("plane_" + d, lambda d=d: get_detection_data(d))
    MetadataCatalog.get("plane_" + d).set(thing_classes=["class1","class2","class3","class4","plane"])

plane_metadata = MetadataCatalog.get("plane_train")

# Load some samples from the training dataset
samples = get_detection_data("train")


'''
# Visualize some samples using Visualizer to make sure that the function works correctly
# TODO: approx 5 lines
'''
for idx, d in enumerate(random.sample(samples, 1)):
    img = cv2.imread(d["file_name"])
    # print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=plane_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    save_path = IMAGE_DIR+f"/Q1_GT_{idx}.jpg"
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print(f"Image saved to {save_path}")


'''
# Set the configs for the detection part in here.
# TODO: approx 15 lines
'''
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)

cfg.DATASETS.TRAIN = ("plane_train",)
cfg.DATASETS.TEST = ()

cfg.SOLVER.MAX_ITER = 500             # Number of training iterations
cfg.SOLVER.BASE_LR = 0.00025          # Learning rate
cfg.SOLVER.IMS_PER_BATCH = 2          # Number of images per batch
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Number of regions per image for training

# 4. Pretrain Model (Improve from 25% to 47%)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")


'''
# Create a DefaultTrainer using the above config and train the model
# TODO: approx 5 lines
'''
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()


'''
# After training the model, you need to update cfg.MODEL.WEIGHTS
# Define a DefaultPredictor
'''
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


'''
# Visualize the output for 3 random test samples
# TODO: approx 10 lines
'''

from detectron2.structures import Boxes
dataset_dicts = get_detection_data("val")

all_predictions = []

for idx, d in enumerate(dataset_dicts):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    file_number = d["file_name"][16:21]
    #graph_place = d["file_name"][30:-4]

    scores = outputs["instances"].scores
    keep = scores>0.6
    outputs["instances"] = outputs["instances"][keep]

    dx, dy = d["position"]
    boxes = outputs["instances"].pred_boxes.tensor
    boxes[:, 0::2] += dx
    boxes[:, 1::2] += dy
    outputs["instances"].pred_boxes = Boxes(boxes)


    all_predictions.append({
        "predictions": outputs,
        "file_number": file_number,
    })


dataset_pred = []
for idx in range(len(all_predictions)):
    file_name = os.path.join(data_dirs, "train", "{}.png".format(all_predictions[idx]["file_number"]))

    # Check if filename already exists in dataset
    existing_record = next((record for record in dataset_pred if record["file_name"] == file_name), None)

    if existing_record is None:
        record = {
            "file_name": file_name,
            "instances": [all_predictions[idx]["predictions"]]
        }
        dataset_pred.append(record)
    else:
        record = existing_record
        width, height = get_image_dimensions(file_name)
        data_record = record["instances"][0]["instances"]
        data_new = all_predictions[idx]["predictions"]["instances"]

        if isinstance(data_record.pred_boxes, torch.Tensor):
            boxes = torch.cat((data_record.pred_boxes, data_new.pred_boxes.tensor), dim=0)
        elif isinstance(data_record.pred_boxes, Boxes):
            boxes = torch.cat((data_record.pred_boxes.tensor, data_new.pred_boxes.tensor), dim=0)
        # boxes = torch.cat((data_record.pred_boxes, data_new.pred_boxes), dim=0)
        boxes = Boxes(boxes)
        scores = torch.cat((data_record.scores, data_new.scores), dim=0)
        pred_classes = torch.cat((data_record.pred_classes, data_new.pred_classes), dim=0)
        
        new_instance = detectron2.structures.Instances(image_size=(height, width))
        new_instance.set('pred_boxes', boxes)
        new_instance.set('scores', scores)
        new_instance.set('pred_classes', pred_classes)
        record["instances"][0]["instances"] = new_instance

for idx, d in enumerate(dataset_pred): 
    img = cv2.imread(d["file_name"])
    print(d["file_name"])

    boxes = d["instances"][0]["instances"].pred_boxes.tensor
    scores = d["instances"][0]["instances"].scores
    keep = nms(boxes, scores, iou_threshold=0.2)
    d["instances"][0]["instances"] = d["instances"][0]["instances"][keep]
    # dataset_pred[idx][["instances"]] = d["instances"]
    v = Visualizer(img[:, :, ::-1],
                   #metadata=plane_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(d["instances"][0]["instances"].to("cpu"))
    save_path = f"{IMAGE_DIR}/Q1_PD_{idx}.jpg"
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print(f"Image saved to {save_path}")



'''
# Use COCOEvaluator and build_detection_train_loader
# You can save the output predictions using inference_on_dataset
# TODO: approx 5 lines
'''
evaluator = COCOEvaluator("plane_val_test", output_dir="./COCO_output")
val_loader = build_detection_test_loader(cfg, "plane_val_test")
print(inference_on_dataset(dataset_pred, predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`