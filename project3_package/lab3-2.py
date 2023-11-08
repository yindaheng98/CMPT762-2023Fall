# import some common libraries
from typing import Any
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
import shutil

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.structures import Boxes
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
from detectron2.utils.visualizer import GenericMask
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
if os.path.exists(COCO_EVALUATOR_OUTPUT):
    shutil.rmtree(COCO_EVALUATOR_OUTPUT)
os.makedirs(COCO_EVALUATOR_OUTPUT, exist_ok=True)

IMAGE_DIR = "{}/images_output".format(BASE_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

data_dirs = '{}/data'.format(BASE_DIR)

cropped_dir = os.path.join(data_dirs, "cropped")
os.makedirs(cropped_dir, exist_ok=True)


stride = 500
size = 1000


def get_image_dimensions(filename):
    with Image.open(filename) as img:
        return img.size 


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
    width, height = img.size
    x, y, w, h = annotations["bbox"]

    for i in range(0, width-stride, stride):
        for j in range(0, height-stride, stride):
            i_add = size
            j_add = size
            if i + i_add > width:
                i_add = width - i
            if j + j_add > height:
                j_add = height - j
            
            cropped_filename = os.path.join(cropped_dir, "{}_{}_{}".format(i,j,annotations["file_name"]))
            if not os.path.exists(cropped_filename):
                cropped_img = img.crop((i, j, i + i_add, j + j_add))
                cropped_img.save(cropped_filename,"PNG")

            if x >= i and (x + w) <= (i + size) and y >= j and (y + h) <= (j + size):
                cropped_anno = annotations.copy()
                cropped_anno["bbox"] = [x - i, y - j, w, h]
                cropped_anno["segmentation"] = adjust_segmentation(annotations["segmentation"],i,j)
                cropped_anno["file_name"] = cropped_filename
                cropped_anno["position"] = [i,j]
                cropped_anno["width"] = i_add
                cropped_anno["height"] = j_add
                cropped_annotations.append(cropped_anno)
    return cropped_annotations


def get_detection_data(set_name):
    
    dataset = []
    with open(os.path.join(data_dirs, "train.json"), 'r') as f:
      data = json.load(f)

    val_size = 6274

    train_data = data[:val_size]
    val_data = data[val_size:]

    if set_name == "train":
        annotations = train_data
    elif set_name == "val":
        annotations = val_data
    elif set_name == "all":
        annotations = data
    elif set_name == "test":
        annotations = data
    else:
        raise ValueError("Unknown set_name: {}".format(set_name))

    for anno in annotations:
        filename = os.path.join(data_dirs, "train", anno["file_name"])
        if set_name == "test":
            width, height = get_image_dimensions(filename)
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

            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
                "iscrowd": anno["iscrowd"],
                "area": anno["area"]
            }
            record["annotations"].append(obj)
        else: 
            img = Image.open(filename)
            cropped_anno = split_image_and_annotations(img, anno)
            
            for crop_anno in cropped_anno:
                filename_crop = crop_anno["file_name"]
                existing_record = next((record for record in dataset if record["file_name"] == filename_crop), None)
                if existing_record is None:
                    record = {
                        "file_name": filename_crop,
                        "image_id": crop_anno["image_id"],
                        "height": crop_anno["height"],
                        "width": crop_anno["width"],
                        "position": crop_anno["position"],
                        "annotations": []
                    }
                    dataset.append(record)
                else:
                    record = existing_record

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
for d in ["train", "test", "val", "all"]:
    DatasetCatalog.register("plane_" + d, lambda d=d: get_detection_data(d))
    MetadataCatalog.get("plane_" + d).set(thing_classes=["class1","class2","class3","class4","plane"])

plane_metadata = MetadataCatalog.get("plane_all")


'''
# Visualize some samples using Visualizer to make sure that the function works correctly
# TODO: approx 5 lines
'''
# Load some samples from the training dataset
samples = get_detection_data("all")

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

cfg.DATASETS.TRAIN = ("plane_all",)
cfg.DATASETS.TEST = ()

cfg.SOLVER.MAX_ITER = 1000             # Number of training iterations
cfg.SOLVER.BASE_LR = 0.00025          # Learning rate
cfg.SOLVER.IMS_PER_BATCH = 2          # Number of images per batch
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Number of regions per image for training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# 4. Pretrain Model
cfg.MODEL.WEIGHTS = "{}/500_detection_model.pth".format(cfg.OUTPUT_DIR)


'''
# Create a DefaultTrainer using the above config and train the model
# TODO: approx 5 lines
'''
#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()


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
def calculate_iou(box1, box2):
    # 计算两个框的交集坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集的面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算重叠的面积占各自框面积的比例
    iou_box1 = intersection_area / box1_area
    iou_box2 = intersection_area / box2_area

    return iou_box1, iou_box2


def filter_overlap_bbox(dataset_pred, iou_threshold=0.6, overlap_ratio=0.7, size_ratio=0.01):
    dataset_pred = dataset_pred
    for idx, d in enumerate(dataset_pred): 
        boxes = d["instances"][0]["instances"].pred_boxes.tensor
        scores = d["instances"][0]["instances"].scores


        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        average_area = areas.mean()

        keep_size = areas >= (average_area * size_ratio)
        boxes = boxes[keep_size]
        scores = scores[keep_size]

        keep = nms(boxes, scores, iou_threshold=iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        keep_mask = torch.ones(boxes.size(0), dtype=torch.bool)

        for i in range(boxes.size(0)):
            if not keep_mask[i]:
                continue
            for j in range(boxes.size(0)):
                if i != j and keep_mask[j]:
                    iou_box1, iou_box2 = calculate_iou(boxes[i], boxes[j])

                    if iou_box1 > overlap_ratio or iou_box2 > overlap_ratio:
                        if scores[i] < scores[j]:
                            keep_mask[i] = False
                        else:
                            keep_mask[j] = False
        keep = keep[keep_mask]
        d["instances"][0]["instances"] = d["instances"][0]["instances"][keep]

    return dataset_pred


def predict_and_adjust_bbox(dataset_dicts):
    all_predictions = []
    for idx, d in enumerate(dataset_dicts):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        file_number = d["file_name"][-9:-4]
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
    return all_predictions


def create_predict_result(dataset_dicts):
    dataset_pred = []
    all_predictions = predict_and_adjust_bbox(dataset_dicts)
    for idx in range(len(all_predictions)):
        file_name = os.path.join(data_dirs, "train", "{}.png".format(all_predictions[idx]["file_number"]))

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

            boxes = torch.cat((data_record.pred_boxes.tensor, data_new.pred_boxes.tensor), dim=0)
            boxes = Boxes(boxes)
            scores = torch.cat((data_record.scores, data_new.scores), dim=0)
            pred_classes = torch.cat((data_record.pred_classes, data_new.pred_classes), dim=0)
            
            new_instance = detectron2.structures.Instances(image_size=(height, width))
            new_instance.set('pred_boxes', boxes)
            new_instance.set('scores', scores)
            new_instance.set('pred_classes', pred_classes)
            record["instances"][0]["instances"] = new_instance

    return filter_overlap_bbox(dataset_pred)

dataset_dicts = get_detection_data("all")
dataset_pred = create_predict_result(dataset_dicts)

'''
# Use COCOEvaluator and build_detection_train_loader
# You can save the output predictions using inference_on_dataset
# TODO: approx 5 lines
'''
evaluator = COCOEvaluator("plane_test", output_dir="./COCO_output")
val_loader = build_detection_test_loader(cfg, "plane_test")
print(inference_on_dataset(dataset_pred, predictor.model, val_loader, evaluator))

