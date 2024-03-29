# -*- coding: utf-8 -*-
"""lab3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Lh8fayLRdV6X90be6QsnIccA8kQkY3Eb

# Assignment 3

This is a template notebook for Assignment 3.

## Install dependencies and initialization
"""

# import some common libraries
# from google.colab.patches import cv2_imshow
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw
from tqdm import tqdm
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

# Make sure that GPU is available for your notebook.
# Otherwise, you need to update the settungs in Runtime -> Change runtime type -> Hardware accelerator
torch.cuda.is_available()

# Define the location of current directory, which should contain data/train, data/test, and data/train.json.
# TODO: approx 1 line
BASE_DIR = '.'
OUTPUT_DIR = '{}/output'.format(BASE_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""## Part 1: Object Detection

### Data Loader
"""

'''
# This function should return a list of data samples in which each sample is a dictionary.
# Make sure to select the correct bbox_mode for the data
# For the test data, you only have access to the images, therefore, the annotations should be empty.
# Other values could be obtained from the image files.
# TODO: approx 35 lines
'''
VAL_RATE = 0.2 # Precentage of the validate size
def get_detection_data(set_name, datapath="dataset_converted_part1_no_exclude_small_area.json"):
    data_dirs = '{}/data'.format(BASE_DIR)
    # return test_set, no annotations
    if set_name == "test":
        test_set = []
        for fname in os.listdir(os.path.join(data_dirs, "test")):
            if os.path.splitext(fname)[1] == ".png":
                path = os.path.join(data_dirs, "test", fname)
                width, height = Image.open(path).size
                test_set.append({
                    "file_name": path,
                    "image_id": os.path.splitext(fname)[0],
                    "height": height,
                    "width": width,
                    "annotations": []
                })
        return test_set
    # return validate_set or train_set, with annotations
    with open(datapath) as f:
        data = json.load(f)
    validate_size = int(len(data)*VAL_RATE)
    train_annotations, validate_annotations = data[0:len(data)-validate_size], data[len(data)-validate_size:]
    annotations = validate_annotations if set_name == "val" else (data if set_name == "all" else train_annotations)
    # return validate_set or train_set, with annotations
    datadict = {}
    for annotation in annotations:
        path = os.path.join(data_dirs, "train", annotation["file_name"])
        anno = {
            "bbox": annotation["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": annotation["segmentation"],
            "category_id": annotation["category_id"],
            "iscrowd": annotation["iscrowd"],
            "area": annotation["area"]
        }
        if path in datadict:
            datadict[path]["annotations"].append(anno)
            continue
        width, height = Image.open(path).size
        datadict[path] = {
            "image_id": annotation["image_id"],
            "height": height,
            "width": width,
            "annotations": [{
                "bbox": annotation["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": annotation["segmentation"],
                "category_id": annotation["category_id"],
                "iscrowd": annotation["iscrowd"],
                "area": annotation["area"]
            }]
        }
    return [{"file_name": path, **data} for path, data in datadict.items()]

'''
# Remember to add your dataset to DatasetCatalog and MetadataCatalog
# Consdier "data_detection_train" and "data_detection_test" for registration
# You can also add an optional "data_detection_val" for your validation by spliting the training data
# TODO: approx 5 lines
'''
for i in ["train", "val", "all", "test"]:
    DatasetCatalog.register("data_detection_{}".format(i), lambda i=i: get_detection_data(i))
    MetadataCatalog.get("data_detection_{}".format(i)).set(thing_classes=["not plane 1", "not plane 2", "not plane 3", "not plane 4", "plane"])

DatasetCatalog.register("data_detection_all_ori", lambda i=i: get_detection_data("all", datapath="data/train.json"))
MetadataCatalog.get("data_detection_all_ori").set(thing_classes=["not plane 1", "not plane 2", "not plane 3", "not plane 4", "plane"])
