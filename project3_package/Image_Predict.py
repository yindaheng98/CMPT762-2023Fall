import torch
import detectron2
from detectron2.structures import Boxes
from detectron2.layers import nms


size = 1000
stride = 500

BASE_DIR = './'
data_dirs = '{}/data'.format(BASE_DIR)
   
def split_image_to_Patch(img, height, width):
    cropped_positions = []
    cropped_images = []

    for i in range(0, width-stride, stride):
        for j in range(0, height-stride, stride):
            i_add = size
            j_add = size
            if i + i_add > width:
                i_add = width - i
            if j + j_add > height:
                j_add = height - j
            
            cropped_img = img[j: j + j_add, i: i + i_add] 
            cropped_images.append(cropped_img)

            position = [i,j]
            cropped_positions.append(position)

    return cropped_images, cropped_positions


def add_back_bbox(outputs, dx, dy):
    boxes = outputs["instances"].pred_boxes.tensor
    boxes[:, 0::2] += dx
    boxes[:, 1::2] += dy
    new_box = Boxes(boxes)
    return new_box


def predict_batch_result(predictor, cropped_images, cropped_positions):
    all_predictions = []
    for idx, img in enumerate(cropped_images):
        dx, dy = cropped_positions[idx]

        outputs = predictor(img)
        scores = outputs["instances"].scores
        keep = scores>0.6
        outputs["instances"] = outputs["instances"][keep]

        outputs["instances"].pred_boxes = add_back_bbox(outputs, dx, dy)

        all_predictions.append(outputs)
    return all_predictions


def remove_overlap_box(record):
    output = record
    boxes = output["instances"].pred_boxes.tensor
    scores = output["instances"].scores
    keep = nms(boxes, scores, iou_threshold=0.2)
    output["instances"] = output["instances"][keep]
    
    return output


def Batch_to_Original_Image(all_predictions, height, width):
    record = all_predictions[0]
    for idx in range(len(all_predictions)):
            data_record = record["instances"]
            data_new = all_predictions[idx]["instances"]

            boxes = torch.cat((data_record.pred_boxes.tensor, data_new.pred_boxes.tensor), dim=0)
            boxes = Boxes(boxes)
            scores = torch.cat((data_record.scores, data_new.scores), dim=0)
            pred_classes = torch.cat((data_record.pred_classes, data_new.pred_classes), dim=0)
            
            new_instance = detectron2.structures.Instances(image_size=(height, width))
            new_instance.set('pred_boxes', boxes)
            new_instance.set('scores', scores)
            new_instance.set('pred_classes', pred_classes)
            record["instances"] = new_instance     
    return remove_overlap_box(record)


def get_predict(obj_detect_model, img, height, width):
    
    cropped_images, cropped_positions = split_image_to_Patch(img, height, width)
    all_predictions = predict_batch_result(obj_detect_model, cropped_images, cropped_positions)
    output = Batch_to_Original_Image(all_predictions, height, width)
    
    return output
    
    
    
    
    
    
    