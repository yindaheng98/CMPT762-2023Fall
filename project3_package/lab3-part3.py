from lab3_dataset2 import *

from detectron2.structures.boxes import pairwise_point_box_distance
from detectron2.structures.masks import polygons_to_bitmask
import colorsys
import matplotlib.pyplot as plt

from Image_Predict import get_predict as detect_model_with_image_split

IMAGE_SIZE = 128

def remove_overlapped_bbox(instances, score_threshold=0.8):
  removed_idx = []
  num_of_ins = len(instances['instances'])
  for i in range(num_of_ins):
    if i in removed_idx:
      continue
    for j in range(i + 1, num_of_ins):
      if j in removed_idx:
        continue
      
      dis1 = pairwise_point_box_distance(instances['instances'][i].pred_boxes.get_centers(),
                                         instances['instances'][j].pred_boxes)
      dis2 = pairwise_point_box_distance(instances['instances'][j].pred_boxes.get_centers(),
                                        instances['instances'][i].pred_boxes)
      
      # i is in j or j is in i
      if torch.all(dis1 > 0).item() or torch.all(dis2 > 0).item():
        if instances['instances'][i].scores.item() > instances['instances'][j].scores.item():
          removed_idx.append(j)
        else:
          removed_idx.append(i)

  # Create new instances with overlapped bbox removed
  height, width = instances['instances'].image_size
  
  kept_idx = [x for x in range(len(instances['instances'])) if x not in removed_idx]
  new_instances = detectron2.structures.Instances(image_size=(height, width))
          
  classes = instances['instances'].pred_classes[kept_idx]
  scores = instances['instances'].scores[kept_idx]
  boxes = instances['instances'].pred_boxes[kept_idx]
  
  new_instances.set('pred_classes', classes)
  new_instances.set('scores', scores)
  new_instances.set('pred_boxes', boxes)
  
  # Remove low score bbox
  new_instances = new_instances[new_instances.scores >= score_threshold]        
  
  return {"instances": new_instances}

def get_prediction_mask(data, obj_detect_model, seg_model):
  img = cv2.imread(data['file_name'])
  height, width = img.shape[:2]
  
  # obj_detect_result = obj_detect_model(img)
  # img_PIL = Image.fromarray(np.uint8(img)).convert('RGB')
  obj_detect_result = detect_model_with_image_split(obj_detect_model, img, height, width)
  
  # obj_detect_result = remove_overlapped_bbox(obj_detect_result)
  
  pred_mask = np.zeros((height,width), np.uint8)
  
  num_of_bbox = len(obj_detect_result['instances'])
  # print("Number of bbox: {}".format(num_of_bbox))
  
  for idx in range(num_of_bbox):
    # Obtain the bbox from Instances
    x0, y0, x1, y1 = [int(x) for x in obj_detect_result['instances'][idx].pred_boxes.tensor.cpu().numpy()[0]]
    crop_img = img[y0:y1, x0:x1]
    ori_h, ori_w = crop_img.shape[:2]
    obj_img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # convert obj_img to suitable format
    image_preprocess = transforms.Compose([
      transforms.ToTensor(), # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
      transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
    ])
    
    obj_img = image_preprocess(obj_img)
    obj_img = obj_img.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))

    with torch.no_grad():
      obj_img = obj_img.cuda()
      pred = seg_model(obj_img)
      
      sig_pred = pred
      
      # Take the first class (plane)
      sig_pred = sig_pred[:, 0, :, :]
      
      sig_pred[sig_pred > 0.5] = idx + 1
      sig_pred[sig_pred <= 0.5] = 0
      
            
      sig_pred = np.transpose(sig_pred.cpu(), (1, 2, 0))
      sig_pred = np.reshape(sig_pred, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Resize the colored_pred to original size, and update the pred_mask
    sig_pred = cv2.resize(sig_pred.numpy(), (ori_w, ori_h), interpolation = cv2.INTER_NEAREST)
    pred_mask[y0:y1, x0:x1] = sig_pred
  
  # print("After Number of plane: {}".format(len(np.unique(pred_mask)) - 1))
  
  gt_mask = None
  img = torch.tensor(img, device='cuda').int()
  # gt_mask = torch.tensor(gt_mask)
  pred_mask = torch.tensor(pred_mask, device='cuda').int()
  
  return img, gt_mask, obj_detect_result, pred_mask # gt_mask could be all zero when the ground truth is not given.

def rle_encoding(x):
  '''
  x: pytorch tensor on gpu, 1 - mask, 0 - background
  Returns run length as list
  '''
  dots = torch.where(torch.flatten(x.long())==1)[0]
  if(len(dots)==0):
    return []
  inds = torch.where(dots[1:]!=dots[:-1]+1)[0]+1
  inds = torch.cat((torch.tensor([0], device=torch.device('cuda'), dtype=torch.long), inds))
  tmpdots = dots[inds]
  inds = torch.cat((inds, torch.tensor([len(dots)], device=torch.device('cuda'))))
  inds = inds[1:] - inds[:-1]
  runs = torch.cat((tmpdots, inds)).reshape((2,-1))
  runs = torch.flatten(torch.transpose(runs, 0, 1)).cpu().data.numpy()
  return ' '.join([str(i) for i in runs])
  
# The getDistinctColors function is from https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
def HSVToRGB(h, s, v): 
 (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
 return [int(255*r), int(255*g), int(255*b)]
 
def getDistinctColors(n): 
 huePartition = 1.0 / (n + 1) 
 return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]

def load_detect_and_seg_model():
  # Load object detection model
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

  cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)

  cfg.DATASETS.TRAIN = ("plane_train",)
  cfg.DATASETS.TEST = ()

  cfg.SOLVER.MAX_ITER = 500             # Number of training iterations
  cfg.SOLVER.BASE_LR = 0.00025          # Learning rate
  cfg.SOLVER.IMS_PER_BATCH = 2          # Number of images per batch
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Number of regions per image for training
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_detection.pth")
  # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") # pretrain model
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
  obj_detect_model = DefaultPredictor(cfg)

  # Load segmentation model 
  seg_model = MyModel().cuda()
  seg_model.load_state_dict(torch.load('{}/output/final_segmentation_model.pth'.format(BASE_DIR)))
  seg_model = seg_model.eval() # chaning the model to evaluation mode will fix the bachnorm layers
  
  return obj_detect_model, seg_model

def generate_pred_csv():
  obj_detect_model, seg_model = load_detect_and_seg_model()

  preddic = {"ImageId": [], "EncodedPixels": []}
  '''
  # Writing the predictions of the all train set
  '''
  my_data_list = DatasetCatalog.get("data_detection_all_ori")
  for i in tqdm(range(len(my_data_list)), position=0, leave=True):
    sample = my_data_list[i]
    sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
    _, _, _, pred_mask = get_prediction_mask(sample, obj_detect_model, seg_model)
    inds = torch.unique(pred_mask)
    if(len(inds)==1):
      preddic['ImageId'].append(sample['image_id'])
      preddic['EncodedPixels'].append([])
    else:
      for index in inds:
        if(index == 0):
          continue
        tmp_mask = (pred_mask==index)
        encPix = rle_encoding(tmp_mask)
        preddic['ImageId'].append(sample['image_id'])
        preddic['EncodedPixels'].append(encPix)

  '''
  # Writing the predictions of the test set
  '''
  my_data_list = DatasetCatalog.get("data_detection_{}".format('test'))
  for i in tqdm(range(len(my_data_list)), position=0, leave=True):
    sample = my_data_list[i]
    sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
    _, _, _, pred_mask = get_prediction_mask(sample, obj_detect_model, seg_model)
    inds = torch.unique(pred_mask)
    if(len(inds)==1):
      preddic['ImageId'].append(sample['image_id'])
      preddic['EncodedPixels'].append([])
    else:
      for j, index in enumerate(inds):
        if(index == 0):
          continue
        tmp_mask = (pred_mask==index).double()
        encPix = rle_encoding(tmp_mask)
        preddic['ImageId'].append(sample['image_id'])
        preddic['EncodedPixels'].append(encPix)

  pred_file = open("{}/pred.csv".format(BASE_DIR), 'w')
  pd.DataFrame(preddic).to_csv(pred_file, index=False)
  pred_file.close()
  
def plot_part3_result(num_of_images=None, data_set='test'):
  obj_detect_model, seg_model = load_detect_and_seg_model()
  os.makedirs('images/', exist_ok=True)
  
  if data_set == 'test':
    data = DatasetCatalog.get("data_detection_test")
  elif data_set == 'val':
    data = DatasetCatalog.get("data_detection_val")
  elif data_set == 'train':
    data = DatasetCatalog.get("data_detection_train")
  else:
    raise SystemExit('Uknown data set')
  
  if num_of_images == None:
    num_of_images = len(data)

  total_iou = 0
  for i in tqdm(range(num_of_images)):
    image_name = data[i]['file_name'].split('/')[-1]
    img, gt_mask, bbox_result, pred_mask = get_prediction_mask(data[i],
                                                  obj_detect_model,
                                                  seg_model)

    # convert the gray scale mask to a colored mask
    colored_pred = torch.zeros([pred_mask.shape[0], pred_mask.shape[1], 3])
    num_of_bbox = torch.max(pred_mask).cpu().item()
    distinct_colors = getDistinctColors(num_of_bbox)

    for rgb in range(2):
      for idx in range(1, num_of_bbox + 1):
        colored_pred[:, :, rgb][pred_mask == idx] = distinct_colors[idx - 1][rgb]

    img = img.cpu().numpy().astype('int')
    colored_pred = colored_pred.cpu().numpy().astype('int')

    plt.cla()
    if data_set == 'test':
      fig, ax = plt.subplots(1, 3)
      ax[0].imshow(img, vmin=0, vmax=255)
      ax[1].imshow(colored_pred, vmin=0, vmax=255)
      visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("data_detection_train"), scale=1)
      bbox_visual = visualizer.draw_instance_predictions(bbox_result["instances"].to("cpu"))
      ax[2].imshow(bbox_visual.get_image()[:, :, ::-1])
      
    else:
      fig, ax = plt.subplots(1, 4)
      ax[0].imshow(img, vmin=0, vmax=255)
      ax[1].imshow(colored_pred, vmin=0, vmax=255)
      visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("data_detection_train"), scale=1)
      bbox_visual = visualizer.draw_instance_predictions(bbox_result["instances"].to("cpu"))
      ax[2].imshow(bbox_visual.get_image()[:, :, ::-1])
      
      ground_truth_mask = get_binary_seg_from_detectron2(data[i])
      ax[3].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)
      sum_mask = np.sum(colored_pred, axis=2)
      sum_mask[sum_mask!=0] = 1
      pred = torch.sum(torch.tensor(colored_pred), axis=2)!=0
      mask = torch.tensor(ground_truth_mask) > 0.5
      intersection = (pred & mask).float().sum((0, 1))  # Will be zero if Truth=0 or Prediction=0
      union = (pred | mask).float().sum((0, 1))         # Will be zzero if both are 0
      SMOOTH = 1e-6
      iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0\
      total_iou += iou

    plt.tight_layout()
    plt.savefig("images/part3_{}_{}".format(data_set, image_name), dpi=600)
  print("\n #images: {}, Mean IoU: {}".format(num_of_images, total_iou/num_of_images))

def get_binary_seg_from_detectron2(data):
  output = np.zeros((data['height'], data['width'])).astype('int')
  for anno in data['annotations']:
    polygon_bitmap = polygons_to_bitmask(anno['segmentation'], data['height'], data['width']).astype('int')
    output = np.bitwise_or(polygon_bitmap, output)
  
  return output

if __name__ == "__main__":

  generate_pred_csv()

  plot_part3_result(10, 'val')