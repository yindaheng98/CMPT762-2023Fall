# -*- coding: utf-8 -*-
from lab3_dataset2 import *
"""### Evaluation and Visualization"""

'''
# Before starting the evaluation, you need to set the model mode to eval
# You may load the trained model again, in case if you want to continue your code later
# TODO: approx 15 lines
'''
batch_size = 64
model = MyModel().cuda()
model.load_state_dict(torch.load('{}/output/999_segmentation_model.pth'.format(BASE_DIR)))
model = model.eval() # chaning the model to evaluation mode will fix the bachnorm layers
loader, dataset = get_plane_dataset('val', batch_size)

# total_iou = 0
# images = 0
# SMOOTH = 1e-6
# for (img, mask) in tqdm(loader):
#   with torch.no_grad():
#     img = img.cuda()
#     mask = mask.cuda()
#     pred = model(img)
#     pred = pred[:, 0, :, :].squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#     pred = pred > 0.5
#     mask = mask > 0.5
    
#     intersection = (pred & mask).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (pred | mask).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     total_iou += sum(thresholded.tolist())
#     images += len(thresholded.tolist())

#     '''
#     ## Complete the code by obtaining the IoU for each img and print the final Mean IoU
#     '''


# print("\n #images: {}, Mean IoU: {}".format(images, total_iou/images))

# '''
# # Visualize 3 sample outputs
# # TODO: approx 5 lines
# '''
# import matplotlib.pyplot as plt
# for (img, mask) in loader:
#     break
# with torch.no_grad():
#   img = img.cuda()
#   mask = mask.cuda()
#   pred = model(img)
#   pred = pred[:, 0, :, :].squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#   pred[pred > 0.5] = 255
#   pred[pred <= 0.5] = 0
#   mask[mask > 0.5] = 255
#   mask[mask <= 0.5] = 0
#   mask = mask.cpu().numpy()
#   pred = pred.cpu().numpy()
#   for k in range(batch_size):
#     p, m = pred[k, ...], mask[k, ...]
#     fig = plt.figure(figsize=(8, 4))
#     ax = fig.subplots(nrows=1, ncols=2)
#     ax[0].imshow(p, cmap='gray')
#     ax[1].imshow(m, cmap='gray')
#     fig.savefig(os.path.join(OUTPUT_DIR, f"val_set_{k+1}.png"))
#     plt.close(fig=fig)
    
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

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") # pretrain model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
object_detect_model = DefaultPredictor(cfg)

from part3_utils import get_prediction_mask, rle_encoding, getDistinctColors

# '''
# # You need to upload the csv file on kaggle
# # The speed of your code in the previous parts highly affects the running time of this part
# '''

# preddic = {"ImageId": [], "EncodedPixels": []}
# '''
# # Writing the predictions of the training set
# '''
# my_data_list = DatasetCatalog.get("data_detection_{}".format('train'))
# for i in tqdm(range(len(my_data_list)), position=0, leave=True):
#   sample = my_data_list[i]
#   sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
#   img, true_mask, pred_mask = get_prediction_mask(sample, object_detect_model, model)
#   inds = torch.unique(pred_mask)
#   if(len(inds)==1):
#     preddic['ImageId'].append(sample['image_id'])
#     preddic['EncodedPixels'].append([])
#   else:
#     for index in inds:
#       if(index == 0):
#         continue
#       tmp_mask = (pred_mask==index)
#       encPix = rle_encoding(tmp_mask)
#       preddic['ImageId'].append(sample['image_id'])
#       preddic['EncodedPixels'].append(encPix)

# '''
# # Writing the predictions of the val set
# '''
# my_data_list = DatasetCatalog.get("data_detection_{}".format('val'))
# for i in tqdm(range(len(my_data_list)), position=0, leave=True):
#   sample = my_data_list[i]
#   sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
#   img, true_mask, pred_mask = get_prediction_mask(sample, object_detect_model, model)
#   inds = torch.unique(pred_mask)
#   if(len(inds)==1):
#     preddic['ImageId'].append(sample['image_id'])
#     preddic['EncodedPixels'].append([])
#   else:
#     for index in inds:
#       if(index == 0):
#         continue
#       tmp_mask = (pred_mask==index)
#       encPix = rle_encoding(tmp_mask)
#       preddic['ImageId'].append(sample['image_id'])
#       preddic['EncodedPixels'].append(encPix)

# '''
# # Writing the predictions of the test set
# '''

# my_data_list = DatasetCatalog.get("data_detection_{}".format('test'))
# for i in tqdm(range(len(my_data_list)), position=0, leave=True):
#   sample = my_data_list[i]
#   sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
#   img, true_mask, pred_mask = get_prediction_mask(sample, object_detect_model, model)
#   inds = torch.unique(pred_mask)
#   if(len(inds)==1):
#     preddic['ImageId'].append(sample['image_id'])
#     preddic['EncodedPixels'].append([])
#   else:
#     for j, index in enumerate(inds):
#       if(index == 0):
#         continue
#       tmp_mask = (pred_mask==index).double()
#       encPix = rle_encoding(tmp_mask)
#       preddic['ImageId'].append(sample['image_id'])
#       preddic['EncodedPixels'].append(encPix)

# pred_file = open("{}/pred.csv".format(BASE_DIR), 'w')
# pd.DataFrame(preddic).to_csv(pred_file, index=False)
# pred_file.close()

import matplotlib.pyplot as plt

visual_id = 21

# Test get_prediction_mask
print(get_detection_data("test")[visual_id]['file_name'])
img, gt_mask, pred_mask = get_prediction_mask(get_detection_data("test")[visual_id],
                                              object_detect_model,
                                              model)

print("Num of mask: {}".format(len(torch.unique(pred_mask))))

# convert the gray scale mask to a colored mask
colored_pred = torch.zeros([pred_mask.shape[0], pred_mask.shape[1], 3])
num_of_bbox = torch.max(pred_mask).cpu().item()
distinct_colors = getDistinctColors(num_of_bbox)
# print(distinct_colors)
for rgb in range(2):
  for idx in range(1, num_of_bbox + 1):
    colored_pred[:, :, rgb][pred_mask == idx] = distinct_colors[idx - 1][rgb]

img = cv2.resize(img.cpu().numpy(), (0, 0), fx=0.5, fy=0.5)
pred_mask = cv2.resize(colored_pred.cpu().numpy(), (0, 0), fx=0.5, fy=0.5)

pred_mask = pred_mask.astype('int')

plt.cla()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(pred_mask)

plt.savefig("temp.png", dpi=800)