# -*- coding: utf-8 -*-
from lab3_dataset2 import *
from part3_utils import generate_pred_csv, plot_part3_result
"""### Evaluation and Visualization"""

'''
# Before starting the evaluation, you need to set the model mode to eval
# You may load the trained model again, in case if you want to continue your code later
# TODO: approx 15 lines
'''
# batch_size = 64
# model = MyModel().cuda()
# model.load_state_dict(torch.load('{}/output/999_segmentation_model.pth'.format(BASE_DIR)))
# model = model.eval() # chaning the model to evaluation mode will fix the bachnorm layers
# loader, dataset = get_plane_dataset('val', batch_size)

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
    
plot_part3_result(10)