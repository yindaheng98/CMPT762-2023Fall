# -*- coding: utf-8 -*-
from lab3_dataset2 import *
"""### Evaluation and Visualization"""

'''
# Before starting the evaluation, you need to set the model mode to eval
# You may load the trained model again, in case if you want to continue your code later
# TODO: approx 15 lines
'''
batch_size = 32
model = MyModel().cuda()
model.load_state_dict(torch.load('{}/output/999_segmentation_model_with_random_flip.pth'.format(BASE_DIR)))
model = model.eval() # chaning the model to evaluation mode will fix the bachnorm layers
loader, dataset = get_plane_dataset('val', batch_size, flip=False)

total_iou = 0
images = 0
SMOOTH = 1e-6
for (img, mask) in tqdm(loader):
  with torch.no_grad():
    img = img.cuda()
    mask = mask.cuda()
    pred = model(img)
    pred = pred[:, 0, :, :].squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    pred = pred > 0.5
    mask = mask > 0.5
    
    intersection = (pred & mask).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | mask).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    thresholded = iou
    total_iou += sum(thresholded.tolist())
    images += len(thresholded.tolist())

    '''
    ## Complete the code by obtaining the IoU for each img and print the final Mean IoU
    '''


print("\n #images: {}, Mean IoU: {}".format(images, total_iou/images))

# '''
# # Visualize 3 sample outputs
# # TODO: approx 5 lines
# '''
import matplotlib.pyplot as plt
for (img, mask) in loader:
    break
with torch.no_grad():
  img = img.cuda()
  mask = mask.cuda()
  pred = model(img)
  pred = pred[:, 0, :, :].squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
  pred[pred > 0.5] = 255
  pred[pred <= 0.5] = 0
  mask[mask > 0.5] = 255
  mask[mask <= 0.5] = 0
  mask = mask.cpu().numpy()
  pred = pred.cpu().numpy()
  img = img.cpu().numpy()
  for k in range(batch_size):
    p, m, i = pred[k, ...], mask[k, ...], img[k, ...]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.subplots(nrows=1, ncols=3)
    ax[0].imshow(p, cmap='gray')
    ax[1].imshow(m, cmap='gray')
    std = [58.395, 57.12, 57.375]
    mean = [123.675, 116.28, 103.53]
    for d in range(3):
      i[d, ...] = i[d, ...] * std[d] + mean[d]
    i = torch.tensor(i).permute((1,2,0)).numpy()
    ax[2].imshow(i)
    fig.savefig(os.path.join(OUTPUT_DIR, f"val_set_{k+1}.png"))
    plt.close(fig=fig)