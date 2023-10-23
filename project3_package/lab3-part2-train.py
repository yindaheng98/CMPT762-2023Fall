# -*- coding: utf-8 -*-
from lab3_dataset2 import *
"""### Training"""

'''
# The following is a basic training procedure to train the network
# You need to update the code to get the best performance
# TODO: approx ? lines
'''

# Set the hyperparameters
num_epochs = 200
batch_size = 32
momentum = 0.9
learning_rate = 0.1
weight_decay = 0.0001

model = MyModel() # initialize the model
model = model.cuda() # move the model to GPU
loader, _ = get_plane_dataset('train', batch_size, flip=True) # initialize data_loader
crit = nn.BCEWithLogitsLoss() # Define the loss function
optim = torch.optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD
from torch.optim.lr_scheduler import MultiStepLR
# scheduler = CosineAnnealingLR(optim, T_max=num_epochs*len(loader), eta_min=1e-6) #learning rate decay
max_iters = num_epochs*len(loader)
scheduler = MultiStepLR(optim, milestones=[int(max_iters*0.3),int(max_iters*0.6),int(max_iters*0.8)], gamma=0.2) #learning rate decay
# start the training procedure
for epoch in range(num_epochs):
  total_loss = 0
  for (img, mask) in tqdm(loader):
    img = img.to(device=torch.device('cuda'))
    mask = mask.to(device=torch.device('cuda'))
    pred = model(img)
    pred_mask = pred[:, 0, ...].squeeze(1)
    loss = crit(pred_mask, mask)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2)
    optim.step()
    total_loss += loss.cpu().data
    scheduler.step()
  print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(loader)))
  torch.save(model.state_dict(), '{}/output/{}_segmentation_model.pth'.format(BASE_DIR, epoch))
  print("Next learning rate:", scheduler.get_last_lr())

'''
# Saving the final model
'''
torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))
