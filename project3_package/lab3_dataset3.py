# -*- coding: utf-8 -*-
from lab3_dataset import *

"""## Part 2: Semantic Segmentation

### Data Loader
"""

'''
# Write a function that returns the cropped image and corresponding mask regarding the target bounding box
# idx is the index of the target bbox in the data
# high-resolution image could be passed or could be load from data['file_name']
# You can use the mask attribute of detectron2.utils.visualizer.GenericMask
#     to convert the segmentation annotations to binary masks
# TODO: approx 10 lines
'''
from detectron2.utils.visualizer import GenericMask
def get_instance_sample(data):
    height, width = data['height'], data['width']
    obj_img = cv2.imread(data['file_name'])
    obj_mask = np.zeros((height, width))
    for anno in data['annotations']:
        bbox = anno['bbox']
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = x1 + int(bbox[2]), y1 + int(bbox[3])
        small_mask = GenericMask(anno['segmentation'], height, width).mask[y1:y2,x1:x2]
        obj_mask[y1:y2,x1:x2] = small_mask
    return obj_img, obj_mask

'''
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
'''

class PlaneDataset(Dataset):
    def __init__(self, set_name, data_list, flip=False):
        self.transforms = transforms.Compose([
            transforms.ToTensor(), # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
            transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        ])
        self.transforms_flip = transforms.Compose([])
        if flip:
            self.transforms_flip = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        self.set_name = set_name
        self.data = data_list

    '''
    # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
    # make sure to use the correct length for the final training
    '''
    def __len__(self):
        return len(self.data)

    def numpy_to_tensor(self, img, mask):
        if self.transforms is not None:
            img = self.transforms(img)
        mask = torch.tensor(mask, dtype=torch.float)
        both_images = torch.cat((img, mask.unsqueeze(0)), 0)
        both_images = self.transforms_flip(both_images)
        img, mask = both_images[0:3], both_images[3]
        return img, mask

    '''
    # Complete this part by using get_instance_sample function
    # make sure to resize the img and mask to a fixed size (for example 128*128)
    # you can use "interpolate" function of pytorch or "numpy.resize"
    # TODO: 5 lines
    '''
    def __getitem__(self, idx):
        data = self.data[idx]

        img, mask = get_instance_sample(data)
        img, mask = self.numpy_to_tensor(img, mask)

        return img, mask

def get_plane_dataset(set_name='train', batch_size=2, flip=False, shuffle=False):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list, flip=flip)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)
    return loader, dataset

"""### Network"""

from hubconf import farseg_resnet50 as MyModel

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    set_name = 'train'
    loader, dataset = get_plane_dataset(set_name=set_name, batch_size=1)
    for idx, (img, mask) in enumerate(tqdm(loader)):
        _, mask = dataset[idx]
        img = cv2.imread(dataset.data[idx]['file_name'])
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots(nrows=1, ncols=2)
        ax[0].imshow(img, vmin=0, vmax=255)
        ax[1].imshow(mask, vmin=0, vmax=1, cmap='gray')
        fig.savefig(os.path.join(BASE_DIR, "output", f"seg_{set_name}_{idx}.png"))
        plt.close(fig=fig)