# -*- coding: utf-8 -*-
from lab3_dataset import *
'''
# Visualize some samples using Visualizer to make sure that the function works correctly
# TODO: approx 5 lines
'''
train_set = get_detection_data("train")
data = train_set[random.randrange(0, len(train_set))]
img = cv2.imread(data["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("data_detection_train"), scale=0.5)
out = visualizer.draw_dataset_dict(data)
save_path = os.path.join(BASE_DIR, "output", "train_set.jpg")
cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

"""----------------Part 1 deleted here----------------"""

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
cache_dir = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(cache_dir, exist_ok=True)
big_cache = {}
queue = []
def get_instance_sample(data, idx, img=None):
    height, width = data['height'], data['width']
    bbox = data['annotations'][idx]['bbox']
    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = x1 + int(bbox[2]), y1 + int(bbox[3])
    cache_path = os.path.join(cache_dir, os.path.basename(data['file_name']) + f"-{idx}.png")
    if not os.path.isfile(cache_path):
        if data['file_name'] not in big_cache:
            if len(big_cache) >= 2:
                big_cache[queue.pop()] = None
            big_cache[data['file_name']] = cv2.imread(data['file_name'])
            queue.append(data['file_name'])
        cv2.imwrite(cache_path, big_cache[data['file_name']][y1:y2,x1:x2,:])
    obj_img = cv2.imread(cache_path)
    obj_mask = np.zeros((int(bbox[3]), int(bbox[2])))
    if len(data['annotations'][idx]['segmentation']) > 0 and sum(len(a) for a in data['annotations'][idx]['segmentation']) > 0:
        obj_mask = GenericMask(data['annotations'][idx]['segmentation'], height, width).mask[y1:y2,x1:x2]
    obj_img = cv2.resize(obj_img, (128, 128))
    obj_mask = cv2.resize(obj_mask, (128, 128))
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
        self.instance_map = []
        for i, d in enumerate(self.data):
            for j in range(len(d['annotations'])):
                self.instance_map.append([i,j])

    '''
    # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
    # make sure to use the correct length for the final training
    '''
    def __len__(self):
        return len(self.instance_map)

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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.instance_map[idx]
        data = self.data[idx[0]]

        img, mask = get_instance_sample(data, idx[1])
        img, mask = self.numpy_to_tensor(img, mask)

        return img, mask

def get_plane_dataset(set_name='train', batch_size=2, flip=False, shuffle=False):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list, flip=flip)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=shuffle)
    return loader, dataset

"""### Network"""

from farseg import farseg_resnet50 as MyModel
