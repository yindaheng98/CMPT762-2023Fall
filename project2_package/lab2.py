# no change👇----------------------------------------------------------👇no change
"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import csv
# changed!👉%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)
# no change👇----------------------------------------------------------👇no change
""""""

class CIFAR10_SFU_CV(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar100'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar100.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, fold="train",
                transform=None, target_transform=None,
                download=False):
        
        fold = fold.lower()

        self.train = False
        self.test = False
        self.val = False

        if fold == "train":
            self.train = True
        elif fold == "test":
            self.test = True
        elif fold == "val":
            self.val = True
        else:
            raise RuntimeError("Not train-val-test")


        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        fpath = os.path.join(root, self.filename)
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                            ' Download it and extract the file again.')

        # now load the picked numpy arrays
        if self.train or self.val:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            
            p = np.arange(0,50000,10)
            mask_train = np.ones((50000,), dtype=bool)
            mask_train[p] = False
            mask_val = np.zeros((50000,), dtype=bool)
            mask_val[p] = True

            copy_all_data = np.array(self.train_data)
            self.val_data = np.array(copy_all_data[mask_val])
            self.train_data = np.array(copy_all_data[mask_train])
            
            copy_all_labels = np.array(self.train_labels)
            self.val_labels = np.array(copy_all_labels[mask_val])
            self.train_labels = np.array(copy_all_labels[mask_train])

        elif self.test:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.test:
            img, target = self.test_data[index], self.test_labels[index]
        elif self.val:
            img, target = self.val_data[index], self.val_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.test:
            return len(self.test_data)
        elif self.val:
            return len(self.val_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100_SFU_CV(CIFAR10_SFU_CV):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar100'
    filename = "cifar100.tar.gz"
    tgz_md5 = 'e68a4c763591787a0b39fe2209371f32'
    train_list = [
        ['train_cs543', '49eee854445c1e2ebe796cd93c20bb0f'],
    ]

    test_list = [
        ['test_cs543', 'd3fe9f6a9251bd443f428f896d27384f'],
    ]
if __name__ == "__main__":
    # no change👇----------------------------------------------------------👇no change
    # <<TODO#5>> Based on the val set performance, decide how many
    # epochs are apt for your model.
    # ---------
    EPOCHS = 120
    # ---------

    IS_GPU = True
    TEST_BS = 800
    TOTAL_CLASSES = 100
    TRAIN_BS = 400
    PATH_TO_CIFAR100_SFU_CV = "./data/"
    # no change👇----------------------------------------------------------👇no change
    def calculate_val_accuracy(valloader, is_gpu):
        """ Util function to calculate val set accuracy,
        both overall and per class accuracy
        Args:
            valloader (torch.utils.data.DataLoader): val set 
            is_gpu (bool): whether to run on GPU
        Returns:
            tuple: (overall accuracy, class level accuracy)
        """    
        correct = 0.
        total = 0.
        predictions = []

        class_correct = list(0. for i in range(TOTAL_CLASSES))
        class_total = list(0. for i in range(TOTAL_CLASSES))

        for data in valloader:
            images, labels = data
            if is_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(list(predicted.cpu().numpy()))
            total += labels.size(0)
            # The following line reported an error for some students. Put a new version.
            # correct += (predicted == labels).sum()
            correct += torch.sum(predicted == labels).detach().cpu().numpy()

            # The following line reported an error for some students. Put a new version.
            # c = (predicted == labels).squeeze()
            c = torch.squeeze(predicted == labels).detach().cpu().numpy()	
            # Added for a fix.
            # c = c.cpu()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

        class_accuracy = 100 * np.divide(class_correct, class_total)
        return 100*correct/total, class_accuracy
    # no change👇----------------------------------------------------------👇no change
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # Using transforms.ToTensor(), transform them to Tensors of normalized range
    # [-1, 1].


    # <<TODO#1>> Use transforms.Normalize() with the right parameters to 
    # make the data well conditioned (zero mean, std dev=1) for improved training.
    # <<TODO#2>> Try using transforms.RandomCrop() and/or transforms.RandomHorizontalFlip()
    # to augment training data.
    # After your edits, make sure that test_transform should have the same data
    # normalization parameters as train_transform
    # You shouldn't have any data augmentation in test_transform (val or test data is never augmented).
    # ---------------------

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    # ---------------------

    trainset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="train",
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BS,
                                            shuffle=True, num_workers=2)
    print("Train set size: "+str(len(trainset)))

    valset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="val",
                                        download=True, transform=test_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=2)
    print("Val set size: "+str(len(valset)))

    testset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="test",
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=2)
    print("Test set size: "+str(len(testset)))

    # The 100 classes for CIFAR100
    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # change!!!👇----------------------------------------------------------👇!!!change
    
    # Stick all the images together to form a 45000 X 3 X 32 X 32 array
    x = np.stack([np.asarray(trainset[i][0]) for i in range(len(trainset))])

    # calculate the mean and std along the (0, 1) axes
    mean = np.mean(x, axis=(0, 2, 3))
    std = np.std(x, axis=(0, 2, 3))
    # the the mean and std
    mean=mean.tolist()
    std=std.tolist()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    print('transforms.Normalize(', mean,',', std, ')')
    
    # change!!!👇----------------------------------------------------------👇!!!change
    ########################################################################
    # 2. Define a Convolution Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # We provide a basic network that you should understand, run and
    # eventually improve
    # <<TODO>> Add more conv layers
    # <<TODO>> Add more fully connected (fc) layers
    # <<TODO>> Add regularization layers like Batchnorm.
    #          nn.BatchNorm2d after conv layers:
    #          http://pytorch.org/docs/master/nn.html#batchnorm2d
    #          nn.BatchNorm1d after fc layers:
    #          http://pytorch.org/docs/master/nn.html#batchnorm1d
    # This is a good resource for developing a CNN for classification:
    # http://cs231n.github.io/convolutional-networks/#layers

    import torch.nn as nn
    import torch.nn.functional as F

    class BaseNet(nn.Module):
        def __init__(self):
            super(BaseNet, self).__init__()
            
            # <<TODO#3>> Add more conv layers with increasing 
            # output channels
            # <<TODO#4>> Add normalization layers after conv
            # layers (nn.BatchNorm2d)

            # Also experiment with kernel size in conv2d layers (say 3
            # inspired from VGGNet)
            # To keep it simple, keep the same kernel size
            # (right now set to 5) in all conv layers.
            # Do not have a maxpool layer after every conv layer in your
            # deeper network as it leads to too much loss of information.

            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)

            # <<TODO#3>> Add more linear (fc) layers
            # <<TODO#4>> Add normalization layers after linear and
            # experiment inserting them before or after ReLU (nn.BatchNorm1d)
            # More on nn.sequential:
            # http://pytorch.org/docs/master/nn.html#torch.nn.Sequential
            
            self.fc_net = nn.Sequential(
                nn.Linear(16 * 5 * 5, TOTAL_CLASSES//2),
                nn.ReLU(inplace=True),
                nn.Linear(TOTAL_CLASSES//2, TOTAL_CLASSES),
            )

        def forward(self, x):

            # <<TODO#3&#4>> Based on the above edits, you'll have
            # to edit the forward pass description here.

            x = self.pool(F.relu(self.conv1(x)))
            # Output size = 28//2 x 28//2 = 14 x 14

            x = self.pool(F.relu(self.conv2(x)))
            # Output size = 10//2 x 10//2 = 5 x 5

            # See the CS231 link to understand why this is 16*5*5!
            # This will help you design your own deeper network
            x = x.view(-1, 16 * 5 * 5)
            x = self.fc_net(x)

            # No softmax is needed as the loss function in step 3
            # takes care of that
            
            return x

    # Create an instance of the nn.module class defined above:
    import resnet # 👈
    from vgg19 import vgg19 # 👈
    net=resnet.resnet50(pretrained=True) # 👈
    # net=vgg19(pretrained=True) # 👈

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if IS_GPU:
        net = net.cuda()
    # no change👇----------------------------------------------------------👇no change
    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Here we use Cross-Entropy loss and SGD with momentum.
    # The CrossEntropyLoss criterion already includes softmax within its
    # implementation. That's why we don't use a softmax in our model
    # definition.

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()

    # Tune the learning rate.
    # See whether the momentum is useful or not
    LEARNING_RATE = 0.001 # 👈
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.001) # 👈

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    plt.ioff()
    fig = plt.figure()
    train_loss_over_epochs = []
    val_accuracy_over_epochs = []
    # no change👇----------------------------------------------------------👇no change
    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize. We evaluate the validation accuracy at each
    # epoch and plot these values over the number of epochs
    # Nothing to change here
    # -----------------------------
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if IS_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, torch.nn.functional.one_hot(labels.type(torch.int64), num_classes=outputs.shape[1]).type(torch.float)) # 👈changed! 此处outputs是softmax的输出, labels是分类标签，不能直接比
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        # Normalizing the loss by the total number of train batches
        running_loss/=len(trainloader)
        print('[%d] loss: %.3f' %
            (epoch + 1, running_loss))

        # Scale of 0.0 to 100.0
        # Calculate validation set accuracy of the existing model
        net.eval()
        val_accuracy, val_classwise_accuracy = \
            calculate_val_accuracy(valloader, IS_GPU)
        print('Accuracy of the network on the val images: %d %%' % (val_accuracy))
        net.train()
        scheduler.step()
        print("Next learning rate:", scheduler.get_last_lr())
        
        # # Optionally print classwise accuracies
        # for c_i in range(TOTAL_CLASSES):
        #     print('Accuracy of %5s : %2d %%' % (
        #         classes[c_i], 100 * val_classwise_accuracy[c_i]))

        train_loss_over_epochs.append(running_loss)
        val_accuracy_over_epochs.append(val_accuracy)
    # -----------------------------


    # Plot train loss over epochs and val set accuracy over epochs
    # Nothing to change here
    # -------------
    plt.subplot(2, 1, 1)
    plt.ylabel('Train loss')
    plt.plot(np.arange(EPOCHS), train_loss_over_epochs, 'k-')
    plt.title('train loss and val accuracy')
    plt.xticks(np.arange(EPOCHS, dtype=int))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    # The line added for a bug fix.
    val_accuracy_over_epochs = torch.tensor(val_accuracy_over_epochs, device = 'cpu')

    plt.plot(np.arange(EPOCHS), val_accuracy_over_epochs, 'b-')
    plt.ylabel('Val accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(EPOCHS, dtype=int))
    plt.grid(True)
    plt.savefig("plot.png")
    plt.close(fig)
    print('Finished Training')
    # -------------
    # no change👇----------------------------------------------------------👇no change
    ########################################################################
    # 5. Try the network on test data, and create .csv file
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ########################################################################

    # Check out why .eval() is important!
    # https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744/2
    net.eval()

    total = 0
    predictions = []
    for data in testloader:
        images, labels = data

        # For training on GPU, we need to transfer net and data onto the GPU
        # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
        if IS_GPU:
            images = images.cuda()
            labels = labels.cuda()
        
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)

    with open('submission_netid.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id", "Prediction1"])
        for l_i, label in enumerate(predictions):
            wr.writerow([str(l_i), str(label)])
