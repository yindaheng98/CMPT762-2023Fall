from lab2 import *
if __name__ == "__main__":
    MODEL_PATH = "data/statedict-200-75.74.pth"
    IS_GPU = True
    TEST_BS = 128
    TOTAL_CLASSES = 100
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

        for data in tqdm(valloader):
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
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343) # 👈
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404) # 👈
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    valset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="val",
                                        download=True, transform=test_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=4)
    print("Val set size: "+str(len(valset)))

    testset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="test",
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS,
                                            shuffle=False, num_workers=4)
    print("Test set size: "+str(len(testset)))

    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        """Residual block for resnet over 50 layers

        """
        expansion = 4
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * ResBlock.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * ResBlock.expansion),
            )

            self.residual_fit = nn.Sequential()

            if stride != 1 or in_channels != out_channels * ResBlock.expansion:
                self.residual_fit = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * ResBlock.expansion, stride=stride, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels * ResBlock.expansion)
                )

        def forward(self, x):
            return nn.ReLU(inplace=True)(self.residual(x) + self.residual_fit(x))

    class BaseNet(nn.Module):

        def __init__(self):
            super().__init__()

            self.in_channels = 64

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            #we use a different inputsize than the original paper
            #so resblock1's stride is 1
            self.resblock1 = self.resblock(64, 3, 1)
            self.resblock2 = self.resblock(128, 4, 2)
            self.resblock3 = self.resblock(256, 6, 2)
            self.resblock4 = self.resblock(512, 3, 2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * ResBlock.expansion, TOTAL_CLASSES)

        def resblock(self, out_channels, num_blocks, stride):
            layers = []
            for stride in [stride] + [1] * (num_blocks - 1):
                layers.append(ResBlock(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * ResBlock.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            output = self.conv1(x)
            output = self.resblock1(output)
            output = self.resblock2(output)
            output = self.resblock3(output)
            output = self.resblock4(output)
            output = self.avgpool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)

            return output

    net=BaseNet()
    net.load_state_dict(torch.load(MODEL_PATH))
    if IS_GPU:
        net = net.cuda()

    from tqdm import tqdm

    net.eval()
    val_accuracy, val_classwise_accuracy = \
        calculate_val_accuracy(valloader, IS_GPU)
    print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

    total = 0
    predictions = []
    for data in tqdm(testloader):
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

    with open(MODEL_PATH+"-%.2f" % val_accuracy+'.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id", "Prediction1"])
        for l_i, label in enumerate(predictions):
            wr.writerow([str(l_i), str(label)])
