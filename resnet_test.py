import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet34
from torchvision.transforms.transforms import Resize
 
transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
#采用自带的Cifar100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
 
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
net=models.resnet34(pretrained=True)
##迁移学习
for param in net.parameters(): #固定参数
    print(param.names)
    param.requires_grad = False
 
fc_inputs = net.fc.in_features #获得fc特征层的输入
net.fc = nn.Sequential(         #重新定义特征层，根据需要可以添加自己想要的Linear层
    nn.Linear(fc_inputs, 100),  #多加几层都没关系
    nn.LogSoftmax(dim=1)
)
 
net = net.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
 
# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
 
        print(batch_idx+1,'/', len(trainloader),'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx,'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
for epoch in range(100): #设置成100循环
    train(epoch)
torch.save(net.state_dict(),'resnet34cifar100.pkl') #训练完成后保存模型，供下次继续训练使用
print('begin  test ')
 
for epoch in range(5): #测试5次
    test(epoch)