import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tensorflow
import sys
from torch.autograd import Variable
import torchvision.datasets

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import models
import my_transform
# net=models.alexnet(pretrained=True)
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

method="SGD"
lr=0.01
batch=sys.argv[1]
batch=int(batch)
batchsize=int(np.power(2,batch+4))
batch_size=batchsize
Net_name="AlexNetMNIST"
t=0

PATH_SAVE="/home/r4user1/RMTLearn/AlexNetMNIST/batch"+str(batch_size)


device=torch.device('cuda')
dtype = torch.float32

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(28),
     transforms.Normalize((0.5,), (0.5, ))])#这里要观察数据channel是一还是三

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
criterion = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 5,stride=1,padding=2)# channels, output, (height * width)
        self.B1=nn.BatchNorm2d(96,affine=False) # (imagesize-kernalsize+2*padding)/stride+1
        self.pool1 = nn.MaxPool2d(3,stride=3,padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(96, 256, 5,stride=1,padding=2)
        self.B2 = nn.BatchNorm2d(256, affine=False)
        self.fc1 = nn.Linear(4096, 384,bias=False)
        self.fc2 = nn.Linear(384, 192,bias=False)
        self.fc3 = nn.Linear(192, 10,bias=False)
        self.accuracy=0
        self.trainaccuracy=0
        self.loss=0
    def forward(self, x):
        # print(x.size())
        x = torch.relu(self.conv1(x))
        # print(x.size())
        # x=self.B1(x)
        x=self.pool1(x)
        # print(x.size())
        x=torch.relu(self.conv2(x))
        # print(x.size())
        # x = self.B2(x)
        x = self.pool2(x)
        # print(x.size())
        x0 = x.view(-1,4096)
        # print(x.size())
        x1_p = self.fc1(x0)
        x1=torch.relu(x1_p)
        x2_p = self.fc2(x1)
        x2=torch.relu(x2_p)
        x = self.fc3(x2)
        return x

net = Net()
# torch.nn.init.xavier_normal_(net.conv0.weight)
torch.nn.init.xavier_normal_(net.conv1.weight)
torch.nn.init.xavier_normal_(net.conv2.weight)
torch.nn.init.xavier_normal_(net.fc1.weight)
torch.nn.init.xavier_normal_(net.fc2.weight)
torch.nn.init.xavier_normal_(net.fc3.weight)

# torch.nn.init.constant_(net.conv0.bias,0.1)
torch.nn.init.constant_(net.conv1.bias,0.1)
torch.nn.init.constant_(net.conv2.bias,0.1)
# torch.nn.init.constant_(net.fc1.bias,0.1)
# torch.nn.init.constant_(net.fc2.bias,0.1)
# torch.nn.init.constant_(net.fc3.bias,0.1)
#

net=net.cuda()
#模型定义完毕


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.9)

# PATH0="D:/torchmodel/batch32/MiniAlexnet/SGD/model"
PATH1=".pth"


def train_model(epoch0,save_path):
    mkdir(save_path)
    for epoch in range(epoch0):  # loop over the dataset multiple times

        PATH0 = save_path + "/model" + str(epoch) +".pth"
        if (((epoch % 4) == 0) | (epoch <= 10)):
            print("model_saving:")
            torch.save(net, PATH0)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            # hbduauh=np.array(inputs)
            # print(np.max(hbduauh))
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # print(torch.autograd.grad(loss, net.parameters(), retain_graph=True, create_graph=True))
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            net.loss=running_loss/2000
            if (i%200==199):    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                # dataiter = iter(testloader)
                # images, labels = dataiter.next()
                # print(net(images))
                running_loss = 0.0

        print('Finished Training')
        print("predicting")

        if (((epoch % 4) == 3) | (epoch <= 10)):
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    images, labels = data
                    images = images.to(device=device, dtype=dtype)
                    labels = labels.to(device=device, dtype=torch.long)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Epoch %d\'s Accuracy : %d %%'  % (epoch+1,
                        100 * correct / total))
                net.accuracy = 100 * correct / total
        if (((epoch % 4) == 3) | (epoch <= 10)):
            with torch.no_grad():
                correct = 0
                total = 0
                for data in trainloader:
                    images, labels = data
                    images = images.to(device=device, dtype=dtype)
                    labels = labels.to(device=device, dtype=torch.long)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Epoch %d\'s Accuracy : %d %%'  % (epoch+1,
                        100 * correct / total))
                net.trainaccuracy = 100 * correct / total


# train_model(2,PATH_SAVE)

if __name__=='__main__':
    t=np.round(t,3)
    PATH=PATH_SAVE+'/'+str(t)
    train_model(249,PATH)

#
# for((j=0;j<3;j++))
# do
# python MiniAlexnetMNIST.py  $((j))
# done