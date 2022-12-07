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
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import models
import my_transform
# net=models.alexnet(pretrained=True)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  # makedirs create the path if it does not exist

method="SGD"
lr=0.01
# batch=sys.argv[1]
# batch=int(batch)
#
# batch_size=int(np.power(2,batch+4))
batch_size=16
Net_name="VGGMNIST"

t=0

PATH_SAVE="/home/r4user1/RMTLearn/VGGMNIST/batch"+str(batch_size)
# t = np.round(t, 3)
PATH = PATH_SAVE + '/' + str(t)
device=torch.device('cuda')
dtype = torch.float32


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(32),
     transforms.Normalize((0.5,), (0.5, ))
     ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)







class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(3, 3, kernel_size=3,padding=1)# channels, output, (height * width)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3,padding=1)
        self.conv22=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv33= nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv55 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*32*4, 500,bias=False)
        # self.fc2 = nn.Linear(2450, 500,bias=False)
        self.fc2 = nn.Linear(500, 10,bias=False)
        self.accuracy=0
        self.trainaccuracy=0
        self.loss=1
    def forward(self, x):
        # print(x.shape)
        x= self.conv1(x)
        # print(x.shape)
        x = self.pool(F.relu(self.conv22(self.conv2(x))))
        x = self.pool(F.relu(self.conv33(self.conv3(x))))
        x=self.conv4(x)
        x=self.pool(F.relu(self.conv44(x)))

        x = self.pool(F.relu(self.conv5(x)))
        x0 = x.view(-1,16*32*4)
        # print(x0.shape)
        # print(x.shape)
        x1_p = self.fc1(x0)
        x1 = torch.relu(x1_p)
        x = self.fc2(x1)
        # x2 = torch.relu(x2_p)
        # x = self.fc3(x2)
        return x

net = Net()


# # torch.nn.init.xavier_normal_(net.conv0.weight)
# torch.nn.init.xavier_normal_(net.conv1.weight)
# torch.nn.init.xavier_normal_(net.conv2.weight)
# torch.nn.init.xavier_normal_(net.fc1.weight)
# torch.nn.init.xavier_normal_(net.fc2.weight)
# torch.nn.init.xavier_normal_(net.fc3.weight)
#
# # torch.nn.init.constant_(net.conv0.bias,0.1)
# torch.nn.init.constant_(net.conv1.bias,0.1)
# torch.nn.init.constant_(net.conv2.bias,0.1)
# # torch.nn.init.constant_(net.fc1.bias,0.1)
# # torch.nn.init.constant_(net.fc2.bias,0.1)
# # torch.nn.init.constant_(net.fc3.bias,0.1)
# #

net=net.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)


mkdir(PATH_SAVE)
def train_model(epoch0,save_path):
    mkdir(save_path)
    for epoch in range(epoch0):  # loop over the dataset multiple times
        print(epoch)
        PATH0 = save_path + "/model" + str(epoch) +".pth"
        if (((epoch % 4) == 0)|(epoch<=10)):
            print("model_saving:")
            torch.save(net, PATH0)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
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
            net.loss = running_loss / 2000
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

                # dataiter = iter(testloader)
                # images, labels = dataiter.next()
                # print(net(images))
                running_loss = 0.0
        print('Finished Training')
        print("predicting")
        correct = 0
        total = 0
        if (((epoch % 4) == 3)|(epoch<=10)):
            with torch.no_grad():
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
                net.accuracy=100 * correct / total
        if (((epoch % 4) == 3)|(epoch<=10)):
            with torch.no_grad():
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
                net.trainaccuracy=100 * correct / total


if __name__=='__main__':

    train_model(249,PATH)


# for((j=4;j>0;j--))
# do
# python VGGMNIST.py  $((j))
# done
