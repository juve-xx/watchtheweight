#Combined with with mydata_numpy,run it in the cmd.
#NN2D1
# Tuning on the mean value
import os

import torch
import torch.nn as nn

import torch.optim as optim
import shutil
import sys

import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader



import createdatashufflemean
from MyData_numpy import MyDataset
from MyData_numpy import MyDataset_Test

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径



batch_size=64
lr=0.01
inputsize=2048
num=sys.argv[1]
# num=2
num=3*int(num)-1
##########################################handle data####################################
autosd=1
t=sys.argv[2]
# t=200
autosd=int(autosd)
t=int(t)/20
print(autosd)
print(t)

# device=torch.device('cuda')
dtype = torch.float32


criterion = nn.CrossEntropyLoss()



##########################################################train networks#########
class simulation_net(nn.Module):
    def __init__(self,batch_size,lr):
        super(simulation_net, self).__init__()
        self.Net_name="simulation4_auto"+str(autosd)
        self.num=num
        self.batch_size=batch_size
        self.accuracy=0
        self.lr=lr
        self.epoch=50
        self.method="SGD"
        self.PATH = "/home/r4user1/RMTLearn/simulation4/" + self.Net_name + "/batch" + str(self.batch_size) + "/" + str(self.lr) + "/" + self.method
        self.fc1 = nn.Linear(inputsize, 1024, bias=False)
        self.fc2=nn.Linear(1024,512,bias=False)

        self.fc3 = nn.Linear(512,self.num, bias=False)
        # self.fc6=nn.Linear(10,self.num, bias=False)

    def forward(self,x):
        x1 = self.fc1(x)
        x1_p=torch.relu(x1)
        x2 = self.fc2(x1_p)
        x2_p=torch.relu(x2)
        x3 = self.fc3(x2_p)

        # x=  torch.relu(self.fc3(x))
        # x = self.fc6(x)
        return x3



net = simulation_net(batch_size=batch_size,lr=lr)
# torch.nn.init.xavier_normal_(net.conv0.weight)

# torch.nn.init.xavier_normal_(net.fc1.weight)
# torch.nn.init.xavier_normal_(net.fc2.weight)
# torch.nn.init.xavier_normal_(net.fc3.weight)
# torch.nn.init.xavier_normal_(net.fc4.weight)
# torch.nn.init.xavier_normal_(net.fc5.weight)
# torch.nn.init.xavier_normal_(net.fc6.weight)
# torch.nn.init.xavier_normal_(net.fc4.weight)
# torch.nn.init.constant_(net.conv0.bias,0.1)

# torch.nn.init.constant_(net.fc1.bias,0.1)
# torch.nn.init.constant_(net.fc2.bias,0.1)
# torch.nn.init.constant_(net.fc3.bias,0.1)
#

net=net.cuda()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





optimizer = optim.SGD(net.parameters(), lr=net.lr,momentum=0.9)


PATH1=".pth"



def train_model(epoch,t):
    net.epoch=epoch

    for epoch in range(epoch):  # loop over the dataset multiple times
        if (((epoch%4)==0)|(epoch<=10)):
            print("model_saving:")
            PATH = net.PATH+"/label"+str(net.num)+"/"+str(t)+"/model" + str(epoch) + PATH1
            mkdir(net.PATH+"/label"+str(net.num)+"/"+str(t))
            torch.save(net, PATH)
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data

            inputs=inputs.to(device=device, dtype=dtype)
            labels=labels.to(device=device, dtype=torch.int64)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # print(torch.autograd.grad(loss, net.parameters(), retain_graph=True, create_graph=True))
            optimizer.step()

            running_loss += loss.item()
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
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # labels = labels.long()
                images = images.to(device=device, dtype=dtype)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = net(images)
                # print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Epoch %d\'s Accuracy : %d %%'  % (epoch+1,
                    100 * correct / total))
            net.accuracy=100 * correct / total
    print("model_saving:")
    PATH = net.PATH + "/model" + str(epoch+1) + PATH1
    # torch.save(net, PATH)

if __name__=="__main__":

    t=np.round(t,3)
    print(t)
    data_path=net.PATH + "/label" + str(net.num) + "/" + str(t)+'/'
    mkdir(net.PATH + "/label" + str(net.num) + "/" + str(t)+'/MyData')
    creat = createdatashufflemean.create_data(NUM=num,K0=num,R=1,inputsize=inputsize,t=t)
    creat.create()
    shutil.copy2('/home/r4user1/MyData/normal_input_train.npy',data_path+'MyData/normal_input_train.npy')
    shutil.copy2('/home/r4user1/MyData/normal_input_test.npy', data_path + 'MyData/normal_input_test.npy')
    shutil.copy2('/home/r4user1/MyData/normal_label_train.npy', data_path + 'MyData/normal_label_train.npy')
    shutil.copy2('/home/r4user1/MyData/normal_label_test.npy', data_path + 'MyData/normal_label_test.npy')

    transform = transforms.Compose(
        [transforms.ToTensor()]) 

    trainset = MyDataset()
    testset = MyDataset_Test()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_model(250, t)
    print(net.accuracy)





# for((i=1;i<4;i++))
# do
# for((j=8;j<20;j++))
# do
# python simulation4.py $((i)) $((j))
# done
# done


