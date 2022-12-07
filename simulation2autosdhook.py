

import os

import torch
import torch.nn as nn

import torch.optim as optim
import shutil


import torchvision.transforms as transforms
import numpy as np


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  # makedirs 



batch_size=64
lr=0.01


num=5

autosd=int(1)

# device=torch.device('cuda')
dtype = torch.float32


criterion = nn.CrossEntropyLoss()


class simulation_net(nn.Module):
    def __init__(self,batch_size,lr):
        super(simulation_net, self).__init__()
        self.Net_name="simulation1_auto"+str(autosd)
        self.num=num
        self.batch_size=batch_size
        self.accuracy=0
        self.lr=lr
        self.epoch=50
        self.method="SGD"
        self.PATH = "/home/r4user1/RMTLearn/simulation1/" + self.Net_name + "/batch" + str(self.batch_size) + "/" + str(self.lr) + "/" + self.method
        self.fc1 = nn.Linear(100, 1024, bias=False)
        self.fc2=nn.Linear(1024,512,bias=False)
        self.fc3 = nn.Linear(512, 384, bias=False)
        self.fc4=nn.Linear(384,192, bias=False)
        self.fc5 = nn.Linear(192,self.num, bias=False)
        # self.fc6=nn.Linear(10,self.num, bias=False)

    def forward(self,x):
        x1 = self.fc1(x)
        x1_p=torch.relu(x1)
        x2 = self.fc2(x1_p)
        x2_p=torch.relu(x2)
        x3 = self.fc3(x2_p)
        x3_p=torch.relu(x3)
        x4 = (self.fc4(x3_p))
        x5=self.fc5(x4)
        # x=  torch.relu(self.fc3(x))
        # x = self.fc6(x)
        return x5
#上面定义了网络结构


net = simulation_net(batch_size=batch_size,lr=lr)

#

net=net.cuda()













