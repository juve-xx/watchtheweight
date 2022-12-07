

# Tuning on the mean value
import os

import torch
import torch.nn as nn

import torch.optim as optim
import shutil
import sys

import torchvision.transforms as transforms
import numpy as np




batch_size=64
lr=0.01
inputsize=2048
num=5

autosd=1
t=1



class simulation_net(nn.Module):
    def __init__(self,batch_size,lr):
        super(simulation_net, self).__init__()
        self.Net_name="simulation3_auto"+str(autosd)
        self.num=num
        self.batch_size=batch_size
        self.accuracy=0
        self.lr=lr
        self.epoch=50
        self.method="SGD"
        self.PATH = "/home/r4user1/RMTLearn/simulation3/" + self.Net_name + "/batch" + str(self.batch_size) + "/" + str(self.lr) + "/" + self.method
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


net=net.cuda()




