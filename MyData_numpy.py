#Combine the generated data with torch.dataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable
import torchvision.datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
import json
transform = transforms.Compose(
    [transforms.ToTensor()])

#The path is exactly the generated data stored. Readers may setup his/her own path.

class MyDataset(Dataset):
    def __init__(self):

        self.inputs=np.load('/home/r4user1/MyData/normal_input_train.npy')
        self.label = np.load('/home/r4user1/MyData/normal_label_train.npy')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input=self.inputs[idx,]
        label=self.label[idx,]
        return input,label
class MyDataset_Test(Dataset):
    def __init__(self):

        self.inputs=np.load('/home/r4user1/MyData/normal_input_test.npy')
        self.label = np.load('/home/r4user1/MyData/normal_label_test.npy')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input=self.inputs[idx,]
        label=self.label[idx,]
        return input,label


if __name__ == '__main__':

    print("reading data")
    dataset = MyDataset()
    print(len(dataset.label))
    print(dataset.inputs.shape)
    print(dataset.inputs.dtype)
    print(dataset.label.dtype)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # print(next(iter(dataloader)))
    for i,data in enumerate(dataloader):
        inputs,labels=data
        print(i,labels.size(),inputs.size())
