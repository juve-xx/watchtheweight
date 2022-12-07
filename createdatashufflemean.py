#这个文件用来生成numpy文件的模拟
#self.inputsize0 -0.2    and  self.inputsize0 -0.2+delta (shuffled)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import random

import scipy as sp
from scipy.linalg import svd
import pandas as pd

from mpmath import gamma
import matplotlib
import numpy as np
import matplotlib.pyplot as plt







# input0=np.array(range(0,self.inputsize))
class create_data():
    def __init__(self, NUM, K0, R, inputsize, t=1, n0=7500):
        self.t = t
        self.NUM = NUM
        self.K0 = K0
        self.R = R
        self.inputsize=inputsize
        self.inputsize0 = int(self.inputsize / 2)
        self.n0 = n0
        self.nmax = int(n0 / R)
        self.K_minority = NUM - K0

    def Init(self):
        self.nk = []
        for i in range(self.K_minority):
            n = max(int(max((1 - 0.1 * i), 0.5) * self.nmax), 5)
            self.nk = self.nk + [n]


    def create(self):
        Mydata_input_train = []
        Mydata_label_train = []
        Mydata_input_test = []
        Mydata_label_test = []
        N = range(0, self.inputsize)
        self.Init()
        position = np.array(range(0, self.inputsize))
        input0 = np.random.normal(loc=-0.2, scale=1, size=self.inputsize)
        for judge in range(self.K0):
            position11 = np.random.choice(N, size=self.inputsize0, replace=False)
            position12 = np.delete(position.copy(), position11.copy())
            for i in range(self.n0):
                X1 = np.random.normal(loc=-0.2, scale=1, size=self.inputsize0)
                X2 = np.random.normal(loc=-0.2 + self.t, scale=1, size=self.inputsize0)
                input = input0
                input[position11] = X1
                input[position12] = X2
                label = judge - 0.0
                input = input.tolist()

                Mydata_input_train.append(input)
                Mydata_label_train.append(label)
            for i in range(800):
                X1 = np.random.normal(loc=-0.2, scale=1, size=self.inputsize0)
                X2 = np.random.normal(loc=-0.2 + self.t, scale=1, size=self.inputsize0)
                input = input0
                input[position11] = X1
                input[position12] = X2
                label = judge - 0.0
                input = input.tolist()

                Mydata_input_test.append(input)
                Mydata_label_test.append(label)

        for judge in range(self.K_minority):
            nk=self.nk[judge]
            position11 = np.random.choice(N, size=self.inputsize0, replace=False)
            position12 = np.delete(position.copy(), position11.copy())
            for i in range(nk):
                X1 = np.random.normal(loc=-0.2, scale=1, size=self.inputsize0)
                X2 = np.random.normal(loc=-0.2 + self.t, scale=1, size=self.inputsize0)
                input = input0
                input[position11] = X1
                input[position12] = X2
                label = judge+self.K0 - 0.0
                input = input.tolist()

                Mydata_input_train.append(input)
                Mydata_label_train.append(label)
            for i in range(800):
                X1 = np.random.normal(loc=-0.2, scale=1, size=self.inputsize0)
                X2 = np.random.normal(loc=-0.2 + self.t, scale=1, size=self.inputsize0)
                input = input0
                input[position11] = X1
                input[position12] = X2
                label = judge +self.K0+ 0.0
                input = input.tolist()

                Mydata_input_test.append(input)
                Mydata_label_test.append(label)



        np.save('/home/r4user1/MyData/normal_input_train.npy',np.array(Mydata_input_train))
        np.save('/home/r4user1/MyData/normal_label_train.npy',np.array(Mydata_label_train))

        np.save('/home/r4user1/MyData/normal_input_test.npy',np.array(Mydata_input_test))
        np.save('/home/r4user1/MyData/normal_label_test.npy',np.array(Mydata_label_test))

if __name__=="__main__":
    creat = create_data(NUM=10,K0=4,R=20,n0=750)
    creat.create()