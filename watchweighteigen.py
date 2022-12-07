import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from simulation1autosdhook import simulation_net
#from VGG import Net  # get different shapes of networks.
from  import *

import shutil
from scipy.linalg import svd
from MyData_numpy import MyDataset
from MyData_numpy import MyDataset_Test
from torch.utils.data import DataLoader


device=torch.device('cuda')
dtype = torch.float32





########################The auto figure of sigma and distance############

lr=0.01
method='SGD'
batch_size=64
simulationname = 'simulation2'
this_im=3
# this_im=1
num=8  #The number of labels
r=248
C_name='/home/r4user1/RMTLearn/' + simulationname + '/' + 'picture' + '/fc'+str(this_im)+'_NUM='+str(num)+'/'
#
mkdir(C_name)
t=0.9   # the tuning parameter of SNR
checkt=3.28   #the specific checked number.
T = list(np.arange(0.91,0.94,0.01).round(3))
# T1=np.array(random.sample(ra nge(76,83),6))*0.005
T1=[0.24,0.64,0.8,1.04,2.0,checkt]
# T1=np.sort(np.random.choice(T1,size=6))
T2=np.arange(3.12,4.8,0.08).round(3)
T2=np.sort(np.random.choice(T2,size=6))
# T1=[0.08,0.96,1.44,1.68,2.24,2.4]
# T2=[2.48,2.96,3.2,3.6,4.0,4.24]
T3=[0.01,0.12,0.07,0.16]
count=0
plt.figure(figsize=(8,7))
# plt.tight_layout()
# plt.subplots_adjust(wspace=0,hspace=2)
for r in [16,28,32,248]:
    print(t)
    # t=t.round(3)
    count=count+1
    auto_num=1
    auto_name = simulationname+'_auto'+str(auto_num)
    xlim=[]
    ylim=[]
    #for r in list(np.arange(1,10,1).round(0))+list(np.arange(12,250,4).round(0)):
    PATH_SAVE = "/home/r4user1/RMTLearn/" + simulationname+'/'+ auto_name + "/batch" \
                + str(batch_size) + "/" + str(lr) + "/" + method + '/' + 'label' + str(num)+'/'+str(t)

    # PATH_SAVE = "/home/r4user1/neuralcollapes/" + Net_name + "/batch" \
    #             + str(batch_size) + "/" + str(lr) + "/" + method + '/' +'sdtime0'+'/'+ 'label' + str(num)

    READ_PATH = PATH_SAVE + '/model' + str(r) + '.pth'
    model = simulation_net  # keep the same of simulation net.
    model = torch.load(READ_PATH, map_location='cpu')
    NET=model
    print('Testing Accuracy:',NET.accuracy)
    shutil.copy2(PATH_SAVE+'/MyData/normal_input_train.npy','/home/r4user1/MyData/normal_input_train.npy')
    shutil.copy2(PATH_SAVE + '/MyData/normal_input_test.npy','/home/r4user1/MyData/normal_input_test.npy')
    shutil.copy2(PATH_SAVE + '/MyData/normal_label_train.npy','/home/r4user1/MyData/normal_label_train.npy')
    shutil.copy2(PATH_SAVE + '/MyData/normal_label_test.npy','/home/r4user1/MyData/normal_label_test.npy')
    trainset = MyDataset()
    testset = MyDataset_Test()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
    with torch.no_grad():
        correct = 0
        total = 0
        net=NET
        net.cuda()
        for data in trainloader:
            images, labels = data
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Training Accuracy : %d %%' % (
                                                100 * correct / total))


    for im, m in enumerate(NET.modules()):
        print(im,m)
        if im == this_im:
            hh,hhh=im,m
            Weight = np.array(m.weight.data.clone().cpu())
            # W=Weight
            W=Weight.T

    # print(W)
    u,sv,sh=svd(Weight)
    evals_FC=sv*sv
    # mkdir(C_name + 'auto' + str(auto_num) + '/' + 'sigma=' + str(t) + '/')

    figpara=int(230+count)
    ax=plt.subplot(figpara)
    plot_esd_fit_mp(evals_FC,num_spikes=(num),alpha=0.25)
    # plot_esd_fit_mp(evals_FC[num:(len(evals_FC)-2*num)], num_spikes=num, alpha=0.25)

    # plot_esd_fit_mp(evals_FC[num:len(evals_FC)], num_spikes=0, alpha=0.25)

    ax.set_title('t='+str(t))
    # if count==3:
    if t==checkt:
        print(evals_FC[0:10])
# plt.savefig(C_name+'epoch'+str(r)+'T1'+'.eps')
plt.show()

