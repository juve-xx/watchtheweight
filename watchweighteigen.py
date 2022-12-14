import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from simulation1autosdhook import simulation_net
#from VGG import Net  # get different shapes of networks.
from spectracriterion import *

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

num=8  #The number of labels
epoch=248
C_name='/home/r4user1/RMTLearn/' + simulationname + '/' + 'picture' + '/fc'+str(this_im)+'_NUM='+str(num)+'/'
#
mkdir(C_name)
#t=0.9   # t is the tuning parameter of SNR
T = list(np.arange(0.01,1.19,0.01).round(3))

count=0
plt.figure(figsize=(8,7))
# plt.tight_layout()
# plt.subplots_adjust(wspace=0,hspace=2)
for t in T:
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

    READ_PATH = PATH_SAVE + '/model' + str(epoch) + '.pth'
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
    numspikes=detectspikes(evals_FC,alpha=5) #we may tune the alpha here to get the correct spikes.
    HS=numspikes[0]
    TS=numspikes[1]
    plot_esd_fit_mp(evals_FC,Headspikes=HS,Tailspikes=TS,alpha=0.25)
    S=Scriteria(eigenvalues=evals_FC,Headspikes=numspikes[0],Tailspikes=numspikes[1])

    length=len(evals_FC)-np.sum(numspikes)

    crit=0.6*np.sqrt((np.log(length)/np.power(length,2/3)))

    ax.set_title('t='+str(t))
    # if count==3:
    if t==checkt:
        print(evals_FC[0:10])
# plt.savefig(C_name+'epoch'+str(epoch)+'T1'+'.eps')
plt.show()

