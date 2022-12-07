import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

import numpy as np



lr = 0.01
method = 'SGD'
batch_size = 64
simulationname = 'simulation2'
r=248
this_im=2
num = 2
T = list(np.arange(0.02, 1.2, 0.01).round(3))




for t in T:


    whole_path = "/home/r4user1/RMTLearn/" + simulationname + '/' + \
                                   simulationname + '_auto1' + "/batch" + str(batch_size) + "/" + \
                                   str(lr) + "/" + method + '/label' + str(num) + \
                                   '/' + str(t) + '/' + 'Save_Sigma' + '/'
    oldfile=whole_path+'sigmaW_FC4.npy'
    targetfile=whole_path+'sigmaW_FC5.npy'

    os.rename(oldfile,targetfile)
