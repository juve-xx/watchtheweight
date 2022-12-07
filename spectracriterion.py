import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
# from MiniAlexnet import Net
from simulation3autosdhook import simulation_net
#from simulation3autosdhook import simulation_net     #  get different shapes of networks.
#
from MyData_numpy import MyDataset
from MyData_numpy import MyDataset_Test
import scipy.integrate as integrate

from scipy.linalg import svd
import shutil


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  
    return path

def marchenko_pastur_pdf(x_min,x_max,Q,sigma=1):
    y=1/Q
    x=np.arange(x_min,x_max,0.001)
    if y<=1:
        b = np.power(sigma * (1 + np.sqrt(1/Q)),2)
        a = np.power(sigma * (1 - np.sqrt(1/Q)),2)
        return x,(1/(2*np.pi*sigma*sigma*x*y))*np.sqrt(np.abs((b-x)*(x-a)))
    if y > 1:
        b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
        a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
        return x, (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt(np.abs((b-x)*(x-a)))*Q

def get_sigma_Q1(x_min,x_max): #Q<1
    if x_min<=0:
        raise Exception('Xmin should be larger than 0')
    if x_max<=x_min:
        raise Exception('Xmax should be larger than xmin')
    Q=1/np.power(2/(1-np.sqrt(x_min/x_max))-1,2)
    sigma=np.sqrt(x_min/np.power(2/(1-np.sqrt(x_min/x_max))-2,2))
    return Q,sigma


def get_sigma_Q0(x_min,x_max): #Q>1
    if x_min<=0:
        raise Exception('Xmin should be larger than 0')
    if x_max<=x_min:
        raise Exception('Xmax should be larger than xmin')
    Q=1/np.power(2/(1+np.sqrt(x_min/x_max))-1,2)
    sigma=np.sqrt(x_min/np.power(2/(1+np.sqrt(x_min/x_max))-2,2))
    return Q,sigma



def plot_esd_fit_mp(eigenvalues=None, Headspikes=0, Tailspikes=0,alpha=0.25, color='blue'):
    if eigenvalues is None:
        return 0
    length=len(eigenvalues)
    evals=np.sort(eigenvalues)[::-1][Headspikes:(length-Tailspikes)]
    x_min,x_max=np.min(evals),np.max(evals)
    plt.hist(eigenvalues,bins=80,alpha=alpha,color=color,density=True,label=r'$\rho_{emp}(\lambda)$')
    if x_min==0:
        x_min1=np.min(evals[evals>0])
        Q,sigma=get_sigma_Q1(x_min1,x_max)
        x,mp=marchenko_pastur_pdf(x_min1,x_max,Q,sigma)

    elif x_min>0:
        Q,sigma=get_sigma_Q0(x_min,x_max)
        x,mp=marchenko_pastur_pdf(x_min,x_max,Q,sigma)
    else:
        raise Exception('Xmin not correct')

    plt.plot(x,mp,linewidth=1, color='r', label='MP fit')
    plt.yticks(fontproperties='Times New Roman', size=6)
    plt.xticks(fontproperties='Times New Roman', size=6)
    return sigma



#############detect the spikes numbers###########
def detectspikes(eigenvalues, alpha=4):
    length=len(eigenvalues)
    eigenvalues=-np.sort(-eigenvalues)
    diff_eigen=-np.diff(eigenvalues)

    threshold=alpha*np.mean(diff_eigen)
    # print(diff_eigen)
    # print(threshold)

    Head=diff_eigen[0:int(length/2)]
    Tail=diff_eigen[int(length/2):(length-1)]

    if (np.sum((Head>threshold)==False)==len(Head)):
        Headspikes=0
    else:
        Headspikes=np.max(np.where((Head>threshold)==True))+1

    if (np.sum((Tail > threshold) == False) == len(Tail)):
        Tailspikes=0
    else:
        Tailspikes=len(Tail)-np.max(np.where((Tail > threshold) == True))

    return [int(Headspikes),int(Tailspikes)]


############MP Law function given eigenvalues##############
def MPLaw(eigenvalues,Headspikes=0,Tailspikes=0):
    length=len(eigenvalues)
    evals = np.sort(eigenvalues)[::-1][Headspikes:(length - Tailspikes)]
    length=len(evals)
    x_min, x_max = np.min(evals), np.max(evals)
    if x_min==0:
        x_min1=np.min(evals[evals>0])
        Q,sigma=get_sigma_Q1(x_min1,x_max)

    elif x_min>0:
        Q,sigma=get_sigma_Q0(x_min,x_max)
    else:
        raise Exception('Xmin not correct')

    y=1/Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    if y<=1:
        f=lambda x:(1/(2*np.pi*sigma*sigma*x*y))*np.sqrt(np.abs((b-x)*(x-a)))
    else:
        b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
        a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
        f=lambda x:(1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt(np.abs((b-x)*(x-a)))*Q

    return f

def Scriteria(eigenvalues,Headspikes=0,Tailspikes=0):
    length = len(eigenvalues)
    evals = np.sort(eigenvalues)[::-1][Headspikes:(length - Tailspikes)]
    length = len(evals)
    f=MPLaw(eigenvalues,Headspikes=Headspikes,Tailspikes=Tailspikes)
    h=2*int(np.power(length,1/3))

    interval=int(length/h)
    S_diff=0
    evals=np.sort(evals)

    for i in range(h-1):
        a=i*interval
        b=(i+1)*interval
        High=(b-a)/length/(evals[b]-evals[a])
        g=lambda x:np.abs(f(x)-High)
        s=integrate.quad(g,evals[a],evals[b])[0]
        S_diff=S_diff+s
        # S_diff=S_diff+stheory[0]

        # print(f(evals[i*h]))

    a=(h-1)*interval
    b=length-1
    High = (b - a) / length / (evals[b] - evals[a])
    g = lambda x: np.abs(f(x) - High)
    s = integrate.quad(g, evals[a], evals[b])[0]
    S_diff = S_diff + s
    return S_diff

T = list(np.arange(0.01,1.19,0.01).round(3))
R=[1,2,3,4,5,6,7,8,9,10]+ list(np.arange(12,252,4).round(0))
# R=list(np.arange(180,252,4).round(0))
xlim=[]
ylim=[]
t=2.4
count=0
if __name__=="__main__":
    for r in R:
        # PATH_SAVE = "/home/r4user1/RMTLearn/AlexNet/batch256/0/model"+str(r)+".pth"
        PATH_SAVE = "/home/r4user1/RMTLearn/simulation3/simulation3_auto1/batch64/0.01/SGD/label8/" + str(
            t) + "/model" + str(r) + ".pth"
        Net = simulation_net
        this_im=1
        model = Net # keep the same of simulation net.
        model = torch.load(PATH_SAVE, map_location='cpu')
        NET=model


        for im, m in enumerate(NET.modules()):
            # print(im,m)
            if im == this_im:
                hh, hhh = im, m
                Weight = np.array(m.weight.data.clone().cpu())
                # W=Weight
                W = Weight.T

        u,sv,sh=svd(Weight)
        evals_FC=sv*sv

        numspikes=detectspikes(evals_FC,alpha=10)
        print(numspikes)
        print(NET.accuracy)
        S=Scriteria(eigenvalues=evals_FC,Headspikes=numspikes[0],Tailspikes=numspikes[1])
        print("epochs=",r,S)
        # xlim.append(t)
        # ylim.append(S)
        length=len(evals_FC)-np.sum(numspikes)

        crit=0.6*np.sqrt((np.log(length)/np.power(length,2/3)))
        if S<=crit:
            count=0
        else:
            count=count+1
        if count==3:
            break
        # plot_esd_fit_mp(eigenvalues=evals_FC, Headspikes=numspikes[0], Tailspikes=numspikes[1], alpha=0.25)
        # plt.show()
        # print("Training:",NET.trainaccuracy)
        # print("\n")
    print("Scriteria:", S)
    print(crit)
    print(length)
    print("Testing:", NET.accuracy)
    plot_esd_fit_mp(eigenvalues=evals_FC, Headspikes=numspikes[0], Tailspikes=numspikes[1], alpha=0.25)
    plt.show()



