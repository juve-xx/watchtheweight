#This file generate the simulation data.
#sigma fixed, with mu different to get the different SNR, tuning by t.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt







# input0=np.array(range(0,4096))
class create_data():
    def __init__(self,NUM,K0,inputsize,R=1,t=1,n0=7500):
        self.t=t
        self.NUM=NUM
        self.K0=K0
        self.R=R
        self.inputsize=inputsize
        self.n0=n0
        self.nmax=int(n0/R)
        self.K_minority=NUM-K0

    def Init(self):
        mu = [-1 / self.NUM] * self.NUM
        mean0 = []
        for i in (range(self.NUM)):
            a = mu.copy()
            a[i] = 1 - 1 / self.NUM
            b = a + [0] * (self.inputsize - self.NUM)
            mean0 = mean0 + [b]
        self.mean0=mean0
        self.nk=[]
        for i in range(self.K_minority):
            n = max(int(max((1 - 0.1 * i),0.5) * self.nmax), 5)
            self.nk=self.nk+[n]


    def create(self):
        Mydata_input = []
        Mydata_label = []

        self.Init()
        print("Generating training sample")

        for judge in (range(self.K0)):
            for i in range(self.n0):
                mu_0=self.mean0[judge]
                mu_0=np.array(mu_0)*self.t



                input=np.array([])
                for j in mu_0[0:self.NUM]:
                    input=np.append(input,np.random.normal(loc=j,scale=1,size=1))
                input0=np.random.normal(loc=0,scale=1,size=self.inputsize-self.NUM)
                input=np.append(input,input0)




                label=judge-0.0
                input=input.tolist()

                Mydata_input.append(input)
                Mydata_label.append(label)
        for judge in range(self.K_minority):
            nk=self.nk[judge]
            for i in range(nk):
                mu_0=self.mean0[int(judge+self.K0)]
                mu_0=np.array(mu_0)*self.t



                input=np.array([])
                for j in mu_0[0:self.NUM]:
                    input=np.append(input,np.random.normal(loc=j,scale=1,size=1))
                input0=np.random.normal(loc=0,scale=1,size=self.inputsize-self.NUM)
                input=np.append(input,input0)




                label=judge+self.K0+0.0
                input=input.tolist()

                Mydata_input.append(input)
                Mydata_label.append(label)
        print('create_traindata_over')



        np.save('/home/r4user1/MyData/normal_input_train.npy',np.array(Mydata_input))
        np.save('/home/r4user1/MyData/normal_label_train.npy',np.array(Mydata_label))

        print("over")
        Mydata_input=[]
        Mydata_label=[]
        input=None


        print("Generating testing sample")
        for judge in (range(self.NUM)):
            for i in range(800):
                mu_0=self.mean0[judge]
                mu_0=np.array(mu_0)*self.t



                input=np.array([])
                for j in mu_0[0:self.NUM]:
                    input=np.append(input,np.random.normal(loc=j,scale=1,size=1))
                input0=np.random.normal(loc=0,scale=1,size=self.inputsize-self.NUM)
                input=np.append(input,input0)


                label=judge-0.0
                input=input.tolist()

                Mydata_input.append(input)
                Mydata_label.append(label)

        print('create_testdata_over')


        np.save('/home/r4user1/MyData/normal_input_test.npy',np.array(Mydata_input))
        np.save('/home/r4user1/MyData/normal_label_test.npy',np.array(Mydata_label))


if __name__=="__main__":
    creat = create_data(NUM=10,K0=10,R=1,n0=75)
    creat.create()
