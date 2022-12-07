
import numpy as np


class AddGaussianNoise(object):
    def __init__(self,mean=0.0,variance=1,amplitude=1.0):
        self.mean=mean
        self.variance=variance
        self.amplitude=amplitude

    def __call__(self, img):
        img=np.array(img)
        # print(np.max(img))
        # print(img.shape)
        c,h,w=img.shape
        # print(img.shape)
        N=np.random.normal(loc=self.mean,scale=self.variance,size=(c,h,w))
        # N=np.repeat(N,c,axis=0)
        # print('Nshape:',N.shape)
        # print('Imgshape:',img.shape)
        # print(np.max(N))
        # print('dka',np.max(img))
        img=N+img/self.amplitude
        img[img>255]=255
        # img.reshape(1,h,w)
        # img=Image.fromarray(img.astype('uint8'))
        return img




class AddSaltNoise(object):
    def __init__(self,density=0):
        self.density=density

    def __call__(self, img):
        img=np.array(img)
        # print(np.max(img))
        # print(img.shape)
        c,h,w=img.shape
        Nd=self.density
        Sd=1-Nd
        mask=np.random.choice((0,1,2),size=(c,h,w),p=[Nd/2.0,Nd/2.0,Sd])
        img[mask==0]=0
        img[mask==1]=255
        # img.reshape(1,h,w)
        # img=Image.fromarray(img.astype('uint8'))
        return img

class AddGaussianNoiseReshape(object):
    def __init__(self,mean=0.0,variance=1,amplitude=1.0):
        self.mean=mean
        self.variance=variance
        self.amplitude=amplitude

    def __call__(self, img):
        img=np.array(img)
        # print(np.max(img))
        # print(img.shape)
        c,h,w=img.shape
        # print(img.shape)
        N=self.amplitude*np.random.normal(loc=self.mean,scale=self.variance,size=(c,h,w))
        # N=np.repeat(N,c,axis=0)
        # print('Nshape:',N.shape)
        # print('Imgshape:',img.shape)
        # print(np.max(N))
        # print('dka',np.max(img))
        img=N+img
        img[img>255]=255
        # img.reshape(1,h,w)
        # img=Image.fromarray(img.astype('uint8'))
        # img=img.reshape((-1,))
        # print(len(img))
        return img

