#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from datasets_config import datasets_config
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from torchvision import transforms
from PIL import Image
import numpy as np
#########loader = get_dataloader()


# In[ ]:





# In[ ]:





# In[2]:


def get_img(file_name,dataset_path):
    file_dir = os.path.join(dataset_path,"Training100","ASOCT_Image",file_name)
    if os.path.exists(file_dir) == False:
        print("not exist :",file_dir)
    img = cv2.imread(file_dir)
    
    return img


# In[3]:


def show_label(img,point):
    point = (int(point[0]),int(point[1]))
    point_size = 15
    point_color = (255, 0, 0) # BGR
    thickness = 8 # 可以为 0 、4、8
    cv2.circle(img, point, point_size, point_color, thickness)
    
    plt.imshow(img)
    plt.show()


# In[ ]:





# In[4]:


class Augmentation(object):
    def __init__(self,datasets_config):
        super(Augmentation,self).__init__()
        self.rotate_angle = datasets_config['rotate_angle']
        self.GaussianBlur_sigma = datasets_config['GaussianBlur_sigma']
        
    def __call__(self, sample):
        image, point= sample['image'], sample['point']
        keypoints=ia.KeypointsOnImage([
                ia.Keypoint(x=int(point[0]), y=int(point[1]))], 
            shape=image.shape)
        
        seq=iaa.Sequential([
            iaa.Affine(
                rotate=(-self.rotate_angle,self.rotate_angle)),
            #iaa.GaussianBlur(
            #    sigma=iap.Uniform(0.0, self.GaussianBlur_sigma))
            ])
        # augmentation choices
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        keypoints = seq_det.augment_keypoints([keypoints])[0]
        return {'image': image_aug, 'point':(keypoints.keypoints[0].x,keypoints.keypoints[0].y)}
    
def get_transform():
    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               
                               
            
                           ])
    return transform
def addNoisy(img,Noisy_prob,SNR,sigma):
    prob = np.random.choice(range(0, 100))/100.0
    #print("prob:",prob)
    if prob < Noisy_prob:
        return spNoisy(img, SNR)
    else:
        return  GaussieNoisy(img,sigma)
    
def spNoisy(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(c, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声
    return img_
def GaussieNoisy(image,sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype(np.uint8)


# In[5]:


class AGE_DataSet(torch.utils.data.Dataset):
    def __init__(self,Training100_Location,datasets_config):
        super(AGE_DataSet,self).__init__()
        self.dataset_path = datasets_config['dataset_path']
        self.img_size = datasets_config['img_size']
        self.SNR = datasets_config['SNR']
        self.Noisy_prob = datasets_config['Noisy_prob']
        self.GaussianBlur_sigma = datasets_config['GaussianBlur_sigma']
        self.ASOCT_Name = Training100_Location['ASOCT_Name']
        self.Left_Label = Training100_Location['Left_Label']
        self.Right_Label = Training100_Location['Right_Label']
        self.X1 = Training100_Location['X1']
        self.Y1 = Training100_Location['Y1']
        self.X2 = Training100_Location['X2']
        self.Y2 = Training100_Location['Y2']
        self.len = len(Training100_Location)
        self.img_shape = get_img(Training100_Location['ASOCT_Name'][0],self.dataset_path).shape
        self.Augmentation = Augmentation(datasets_config)
        self.transform = get_transform()
        
        
    def resize_img(self,img,point,is_left):
        if is_left:
            img = img[0:self.img_size,0:self.img_size,:]
        else :
            shape = img.shape
            img = img[0:self.img_size,img.shape[1]-self.img_size-1:-1,:]
            point = (point[0] - (shape[1] - self.img_size),point[1])
        return img,point
    
    def __getitem__(self,index):
        idx = (index// 2)%self.len
        img = get_img(self.ASOCT_Name[idx],self.dataset_path)
        label = self.Left_Label[idx] if index % 2 == 0 else self.Left_Label[idx]
        point = (self.X1[idx],self.Y1[idx]) if index % 2 == 0 else (self.X2[idx],self.Y2[idx])
        img,point = self.resize_img(img,point,index%2 == 0)
        
        # use self.transform for input images
        sample = self.Augmentation({'image':img,'point':point})
        img,point = sample['image'],sample['point']
        img = addNoisy(img,self.Noisy_prob,self.SNR,self.GaussianBlur_sigma)
        img = self.transform(img)
        #return {'image':img,'point':point,'label':label}
        return (img,point,label)
    def __len__(self):
        return self.len*2


# In[ ]:





# In[ ]:





# In[6]:


def get_dataloader():
    dataset_path = datasets_config['dataset_path']
    Training100_Location_dir = os.path.join(dataset_path,'Training100','Training100_Location.xlsx')
    Training100_Location = pd.read_excel(Training100_Location_dir)
    num_workers = datasets_config['num_workers']
    batch_size = datasets_config['batch_size']
    age_dataset = AGE_DataSet(Training100_Location,datasets_config)
    dataloader=torch.utils.data.DataLoader(age_dataset,   
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=num_workers)
    return dataloader


# In[7]:


dataloader =     get_dataloader()
for i,(img,point,label) in enumerate(dataloader):
    print(point[0][0],point[0][1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




