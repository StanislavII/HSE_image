#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import Guided_filter

class DCP(object):
    
    def __init__(self, image_path, kernel_size, tx, r, eps, omega):
        
        self.image_path = image_path
        self.ksz = kernel_size
        self.radius = r
        self.eps = eps
        self.recover_bound = tx
        self.omega = omega
        
        
    def Image_Open(self):
        
        if isinstance(self.image_path, str):
            
            self.img_ = cv2.imread(self.image_path)
            self.image = self.img_.astype('float64')/255
            
        else:
            
            self.img_ = self.image_path
            self.image = self.image_path.astype('float64')/255
        
    def DarkChannel(self, image = None):
        
        if image is None:
            
            B,G,R = cv2.split(self.image)
        else:
            
            B,G,R = cv2.split(image)
            
        dark_channel = cv2.min(cv2.min(R,G),B)
        kernel = np.ones((self.ksz, self.ksz))
        self.erode_dark = cv2.erode(dark_channel, kernel)
        
        return self.erode_dark
    
    def AtmosphericLight(self):
        
        (h,w) = self.image.shape[:2]
        pixls = int(max(np.floor(h*w/1000),1))
        dark_v = self.erode_dark.reshape(h*w)
        im_v = self.image.reshape(h*w,3)
        indexes = dark_v.argsort()
        indexes = indexes[h*w-pixls::]
        
        atmospheric_sum = np.zeros((1,3))
        for i in range(1,pixls):
            atmospheric_sum += im_v[indexes[i]]
        
        self.Atmos = atmospheric_sum/pixls
        
        return self.Atmos
    
    def Transmission_State(self):
        
        image_ = np.empty(self.image.shape, self.image.dtype)
        
        for i in range(0,3):
            
            image_[:,:,i] = self.image[:,:,i]/self.Atmos[0,i]
            
        self.transmission = 1 - self.omega * self.DarkChannel(image = image_)
        return self.transmission 
    
    def Transmission_Clean(self):
        gray = cv2.cvtColor(self.img_,cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        guid_element = Guided_filter.GuidedFilter(gray, self.transmission, self.radius, self.eps)
        self.transmission_new = guid_element.apply()

        return self.transmission_new
    

    def Recover(self):
        self.restructured = np.empty(self.image.shape, self.image.dtype)
        t_max = cv2.max(self.transmission_new, self.recover_bound)

        for i in range(0,3):
            self.restructured[:,:,i] = (self.image[:,:,i] - self.Atmos[0,i])/t_max + self.Atmos[0,i]

        return self.restructured
    
    def apply(self):
        
        self.Image_Open()
        self.DarkChannel()
        self.AtmosphericLight()
        self.Transmission_State()
        self.Transmission_Clean()
        
        output = self.Recover()
        return output

