#!/usr/bin/python
# encoding: utf-8

import os, sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
#from PIL import Image
sys.path.append(os.path.abspath('../pytorch-yolo3'))
from utils import read_truths_args, read_truths
#from image import *

class listDataset(Dataset):

    def __init__(self, imgs, labels, shape=None, shuffle=False, predict=False,
                    box_size=7):
       self.imgs = imgs
       self.labels = labels
       self.nSamples  = imgs.shape[0] * imgs.shape[1]  #len(self.lines)
       self.predict = predict
       self.shape = shape
       self.box_size = box_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

	    maxPeaks = 1024
        (n,m,h,w) = self.imgs.shape
        ind1 = index / m
        ind2 = index % m
        img = self.imgs[ind1,ind2,:,:]
        img = torch.from_numpy( img )
        img = img.view(-1, h, w )

        if self.predict:
            return img
        else:
            r = np.reshape( self.labels[ind1][1][ self.labels[ind1][0]==ind2 ], (-1,1) )
            c = np.reshape( self.labels[ind1][2][ self.labels[ind1][0]==ind2 ], (-1,1) )
            label = torch.zeros(5*maxPeaks)
            try:
                tmp = np.concatenate( (np.zeros( (r.shape[0],1) ),
                                            1.0*c/w, 1.0*r/h,
                                            1.0*self.box_size/w*np.ones( (r.shape[0],1) ),
                                            1.0*self.box_size/h*np.ones( (r.shape[0],1) )),
                                            axis=1 )
                tmp = torch.from_numpy(tmp)
            except:
                tmp = torch.zeros(1,5)
            tmp = tmp.view(-1)
            if r.shape[0] > 0 and tmp.numel() > 5:
                label[0:(5*r.shape[0])] = tmp
            return (img, label)
