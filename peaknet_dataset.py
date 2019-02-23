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
from darknet_utils import read_truths_args, read_truths
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

        new_h = 192
        new_w = 392
        timg = torch.zeros( ( 1, new_h, new_w) )

        img = self.imgs[ind1,ind2,:,:] / 15000
        timg[:,4:189,2:390] = torch.from_numpy( img )
        timg = timg.view(-1, new_h, new_w )

        if self.predict:
            return timg
        else:
            r = np.reshape( self.labels[ind1][2][ self.labels[ind1][1]==ind2 ], (-1,1) )
            c = np.reshape( self.labels[ind1][3][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bh = np.reshape( self.labels[ind1][4][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bw = np.reshape( self.labels[ind1][5][ self.labels[ind1][1]==ind2 ], (-1,1) )
            cls = np.reshape( self.labels[ind1][0][ self.labels[ind1][1]==ind2 ], (-1,1) )
#             cls = np.zeros( r.shape )
            label = torch.zeros(5*maxPeaks)
            bh[ bh == 0 ] = self.box_size
            bw[ bw == 0 ] = self.box_size
            #print(r, c, bh, bw)
            try:
                tmp = np.concatenate( (cls, np.maximum(np.minimum(1.0*(c+2)/392.0, 1.0), 0.0), 
                                            np.maximum(np.minimum(1.0*(r+4)/192.0, 1.0), 0.0),
                                           1.0*bw/392.0, 1.0*bh/192.0), axis=1 )
                tmp = torch.from_numpy(tmp)
            except:
                tmp = torch.zeros(1,5)
            #print(tmp)
            tmp = tmp.view(-1)
            if r.shape[0] > 0 and tmp.numel() > 5:
                label[0:(5*r.shape[0])] = tmp
            return (timg, label)
