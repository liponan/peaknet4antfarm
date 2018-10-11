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

    def __init__(self, imgs, labels, shape=None, shuffle=True, transform=None,
                    target_transform=None, train=False, seen=0, batch_size=64,
                    box_size=7, num_workers=4):
       # with open(root, 'r') as file:
       #     self.lines = file.readlines()

       # if shuffle:
       #     random.shuffle(self.lines)

       self.imgs = imgs
       self.labels = labels
       self.nSamples  = imgs.shape[0] * imgs.shape[1]  #len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.box_size = box_size
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # imgpath = self.lines[index].rstrip()

        # if self.train and index % 64== 0:
        #     if self.seen < 4000*64:
        #        width = 13*32
        #        self.shape = (width, width)
        #     elif self.seen < 8000*64:
        #        width = (random.randint(0,3) + 13)*32
        #        self.shape = (width, width)
        #     elif self.seen < 12000*64:
        #        width = (random.randint(0,5) + 12)*32
        #        self.shape = (width, width)
        #     elif self.seen < 16000*64:
        #        width = (random.randint(0,7) + 11)*32
        #        self.shape = (width, width)
        #     else: # self.seen < 20000*64:
        #        width = (random.randint(0,9) + 10)*32
        #        self.shape = (width, width)


	maxPeaks = 1024

        if self.train:
            jitter = 0#0.2
            hue = 0#0.1
            saturation = 0#1.5
            exposure = 0#1.5
            #box_size = 7

            (n,m,h,w) = self.imgs.shape
            #print(index)
            ind1 = index / m
            ind2 = index % m

            img = self.imgs[ind1,ind2,:,:]
            timg = torch.zeros( img.shape )
            timg = torch.from_numpy( img )
            timg = timg.view(-1, h, w )
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
            # img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            if r.shape[0] > 0 and tmp.numel() > 5:
                #print(label.size())
                #print(tmp.size())
                label[0:(5*r.shape[0])] = tmp
        else:
            raise("not implemented")
            '''
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)

            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(maxPeaks*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > maxPeaks*5:
                label = tmp[0:maxPeaks*5]
            elif tsz > 0:
                label[0:tsz] = tmp
            '''

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (timg.float(), label.float())
