from __future__ import print_function
import sys

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import peaknet_dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet


def validate_batch( model, imgs, labels, batch_size=32, box_size=7, use_cuda=True, writer=None ):
    debug = True

    val_loader = torch.utils.data.DataLoader(
        peaknet_dataset.listDataset(imgs, labels,
            shape=(imgs.shape[2], imgs.shape[3]),
            predict=False,
            box_size=box_size,
            ),
        batch_size=batch_size, shuffle=False)
    model.eval()
    region_loss = model.loss
    region_loss.seen = model.seen
    t1 = time.time()
    avg_time = torch.zeros(9)

    for batch_idx, (data, target) in enumerate(val_loader):
        if use_cuda:
            data = data.cuda()
            target= target.cuda()
        data, target = Variable(data), Variable(target)
        output, _= model( data.float() )
        if debug:
            print("output", output.size())
            print("label length", len(target))
            print("label[0] length", len(target[0]))
        loss = region_loss(output, target)
        if writer != None:
            writer.add_scalar('loss_val', loss, model.seen)
