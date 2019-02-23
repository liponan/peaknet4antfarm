import sys
import os
import time
#import torch as t
import numpy as np
import h5py
#from peaknet import Peaknet
#import peaknet_train
import pickle
import psana
#from darknet_utils import get_region_boxes, nms
#from torch.autograd import Variable
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm

# read labels from pickle

with open("peaknetLabels.pkl") as f:
    labels = pickle.load(f)

    
exp_name = "cxitut13"
run = 10
event_idx = 19

ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
det = psana.Detector('DscCsPad')
this_run = ds.runs().next()
times = this_run.times()
num_events = len(times)
print("run", run, "number of events available", num_events)
env = ds.env()

evt = this_run.event(times[event_idx])
calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                          edges=True, central=True,
                          unbond=True, unbondnbrs=True)

colors = "cr"
l = labels[event_idx-18]


for i in range(calib.shape[0]):
    img = calib[i,:,:] / 15000.0
    print("img shape", img.shape)
    skip = True
    fig, ax = plt.subplots()
    im0 = plt.imshow( img, cmap=cm.gray, vmin=0, vmax=0.02 )
    for j in range(len(labels[event_idx-18][0])):
        if l[1][j] != i:
            continue
        skip = False
        ww = l[4][j]
        hh = l[5][j]
        x = l[2][j] #- ww/2.0
        y = l[3][j] -1#- hh/2.0
        c = int(l[0][j])
        rect = pat.Rectangle( (x, y), ww, hh, color=colors[c], fill=False, linewidth=1 )
        ax.add_patch(rect)
    fig.set_size_inches(10, 5)
    if not skip:
        plt.savefig( "results/streak/{}_{}_{}.png".format(exp_name, str(event_idx).zfill(6), str(i).zfill(2)), bbox_inces='tight', dpi=300)
