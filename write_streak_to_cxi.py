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


from get_streak import getStreaks

# exp_name = "cxitut13"
# run = 10
exp_name = "cxic0415"
run = 95

# psana 

ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
det = psana.Detector('DscCsPad')
this_run = ds.runs().next()
times = this_run.times()
num_events = len(times)
print("run", run, "number of events available", num_events)
env = ds.env()

# cxi

# h5_filename = "/reg/neh/home/liponan/data/cxitut13/r0010/cxitut13_0010_streak.cxi"
h5_filename = "/reg/neh/home/liponan/data/cxic0415/r0095/cxic0415_0095.cxi"
f = h5py.File(h5_filename, 'r+')

eventNumbers = f["/LCLS/eventNumber"]

nHits = len(eventNumbers)
print(nHits, "hit events")

try:
    nStreaks = f.create_dataset("entry_1/result_1/nStreaks", (nHits, ), dtype='int64')
    pDataset = f.create_dataset("entry_1/result_1/streakPanel", (nHits, 8), chunks=(1,8), dtype='int64')
    xDataset = f.create_dataset("entry_1/result_1/streakXPos", (nHits, 8), chunks=(1,8))
    yDataset = f.create_dataset("entry_1/result_1/streakYPos", (nHits, 8), chunks=(1,8))
    wDataset = f.create_dataset("entry_1/result_1/streakWidth", (nHits, 8), chunks=(1,8))
    hDataset = f.create_dataset("entry_1/result_1/streakHeight", (nHits, 8), chunks=(1,8))
except:
    print("dataset(s) already existed")
    nStreaks = f["entry_1/result_1/nStreaks"]
    pDataset = f["entry_1/result_1/streakPanel"]
    xDataset = f["entry_1/result_1/streakXPos"]
    yDataset = f["entry_1/result_1/streakYPos"]
    wDataset = f["entry_1/result_1/streakWidth"]
    hDataset = f["entry_1/result_1/streakHeight"]

t0 = time.time()
for j, event_idx in enumerate(eventNumbers):
    streaks = getStreaks(det, times, this_run, event_idx)
    nStreaks[j] = len(streaks)
    pDataset[j,:] = 0
    xDataset[j,:] = 0
    yDataset[j,:] = 0
    wDataset[j,:] = 0
    hDataset[j,:] = 0
    for i in range(len(streaks)):
        panel = streaks[i][0]
        ymin = streaks[i][1]
        xmin = streaks[i][2]
        height = streaks[i][3]
        width = streaks[i][4]
        x = 1.0*xmin + 0.5*width
        y = 1.0*ymin + 0.5*height
        print("event", event_idx, (panel, y, x, height, width))
        pDataset[j,i] = panel
        xDataset[j,i] = x
        yDataset[j,i] = y
        wDataset[j,i] = width
        hDataset[j,i] = height
    

f.close()
print("all done")
t2 = time.time()
print("time elapsed", t2-t0)
