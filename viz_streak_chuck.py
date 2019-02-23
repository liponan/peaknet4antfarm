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

from scipy import signal as sg
from skimage.measure import label, regionprops
from skimage import morphology

def getStreaks(det, times, run, j):
    """Finds peaks within an event, and returns the event information, peaks found, and hits found

    Arguments:
    det -- psana.Detector() of this experiment's detector
    times -- all the events for this run
    run -- ds.runs().next(), the run information
    j -- this event's number

    """
    evt = run.event(times[j])

    width = 300  # crop width
    sigma = 1
    smallObj = 50 # delete streaks if num pixels less than this
    calib = det.calib(evt)
    if calib is None:
        return [None, None, None]
    img = det.image(evt, calib)

    # Edge pixels
    edgePixels = np.zeros_like(calib)
    for i in range(edgePixels.shape[0]):
        edgePixels[i, 0, :] = 1
        edgePixels[i, -1, :] = 1
        edgePixels[i, :, 0] = 1
        edgePixels[i, :, -1] = 1
    imgEdges = det.image(evt, edgePixels)

    # Crop centre of image
    (ix, iy) = det.point_indexes(evt)
    halfWidth = int(width // 2)  # pixels
    imgCrop = img[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth]
    imgEdges = imgEdges[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth]
    myInd = np.where(imgEdges == 1)

    # Blur image
    imgBlur = sg.convolve(imgCrop, np.ones((2, 2)), mode='same')
    mean = imgBlur[imgBlur > 0].mean()
    std = imgBlur[imgBlur > 0].std()

    # Mask out pixels above 1 sigma
    mask = imgBlur > mean + sigma * std
    mask = mask.astype(int)
    signalOnEdge = mask * imgEdges
    mySigInd = np.where(signalOnEdge == 1)
    mask[myInd[0].ravel(), myInd[1].ravel()] = 1

    # Connected components
    myLabel = label(mask, neighbors=4, connectivity=1, background=0)
    # All pixels connected to edge pixels is masked out
    myMask = np.ones_like(mask)
    myParts = np.unique(myLabel[myInd])
    for i in myParts:
        myMask[np.where(myLabel == i)] = 0

    # Delete edges
    myMask[myInd] = 1
    myMask[mySigInd] = 0

    # Delete small objects
    myMask = morphology.remove_small_objects(np.invert(myMask.astype('bool')), smallObj)

    # Convert assembled to unassembled
    wholeMask = np.zeros_like(img)
    wholeMask[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth] = myMask

    calibMask = det.ndarray_from_image(evt, wholeMask)
    streaks = []
    for i in range(calib.shape[0]):
        for j in regionprops(calibMask[i].astype('int')):
            xmin, ymin, xmax, ymax = j.bbox
            width = xmax - xmin
            height = ymax - ymin
            streaks.append([i, ymin, xmin, height, width])

            if 0: # show mask and box
                fig,ax = plt.subplots(1)
                ax.imshow(calibMask[i],interpolation='none')
                # Create a Rectangle patch
                rect = pat.Rectangle((ymin,xmin),height,width,linewidth=2,edgecolor='y',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
                plt.show()

    return streaks

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

streaks = getStreaks(det, times, this_run, event_idx)

for i in range(len(streaks)):
    panel = streaks[i][0]
    ymin = streaks[i][1]
    xmin = streaks[i][2]
    height = streaks[i][3]
    width = streaks[i][4]
    fig,ax = plt.subplots(1)
    ax.imshow(calib[panel],interpolation='none', vmax=100, vmin=0)
    # Create a Rectangle patch
    rect = pat.Rectangle((ymin,xmin),height,width,linewidth=2,edgecolor='y',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.title('event {}: panel {}'.format(event_idx, panel))
    plt.show()
    #fig.set_size_inches(10, 5)
    #plt.savefig( "results/streak/{}_{}_{}.png".format(exp_name, str(event_idx).zfill(6), str(panel).zfill(2)), bbox_inces='tight', dpi=300)

