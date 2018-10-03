import sys
import time
import psana
import torch as t
import numpy as np
#sys.path.append( "../pytorch-yolo-v3" )
#import darknetv3 as dn
from peaknet import Peaknet
from peaknet_utils import output_transform
from darknet_utils import get_region_boxes, nms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm
import cv2
from heapq import *
import json

nHeapq = 9#25
save_json = False

#dnmodel = dn.Darknet( "../pytorch-yolo-v3/cfg/newpeaksv10-asic.cfg" )
#dnmodel.load_weights( "../pytorch-yolo-v3/weights/newpeaksv10_40000.weights" )


data = {}

##########

exp_name = "cxic0415"

#run_list = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
run_list = [91]

###########

#eventIdx = 2965
cfd_thresh = 0.8
nms_thresh = 0.45

h = 192
w = 392


pn = Peaknet()
#pn.model = dnmodel
pn.loadDNWeights()
print(pn.model)
print("training?", pn.model.training)
#pn.model.eval()
pn.model.cuda()
pn.model.eval()
print(pn.model)

print("training?", pn.model.training)
pn.model.print_network()


#layers = [0,2,4,6,7,8,9]
#factors = [1,2,4,8,8,8,8]
layers = [0,2,4,6,7,8,9]
fovs = [3+2, None, 8+3, None, 18+5, None, 20+7, 22+7, 24+7, 26+7]

nfactor = [1, 2, 2, 4, 4, 8, 8, 8, 8, 8]
nlayers = [32, 32, 64, 64, 128, 128, 256, 128, 64, 6]

for u in layers:
    for v in range(nlayers[u]):
        data[(u,v)] = [(-1,None)]

for run in run_list:

    print("****************************** run " + str(run) + " ********************************")

    ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
    det = psana.Detector('DscCsPad')
    this_run = ds.runs().next()
    times = this_run.times()
    numEvents = len(times)
    print("run", run, "number of events available", numEvents)
    env = ds.env()


    for eventIdx in range(numEvents): # range(2560,2660):
        print("==================== run " + str(run) + " event " + str(eventIdx) + \
                " ==============================") 
        evt = this_run.event(times[eventIdx])
        calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                                  edges=True, central=True,
                                  unbond=True, unbondnbrs=True)

        imgs = t.zeros( (32, 1, 192, 392) )
        imgs[:,:,4:189,2:390] = t.from_numpy( calib/25500.0 ).view(32,1,185,388)
    
        imgs = imgs.cuda()
        imgs = t.autograd.Variable( imgs )

        output, outputs = pn.model(imgs)
    #print(outputs[0].size())
#output = pn.model(imgs, True)

        for l in layers:
            layer_outputs = outputs[l].data.cpu().numpy()
            #print("layer_outputs", layer_outputs.shape )

            for u in range(layer_outputs.shape[0]):
                for v in range(layer_outputs.shape[1]):
                    max_val = np.max( layer_outputs[u,v,:,:] )
                    ind = np.unravel_index(np.argmax(layer_outputs[u,v,:,:], \
                                            axis=None), layer_outputs[u,v,:,:].shape)
                    if max_val > data[(l,v)][0][0]:
                        heappush( data[(l,v)], (float(max_val), run, eventIdx, u, ind) )
                        print("layer", l, "filter", v, "@ run", run, "event", ind, "val", max_val)
                        if len(data[(l,v)]) > nHeapq:
                            heappop( data[(l,v)] )

print(data)


if save_json:
    with open(exp_name + '_data.json', 'w') as outfile:
        json.dump(data, outfile)

max_x = -1
max_y = -1
max_v = -1
y_offset = -4
x_offset = -2
last_run = -1
for l in layers:
    box_rad = (fovs[l]-1) / 2
    for u in range(nlayers[l]):
        for v in range(len(data[(l,u)])):
            run = data[(l,u)][v][1]
            eventIdx = data[(l,u)][v][2]
            s = data[(l,u)][v][3]
            ind = data[(l,u)][v][4]
            x = nfactor[l] * ind[1] #+ x_offset
            y = nfactor[l] * ind[0] #+ y_offset
            print(l, u, v, x, y)
            if run != last_run:
                ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
                this_run = ds.runs().next()
                env = ds.env()
            last_run = run
            times = this_run.times()
            evt = this_run.event(times[eventIdx])
            calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                                      edges=True, central=True,
                                      unbond=True, unbondnbrs=True)
            calib = np.pad( calib, ( (0,0), (4,3), (2,2) ), "constant", constant_values=0 )
            img = calib[s,max(0,(y-box_rad)):min(h,(y+box_rad+1)),
                          max(0,(x-box_rad)):min(w,(x+box_rad+1))]
       
            if np.max( img ) > max_v:
                max_v = np.max( img )
            img /= (15000.0 / 255.0)
        
            filename = "results/maxact/" \
                        + "layer" + str(l) + '_' \
                        + "filter" + str(u).zfill(3) + '_' \
                        + str(v).zfill(3) + ".png"
            cv2.imwrite( filename, img.astype( np.uint8 ) )
        #print( img.astype( np.uint8 ) )

print("max v", max_v)

'''
    boxes = get_region_boxes(output, cfd_thresh, pn.model.num_classes, pn.model.anchors,     pn.model.num_anchors)
#boxes = get_region_boxes(output, cfd_thresh, 1, [1,1], 1)


    nms_boxes = []

    for box in boxes:
        box = nms(box, nms_thresh)
        nms_boxes.append( box )

    count = 0
    box_rad = 11;

    for i, box in enumerate(nms_boxes):
        if len(box) > 0:
            print("asic", i)
        else:
            continue
        for b in box:
          x = int(w*b[0])
          y = int(h*b[1])
          ww = int(w*b[2])
          hh = int(h*b[3])
          score = int(b[4]*100)
          print("(%d, %d), %d x %d, score: %d" % ( x, y, ww, hh, score ) )
          try:
              img = calib[i,(y-box_rad):(y+box_rad+1),(x-box_rad):(x+box_rad+1)]
          except:
              continue
          img /= 25500 / 255
          filename = "results/peaks/" + str(score) + '_' +  exp_name + '_' + str(run) \
                                + '_' + str(eventIdx).zfill(6) \
                                + '_' + str(count).zfill(6) + ".png"
          count = count + 1
          cv2.imwrite( filename, img )


'''

