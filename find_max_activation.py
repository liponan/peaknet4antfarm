import sys
import os
import time
import psana
import torch as t
import numpy as np
from peaknet import Peaknet
#from peaknet_utils import output_transform
from darknet_utils import get_region_boxes, nms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm
import cv2
from heapq import *
import json

# global variables
layers = [0,2,4,6,7,8,9]
fovs = [3+2, None, 8+3, None, 18+5, None, 20+7, 22+7, 24+7, 26+7]

nfactor = [1, 2, 2, 4, 4, 8, 8, 8, 8, 8]
nlayers = [32, 32, 64, 64, 128, 128, 256, 128, 64, 6]

h = 192
w = 392


def combine_data( data, new_data, nHeapq ):
    for l in layers:
        for v in range(nlayers[l]):
            for k in range( len( new_data[l][v] )-1, -1, -1 ):
                if new_data[l][v][k][0] > data[l][v][0][0]:
                    heappush( data[l][v], new_data[l][v][k] )
                    if len(data[l][v]) > nHeapq:
                        heappop( data[l][v] )
                else:
                    break


def find_max_activatations(exp_name, run_list, nHeapq=9, save_json=False):

    data = {}

    pn = Peaknet()
    pn.loadDNWeights()
    pn.model.cuda()
    pn.model.eval()
    pn.model.print_network()

    for u in layers:
        data[u] = {}
        for v in range(nlayers[u]):
            data[u][v] = [(float("-inf"),None)]

    for run in run_list:

        print("****************************** run " + str(run) + " ********************************")

        ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
        det = psana.Detector('DscCsPad')
        this_run = ds.runs().next()
        times = this_run.times()
        numEvents = len(times)
        print("run", run, "number of events available", numEvents)
        env = ds.env()

        my_data = {}
  
        for u in layers:
            my_data[u] = {}
            for v in range(nlayers[u]):
                my_data[u][v] = [(float("-inf"),None)]

        for eventIdx in range(numEvents):
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

            for l in layers:
                layer_outputs = outputs[l].data.cpu().numpy()

                for u in range(layer_outputs.shape[0]):
                    for v in range(layer_outputs.shape[1]):
                        max_val = np.max( layer_outputs[u,v,:,:] )
                        ind = np.unravel_index(np.argmax(layer_outputs[u,v,:,:], \
                                                axis=None), layer_outputs[u,v,:,:].shape)
                        if max_val > my_data[l][v][0][0]:
                            heappush( my_data[l][v], (float(max_val), run, eventIdx, u, ind) )
                            print("layer", l, "filter", v, "@ run", run, "pos", ind, "val", max_val)
                            if len(my_data[l][v]) > nHeapq:
                                heappop( my_data[l][v] )

        if save_json:
            with open(exp_name + "_run" + str(run).zfill(4) + '_data.json', 'w') as outfile:
                json.dump(my_data, outfile) 

        combine_data( data, my_data, nHeapq )

    
    return data

def visualize(exp_name, data, output_path = "results/maxact/" ):

    det = psana.Detector('DscCsPad')

    max_x = -1
    max_y = -1
    max_v = -1
    y_offset = -4
    x_offset = -2
    last_run = -1

    for l in layers:
        box_rad = (fovs[l]-1) / 2
        for u in range(nlayers[l]):
            for v in range(len(data[l][u])):
                run = data[l][u][v][1]
                eventIdx = data[l][u][v][2]
                s = data[l][u][v][3]
                ind = data[l][u][v][4]
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

                filename = "layer" + str(l) + '_' \
                            + "filter" + str(u).zfill(3) + '_' \
                            + str(v).zfill(3) + ".png"
                if os.path.isdir( output_path ):
                    pass
                else:
                    os.makedirs( output_path )
                cv2.imwrite( os.path.join(output_path, filename), img.astype( np.uint8 ) )
            #print( img.astype( np.uint8 ) )

    print("max v", max_v)


def main():
    # check that the file is being properly used
    if (len(sys.argv) < 3):
        print("Please specify an experiment name and run number list as args.")
        return
    # input variables
    exp_name = sys.argv[1]
    runs = sys.argv[2:]
    run_list = []
    for run in runs:
        run_list.append( int(run) )
    # find max activations
    data = find_max_activatations(exp_name, run_list, nHeapq=25, save_json=True)
    # visualize
    visualize(exp_name, data, output_path = "results/maxact/" )


if __name__ == "__main__":
     main()
