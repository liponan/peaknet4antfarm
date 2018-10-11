import sys
import time
import torch as t
import numpy as np
import h5py
from peaknet import Peaknet
import peaknet_train
from darknet_utils import get_region_boxes, nms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


def predict( net, filename, idx, conf_thresh=0.2, nms_thresh=0.45, printPeaks=False):

    f = h5py.File(filename, 'r')
    nPeaks = f["entry_1/result_1/nPeaks"].value
    dataset_hits = len(nPeaks)
    print('hits: ' + str(dataset_hits))
    dataset_peaks = np.sum(nPeaks)
    print('peaks: ' + str(dataset_peaks))
    img = f["entry_1/data_1/data"][idx,:,:]
    x_label = f['entry_1/result_1/peakXPosRaw'][idx,:]
    y_label = f['entry_1/result_1/peakYPosRaw'][idx,:]
    f.close()

    imgs = np.reshape( img, (8, 185, 4, 194*2) )
    imgs = np.transpose( imgs, (0, 2, 1, 3) )
    imgs = np.reshape( imgs, (1, 32, 185, 388) )
    n, m, h, w = imgs.shape

    timgs = t.zeros( (32, 1, 192, 392) )
    timgs[:,:,4:189,2:390] = t.from_numpy( imgs/15000.0 )

    s = np.zeros( (nPeaks[idx],) )
    r = np.zeros( (nPeaks[idx],) )
    c = np.zeros( (nPeaks[idx],) )
    for u in range(nPeaks[idx]):
        my_s = (int(y_label[u])/185)*4 + (int(x_label[u])/388)
        my_r = y_label[u] % 185
        my_c = x_label[u] % 388
        s[u] = my_s
        r[u] = my_r
        c[u] = my_c
        labels = (s, r, c)

    timgs = timgs.cuda()
    timgs = t.autograd.Variable( timgs )

    t3 = time.time()

    output, _ = net.model(timgs)
    output = output.data

    t4 = time.time()

    boxes = get_region_boxes(output, conf_thresh, net.model.num_classes,
                                net.model.anchors, net.model.num_anchors)

    t5 = time.time()

    nms_boxes = []
    for box in boxes:
        n0 = len(box)
        box = nms(box, nms_thresh)
        n1 = len(box)
        print(n0, "=>", n1)
        nms_boxes.append( box )

    t6 = time.time()

    if printPeaks:
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

def main():

    net = Peaknet()
    net.loadDNWeights()
    net.model.eval()
    net.model.cuda()

    filename = "/reg/neh/home/liponan/data/cxic0415/r0091/cxic0415_0091.cxi.backup"
    idx = 7

    predict( net, filename, idx, conf_thresh=0.1, nms_thresh=0.45, printPeaks=True )


if __name__ == "__main__":
    main()




'''
eventIdx = 2965
cfd_thresh = 0.4
nms_thresh = 0.45


t0 = time.time()

evt = this_run.event(times[eventIdx])
calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                                  edges=True, central=True,
                                  unbond=True, unbondnbrs=True)

(c, h, w) = calib.shape
h = 192
w = 392

np.save( exp_name + '_' + str(run) + ".npy", calib )

#imgs = t.zeros( (32, 1, 392, 192) )
#imgs[:,:,2:390,4:189] = t.from_numpy( calib/25000.0 ).view(32,1,185,388).transpose(2,3)
imgs = t.zeros( (32, 1, 192, 392) )
imgs[:,:,4:189,2:390] = t.from_numpy( calib/10000.0 ).view(32,1,185,388)
global_max = np.max(calib)
print("global max", global_max)
print("calib shape", calib.shape)
print("imgs size", imgs.size() )

t1 = time.time()

imgs = imgs.cuda()
imgs = t.autograd.Variable( imgs )

t2 = time.time()

pn = Peaknet()
#pn.model = dnmodel
pn.loadDNWeights()
print(pn.model)
print("training?", pn.model.training)
pn.model.eval()
pn.model.cuda()
#pn.model.eval()
print(pn.model)

print("training?", pn.model.training)
pn.model.print_network()
print(pn.model.anchors)

t3 = time.time()

output = pn.model(imgs)
print(output)
#output = pn.model(imgs, True)
output = output.data

print(output)
print("output size", output.size() )

t4 = time.time()

boxes = get_region_boxes(output, cfd_thresh, pn.model.num_classes, pn.model.anchors, pn.model.num_anchors)
#boxes = get_region_boxes(output, cfd_thresh, 1, [1,1], 1)


t5 = time.time()

nms_boxes = []

for box in boxes:
    n0 = len(box)
    box = nms(box, nms_thresh)
    n1 = len(box)
    print(n0, "=>", n1)
    nms_boxes.append( box )
t6 = time.time()

printPeaks = False

if printPeaks:
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
print('-----------------------------------')
print(' image to tensor : %f' % (t1 - t0))
print('  tensor to cuda : %f' % (t2 - t1))
print('  set-up network : %f' % (t3 - t2))
print('         predict : %f' % (t4 - t3))
print('get_region_boxes : %f' % (t5 - t4))
print('             nms : %f' % (t6 - t5))
print('           total : %f' % (t6 - t0))
print('-----------------------------------')


predictions = output_transform( output, [192, 392], [[1, 1]], 1, True )
print( "predictions size", predictions.size() )


outputFig = True
mode = "transform"
transform = True
nms = True

for i, box in enumerate(nms_boxes):
    print(predictions[i,0,:])
    if len(box) == 0:
        continue
    fig, ax = plt.subplots(1)
    img = imgs.data.cpu().numpy()[i,:,:] # calib[i,:,:]/15000
    print("img shape", img.shape)
    im0 = plt.imshow(img.reshape(h,w), vmin=0, vmax=1, cmap=cm.gray)
    if transform:
        for j in range(predictions.size(1)):
            peak = predictions[i,j,:]
            #print(peak)
            if peak[4] < cfd_thresh:
                continue
            #print(peak)
            x = 1 * ( peak[0]-0.5*peak[2] )
            y = 1 * ( peak[1]-0.5*peak[3] )
            ww = 1 * peak[2]
            hh = 1 * peak[3]
            rect = pat.Rectangle( (x, y), ww, hh, color="c", fill=False, linewidth=1 )
            ax.add_patch(rect)
    if nms:
        for peak in nms_boxes[i]:
            if peak[4] < cfd_thresh:
                continue
            x = w * ( peak[0]-0.5*peak[2] )
            y = h * ( peak[1]-0.5*peak[3] )
            ww = w * peak[2]+2
            hh = h * peak[3]+2
            rect = pat.Rectangle( (x, y), ww, hh, color="m", fill=False, linewidth=1 )
            ax.add_patch(rect)

    fig.set_size_inches(5, 5)
    if outputFig:
        plt.savefig( "results/{}_{}.png".format(str(eventIdx).zfill(6), str(i).zfill(2)), bbox_inces='tight', dpi=300)


'''
