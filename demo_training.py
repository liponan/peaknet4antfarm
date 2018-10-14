import sys
import time
#import psana
import torch as t
import numpy as np
import h5py
#sys.path.append( "../pytorch-yolo-v3" )
#import darknetv3 as dn
from peaknet import Peaknet
import peaknet_train
#from peaknet_utils import output_transform
from darknet_utils import get_region_boxes, nms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from demo_predict import load_from_cxi, predict

#######################

set_algo = "ada"
set_lr = 0.0001
set_comment = "kaiming"
project = "cxic0415_0092"

t_init = time.time()
writer = SummaryWriter( "runs/" + project + '_' + set_algo + '_' + str(set_lr) + '_' + set_comment )

filename = "/reg/neh/home/liponan/data/cxic0415/r0092/cxic0415_0092.cxi.backup"

########################


print(filename)
f = h5py.File(filename, 'r')
nPeaks = f["entry_1/result_1/nPeaks"].value
dataset_hits = len(nPeaks)
print('hits: ' + str(dataset_hits))
dataset_peaks = np.sum(nPeaks)
print('peaks: ' + str(dataset_peaks))
   
imgs = f["entry_1/data_1/data"]
_, h, w = imgs.shape
f.close()
print('img:', h, w)

batch_size = 3
nEpoch = 10

###########

t3 = time.time()
pn = Peaknet()
pn.init_model()
#pn.loadDNWeights()
pn.model.train()
#print("training?", pn.model.training)
#pn.model.print_network()
pn.model.cuda()
t4 = time.time()
print("model init time", t4-t3)



###########

#algo = "ada"


pn.model.save_weights( "results/weights/" + project + "_init.weights" )
#print("before", next( pn.model.parameters())[-1,0,:,:])
model0 = pn.model
#model0_dict = dict( model0.named_parameters() )
#pn.model.eval()
#nms_boxes = predict( pn, imgs, conf_thresh=0.15, nms_thresh=0.1, printPeaks=True )



for ep in range(nEpoch): 
    print("============= EPOCH %d ==========" % (ep+1))
    for t in range(dataset_hits): #range(468,469):
        if t % batch_size == 0:
            batch_imgs = None
            batch_labels = []
            t0 = time.time()
        print("ep", ep+1, "img", t)
	t1 = time.time()
        f = h5py.File(filename, 'r')
        img = f["entry_1/data_1/data"][t,:,:]
        mask = f["entry_1/data_1/mask"][t,:,:]
        img = img * (1-mask)
        x_label = f['entry_1/result_1/peakXPosRaw'][t,:]
        y_label = f['entry_1/result_1/peakYPosRaw'][t,:]
        f.close()
        #print("img shape", img.shape)
        imgs = np.reshape( img, (8, 185, 4, 194*2) )
        imgs = np.transpose( imgs, (0, 2, 1, 3) )
        imgs = np.reshape( imgs, (1, 32, 185, 388) )
        cls = np.zeros( (nPeaks[t],) )
        s = np.zeros( (nPeaks[t],) )
        r = np.zeros( (nPeaks[t],) )
        c = np.zeros( (nPeaks[t],) )
        bh = 7*np.ones( (nPeaks[t],) )
        bw = 7*np.ones( (nPeaks[t],) )
        for u in range(nPeaks[t]):
            my_s = (int(y_label[u])/185)*4 + (int(x_label[u])/388)
	    my_r = y_label[u] % 185
	    my_c = x_label[u] % 388
	    s[u] = my_s
	    r[u] = my_r
	    c[u] = my_c
        labels = (cls, s, r, c, bh, bw)
        #print(labels)
        if t % batch_size == 0 or t ==468:
            batch_imgs = imgs
        else:
            batch_imgs = np.concatenate( (batch_imgs, imgs), axis=0 )
        batch_labels.append( labels )
        t2 = time.time()
        #print("data proceessing time", t2-t1)
        if t % batch_size == (batch_size-1) or t == (dataset_hits-1) or t == 468:
	    '''
	    t3 = time.time()
	    pn = Peaknet()
	    pn.loadDNWeights()
	    pn.model.train()
	    #print("training?", pn.model.training)
	    #pn.model.print_network()
	    pn.model.cuda()
	    t4 = time.time()
	    print("model init time", t4-t3)
	    '''
	    #print("batch_imgs shape", batch_imgs.shape)
	    #print("batch_labels shape", len(batch_labels), len(batch_labels[0]), batch_labels[0][0].shape)
            optimizer = pn.optimizer(adagrad=set_algo=="ada", lr=set_lr )
	    pn.train( batch_imgs, batch_labels, batch_size=32*3, use_cuda=True, writer=writer )
	    pn.optimize( optimizer )
	    
	    t5 = time.time()
	    print("time per event", 1.0*(t5-t0)/batch_size)

            #pn.model.eval()
            #nms_boxes = predict( pn, imgs, conf_thresh=0.15, nms_thresh=0.1, printPeaks=True )
            #model = pn.model.cpu()
            #print("after", next( pn.model.parameters())[-1,0,:,:] )
            
            #model_dict2 = dict( model2.named_parameters() )
    pn.model.save_weights( "results/weights/" + project + "_ep"+str(ep+1)+".weights" )
            #model.load_weights( "results/cxic0415_0091_ep"+str(ep)+".weights" )
            #model_dict = dict( model.named_parameters() )
            #for key, value in model_dict.items():
            #    #model_dict[key].grad.data = grad[key].data
            #     print(key)
            #     print("original",model_dict[key].sum().data)
            #     #print()
            #     print("updated",model0_dict[key].sum().data)
            #     #print()


            #print("before", next( pn.model.parameters() ).grad[0,:,:] )
            #pn.updateGrad( pn.getGrad() )
            #print("after", next( pn.model.parameters() ).grad[0,:,:] )

t_end = time.time()            
print("total time elapsed", t_end-t_init)
writer.close()

    
    

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
