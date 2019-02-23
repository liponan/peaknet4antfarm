import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm

def loadLabels(path, num):
    data = []
    for u in range(num):
        labels = []
        txt = open( path + str(u).zfill(6) + ".txt", 'r').readlines()
        for v, line in enumerate(txt):
            vals = line.split(" ")
            x = round( w * float(vals[1]))
            y = round( h * float(vals[2]))
            labels.append( (x,y) )
        data.append( labels )
    return data


def visualize( imgs, labels=None, nms_boxes=None, plot_label=False, plot_box=False, box_size=7, indexes=[], vmax=10000):
    for i in indexes:
        box = nms_boxes[i]
        if len(box) == 0:
            continue
        fig, ax = plt.subplots(1)
        img = imgs[0,i,:,:]
        #h, w = img.shape
        h, w = 192, 392
        # plot image
        im0 = plt.imshow(img, vmin=0, vmax=vmax, cmap=cm.gray)
        plt.title( "ASIC " + str(i) ) 
        fig.set_size_inches(12, 6)
        # plot labels
        if plot_label:
            my_r = labels[2][ labels[1] == i ]
            my_c = labels[3][ labels[1] == i ]
            my_h = labels[4][ labels[1] == i ]
            my_w = labels[5][ labels[1] == i ]
            my_l = labels[0][ labels[1] == i ]
            for j in range(len(my_r)):
                x = my_c[j] - my_w[j]/2.0
                y = my_r[j] - my_h[j]/2.0
                ww = box_size
                hh = box_size
                if my_l == 0:
                    color == "c"
                else:
                    color == "g"
                rect = pat.Rectangle( (x, y), ww, hh, color=color, fill=False, linewidth=1 )
                ax.add_patch(rect)
        # plot predictions
        if plot_box:
            for peak in nms_boxes[i]:
                x = w * ( peak[0]-0.5*peak[2] ) - 2
                y = h * ( peak[1]-0.5*peak[3] ) - 4
                ww = w * peak[2]
                hh = h * peak[3]
                if peak[5] > peak[6]:
                    color = "m"
                else:
                    color = "y"
                rect = pat.Rectangle( (x, y), ww, hh, color=color, fill=False, linewidth=1 )
                ax.add_patch(rect)

                
def load_cxi_labels( filename, box_size=7, total_size=-1, shuffle=False ):
    # H5 file IO
    with h5py.File(filename, 'r') as f:
        nPeaks = f["entry_1/result_1/nPeaks"].value
        nStreaks = f["entry_1/result_1/nStreaks"].value
        dataset_hits = len(nPeaks)
        print('hits: ' + str(dataset_hits))
        if total_size == -1:
            n_total = dataset_hits
        else:
            n_total = total_size
        eventIdx = f["LCLS/eventNumber"][:n_total]
        peak_x_label = f['entry_1/result_1/peakXPosRaw'][:n_total,:]
        peak_y_label = f['entry_1/result_1/peakYPosRaw'][:n_total,:]
        streak_p_label = f["entry_1/result_1/streakPanel"][:n_total,:]
        streak_x_label = f["entry_1/result_1/streakXPos"][:n_total,:]
        streak_y_label = f["entry_1/result_1/streakYPos"][:n_total,:]
        streak_w_label = f["entry_1/result_1/streakWidth"][:n_total,:]
        streak_h_label = f["entry_1/result_1/streakHeight"][:n_total,:]
    # image reshaping

    # label formatting
    labels = []
    for i in range(n_total):
        nObj = nPeaks[i] + nStreaks[i]
        offset = nPeaks[i]
        cls = np.zeros( (nObj,) )
        s = np.zeros( (nObj,) )
        r = np.zeros( (nObj,) )
        c = np.zeros( (nObj,) )
        ww = np.zeros( (nObj,) )
        hh = np.zeros( (nObj,) )
        for j in range(nPeaks[i]):
            my_s = (int(peak_y_label[i,j])/185)*4 + (int(peak_x_label[i,j])/388)
            my_r = peak_y_label[i,j] % 185
            my_c = peak_x_label[i,j] % 388
            s[j] = my_s
            r[j] = my_r
            c[j] = my_c
            hh[j] = box_size
            ww[j] = box_size
        for j in range(nStreaks[i]):
            my_s = streak_p_label[i,j]
            my_r = streak_x_label[i,j]
            my_c = streak_y_label[i,j]
            my_h = streak_w_label[i,j]
            my_w = streak_h_label[i,j]
            s[offset+j] = my_s
            r[offset+j] = my_r
            c[offset+j] = my_c
            hh[offset+j] = np.minimum( np.minimum( 2*my_r, my_h ), 2*(185-my_r) )
            ww[offset+j] = np.minimum( np.minimum( 2*my_c, my_w ), 2*(388-my_c) )
        my_label = (cls, s, r, c, hh, ww)
        labels.append( my_label )
    # randomize    
    if shuffle:
        rand_idxs = np.random.permutation( n_total )
        labels = [labels[i] for i in rand_idxs]
        eventIdx = [eventIdx[i] for i in rand_idxs]
    return labels, eventIdx

def psana_img_loader(eventIdxs, startIdx, n, det, this_run, times):
    imgs = np.zeros( (n, 32, 185, 388) )
    for i, event_idx in enumerate(eventIdxs[startIdx:(startIdx+n)]):
        evt = this_run.event(times[event_idx])
        calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                              edges=True, central=True,
                              unbond=True, unbondnbrs=True)
        imgs[i,:,:,:] = calib
    return imgs
    

def build_dataset( filename, dev_size, box_size=7, total_size=-1 ):
    # H5 file IO
    with h5py.File(filename, 'r') as f:
        nPeaks = f["entry_1/result_1/nPeaks"].value
        dataset_hits = len(nPeaks)
        print('hits: ' + str(dataset_hits))
        if total_size == -1:
            n_total = dataset_hits
        else:
            n_total = total_size
        imgs = f["entry_1/data_1/data"][:n_total,:,:]
        masks = f["entry_1/data_1/mask"][:n_total,:,:]
        imgs= imgs * (1-masks)
        x_label = f['entry_1/result_1/peakXPosRaw'][:n_total,:]
        y_label = f['entry_1/result_1/peakYPosRaw'][:n_total,:]
    # image reshaping
    imgs = np.reshape( imgs, (-1, 8, 185, 4, 194*2) )
    imgs = np.transpose( imgs, (0, 1, 3, 2, 4) )
    imgs = np.reshape( imgs, (-1, 32, 185, 388) )
    n, m, h, w = imgs.shape
    # label formatting
    labels = []
    for i in range(n_total):
        cls = np.zeros( (nPeaks[i],) )
        s = np.zeros( (nPeaks[i],) )
        r = np.zeros( (nPeaks[i],) )
        c = np.zeros( (nPeaks[i],) )
        ww = np.zeros( (nPeaks[i],) )
        hh = np.zeros( (nPeaks[i],) )
        for j in range(nPeaks[i]):
            my_s = (int(y_label[i,j])/185)*4 + (int(x_label[i,j])/388)
            my_r = y_label[i,j] % 185
            my_c = x_label[i,j] % 388
            s[j] = my_s
            r[j] = my_r
            c[j] = my_c
            hh[j] = box_size
            ww[j] = box_size
        my_label = (cls, s, r, c, hh, ww)
        labels.append( my_label )
    # randomize    
    rand_idxs = np.random.permutation( n_total )
    imgs = imgs[ rand_idxs, :, :, : ]
    labels = [labels[i] for i in rand_idxs]
    # build dev set
    if dev_size > 0:
        if dev_size < 1:
            dev_size = round( dev_size * dataset_hits )
    dev_imgs = imgs[:dev_size,:,:,:]
    dev_labels = labels[:dev_size]
    train_imgs = imgs[dev_size:,:,:,:]
    train_labels = labels[dev_size:]

    return train_imgs, train_labels, dev_imgs, dev_labels
 
 
def load_from_cxi( filename, idx, box_size=7 ):
    f = h5py.File(filename, 'r')
    nPeaks = f["entry_1/result_1/nPeaks"].value
    dataset_hits = len(nPeaks)
    #print('hits: ' + str(dataset_hits))
    dataset_peaks = np.sum(nPeaks)
    #print('peaks: ' + str(dataset_peaks))
    img = f["entry_1/data_1/data"][idx,:,:]
    mask = f["entry_1/data_1/mask"][idx,:,:]
    img = img * (1-mask).astype(float)
    x_label = f['entry_1/result_1/peakXPosRaw'][idx,:]
    y_label = f['entry_1/result_1/peakYPosRaw'][idx,:]
    f.close()

    imgs = np.reshape( img, (8, 185, 4, 194*2) )
    imgs = np.transpose( imgs, (0, 2, 1, 3) )
    imgs = np.reshape( imgs, (1, 32, 185, 388) )
    n, m, h, w = imgs.shape

    cls = np.zeros( (nPeaks[idx],) )
    s = np.zeros( (nPeaks[idx],) )
    r = np.zeros( (nPeaks[idx],) )
    c = np.zeros( (nPeaks[idx],) )
    ww = np.zeros( (nPeaks[idx],) )
    hh = np.zeros( (nPeaks[idx],) )
    for u in range(nPeaks[idx]):
        my_s = (int(y_label[u])/185)*4 + (int(x_label[u])/388)
        my_r = y_label[u] % 185
        my_c = x_label[u] % 388
        s[u] = my_s
        r[u] = my_r
        c[u] = my_c
        hh[u] = box_size
        ww[u] = box_size
    labels = (cls, s, r, c, hh, ww)

    return imgs, labels