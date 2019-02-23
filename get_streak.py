import psana
import numpy as np
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