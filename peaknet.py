import os
import os.path as osp
import time
from torch.autograd import Variable
import torch as t
import sys
sys.path.append(os.path.abspath('../pytorch-yolo-v3'))
from darknet import Darknet
from preprocess import prep_image, inp_to_image
from util import loss, loadLabels, IOU


class Peaknet():

    '''
    "imgs: variable name"
    "labels: variable name"
    '''
    def train( self, model, imgs, labels, box_size = 7, tmp_path="tmps" ):
        dn = Darknet('cfg/newpeaksv9-yolo.cfg')
        dn.load_weights("weights/newpeaksv9_40000.weights")
        return dn

    '''
    "imgs: string"
    "labels: string"
    '''
    def train_from_shm( self, model, imgs, labels, box_size = 7, tmp_path="tmps" ):
        model = None
        return model


    def test( self, imgs, box_size = 7):
        results = None
        return results


    def validate( self, imgs, golden_labels, box_size = 7, tmp_path="tmps" ):
        results = None
        return results

    def updateModel():
        return None
