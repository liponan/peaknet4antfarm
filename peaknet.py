import os
import os.path as osp
import time
# from torch.autograd import Variable
import torch as t
import sys
sys.path.append(os.path.abspath('../pytorch-yolo2'))
from darknet import Darknet
# from train import train_peaknet
# from preprocess import prep_image, inp_to_image
# from util import loss, loadLabels, IOU

# workPath = "/reg/neh/home/liponan/ai/peaknet4antfarm/"
workPath = "../pytorch-yolo2/"

class Peaknet():

    def __init__(self):
        self.model = None


    def loadDNWeights( self, cfgPath, weightsPath ):
        self.model = Darknet(workPath + 'cfg/newpeaksv5.cfg')
        self.model.load_weights(workPath + "weights/newpeaksv5.backup")

    '''
    "imgs: variable name"
    "labels: variable name"
    '''

    def train( self, imgs, labels, box_size = 7 ):
        print("training...")

    def model( self ):
        return self.model

    '''
    "imgs: string"
    "labels: string"
    '''
    def train_from_shm( self, imgs, labels, box_size = 7 ):
        model = None
        return model


    def test( self, imgs, box_size = 7 ):
        results = None
        return results


    def validate( self, imgs, golden_labels, box_size = 7 ):
        results = None
        return results

    def updateModel():
        return None
