import os
import os.path as osp
import time
# from torch.autograd import Variable
import torch as t
import sys
sys.path.append(os.path.abspath('../pytorch-yolo2'))
from darknet import Darknet
import peaknet_train
#from peaknet_train import train_batch, updateGrad, optimize
# from train import train_peaknet
# from preprocess import prep_image, inp_to_image
# from util import loss, loadLabels, IOU

# workPath = "/reg/neh/home/liponan/ai/peaknet4antfarm/"
workPath = "../pytorch-yolo2/"

class Peaknet():

    def __init__(self):
        self.model = None

    def loadDNWeights( self ):
        # self.model = Darknet(workPath + 'cfg/newpeaksv5.cfg')
        # self.model.load_weights(workPath + "weights/newpeaksv5.backup")

        self.model = Darknet(workPath + 'cfg/newpeaksv9.cfg')
        self.model.load_weights(workPath + "weights/newpeaksv9.backup")

    def train( self, imgs, labels, box_size = 7 ):
        peaknet_train.train_batch( self.model, imgs, labels, batch_size=32, box_size=7, use_cuda=True )        

    def model( self ):
        return self.model

    def test( self, imgs, box_size = 7 ):
        results = None
        return results

    def validate( self, imgs, golden_labels, box_size = 7 ):
        results = None
        return results

    def updateModel( self, model ):
        self.model = model

    def updateGrad( self, model ):
        peaknet_train.updateGrad( self.model, model )

    def optimize( self )
        peaknet_train.optimize( self.model )


    # def optimize( self, model ):
        # for param in self.model.parameters():
        #     param.grad.data = model.
