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


cwd = os.path.abspath(os.path.dirname(__file__))


class Peaknet():

    def __init__(self):
        self.model = None

    def loadWeights( self, cfgFile, weightFile ):
        self.model = Darknet( cfgFile )
        self.model.load_weights( weightFile )

    def loadDNWeights( self ):
        # self.model = Darknet(workPath + 'cfg/newpeaksv5.cfg')
        # self.model.load_weights(workPath + "weights/newpeaksv5.backup")

        #self.model = Darknet( os.path.join( cwd, workPath, 'cfg/newpeaksv9-asic.cfg' ) )
        #self.model.load_weights( os.path.join( cwd, workPath, "weights/newpeaksv9_40000.weights") )
        self.model = Darknet( os.path.join( cwd, workPath, 'cfg/newpeaksv10-asic.cfg' ) )
        #self.model.load_weights( os.path.join( cwd, workPath, "weights/newpeaksv10_40000.weights") )
        self.model.load_weights( os.path.join( cwd, workPath, "../darknet/backup/newpeaksv10_100.weights") )


    def train( self, imgs, labels, box_size = 7, batch_size=1, use_cuda=True, writer=None ):
        """
        labels:
            class, s, r, c, h, w
                0: peak
                1: streak
        """
        peaknet_train.train_batch( self.model, imgs, labels, batch_size=batch_size, box_size=box_size, 
                                    use_cuda=use_cuda, writer=writer)        

    def init_model( self, cfgFile=os.path.join( cwd, workPath, 'cfg/newpeaksv10-asic.cfg' ) ):
        self.model = Darknet( cfgFile )
        peaknet_train.init_model( self.model )

    
    def model( self ):
        return self.model

    def getGrad( self ):
        grad = {}
        model_dict = dict( self.model.named_parameters() )
        for key, val in model_dict.items():
            grad[key] = val.grad.cpu()
        return grad

    def test( self, imgs, box_size = 7 ):
        results = None
        return results

    def validate( self, imgs, golden_labels, box_size = 7 ):
        results = None
        return results

    def updateModel( self, model ):
        self.model = model

    def updateGrad( self, grads ):
        peaknet_train.updateGrad( self.model, grads )

    def optimizer( self, adagrad=False, lr=0.001 ):
        optimizer = peaknet_train.optimizer( self.model, adagrad=adagrad, lr=lr )
        return optimizer     


    def optimize( self, optimizer ):
        peaknet_train.optimize( self.model, optimizer )


    # def optimize( self, model ):
        # for param in self.model.parameters():
        #     param.grad.data = model.
