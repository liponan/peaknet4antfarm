import os
import os.path as osp
import time
import torch as t
import sys
# sys.path.append(os.path.abspath('../pytorch-yolo2'))
from darknet import Darknet
import peaknet_train
from peaknet_validate import validate_batch
from peaknet_test import test_batch
from tensorboardX import SummaryWriter

# workPath = "/reg/neh/home/liponan/ai/peaknet4antfarm/"
workPath = "../pytorch-yolo2/"

cwd = os.path.abspath(os.path.dirname(__file__))

class Peaknet():

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.writer = None

    def set_writer(self, project_name=None, parameters={}):
        if project_name == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter( project_name )
        self.writer.add_custom_scalars( parameters )

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
        peaknet_train.train_batch( self.model, imgs, labels, batch_size=batch_size,
                                box_size=box_size, use_cuda=use_cuda, writer=writer)

    def model( self ):
        return self.model

    def getGrad( self ):
        grad = {}
        model_dict = dict( self.model.named_parameters() )
        for key, val in model_dict.items():
            grad[key] = val.grad.cpu()
        return grad

    def predict( self, imgs, box_size = 7 ):
        results = predict_batch( self.model, imgs, batch_size=batch_size,
                                box_size=box_size, use_cuda=use_cuda)
        return results

    def validate( self, imgs, golden_labels, box_size = 7 ):
        results = validate_batch( self.model, imgs, labels, batch_size=batch_size,
                                box_size=box_size, use_cuda=use_cuda, writer=writer)
        return results

    def updateModel( self, model ):
        self.model = model

    def updateGrad( self, grads ):
        peaknet_train.updateGrad( self.model, grads )

    def set_optimizer( self, adagrad=False, lr=0.001 ):
        self.optimizer = peaknet_train.optimizer( self.model, adagrad=adagrad, lr=lr )

    def optimize( self ):
        peaknet_train.optimize( self.model, self.optimizer )
