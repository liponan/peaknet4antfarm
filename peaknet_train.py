from __future__ import print_function
import sys

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import peaknet_dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet

def updateGrad( model, grad ):
    #with torch.no_grad():
    model_dict = dict( model.named_parameters() )
    #model_dict2 = dict( model2.named_parameters() )
    for key, value in model_dict.items():
        #model_dict[key].grad.data = grad[key].data
        model_dict[key]._grad = grad[key]
    model.cuda()



def optimize( model, adagrad=False, lr=0.001 ):
    # lr = learning_rate/batch_size
    if adagrad:
        #lr = 0.0005
        decay = 0.005
        optimizer = optim.Adagrad(model.parameters(), lr = lr, weight_decay=decay)
    else:
        #lr = 0.001
        momentum = 0.9
        decay = 0.0005
        optimizer = optim.SGD(model.parameters(), lr=lr,
                            momentum=momentum, dampening=0,
                            weight_decay=decay)
    optimizer.step()



def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train_batch( model, imgs, labels, batch_size=32, box_size=7, use_cuda=True, writer=None ):
    debug = True
    
    train_loader = torch.utils.data.DataLoader(
        peaknet_dataset.listDataset(imgs, labels,
                        shape=(imgs.shape[2], imgs.shape[3]),
                        shuffle=True,
                        #transform=transforms.Compose([
                        #    transforms.ToTensor(),
                        #    ]),
                        transform=None,
                        train=True,
                        box_size=box_size,
                        # seen=cur_model.seen,
                        batch_size=batch_size
                        # num_workers=num_workers
                        ),
        # batch_size=batch_size, shuffle=False, **kwargs)
        batch_size=batch_size, shuffle=False)

    # lr = adjust_learning_rate(optimizer, processed_batches)
    # logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    region_loss = model.loss
    region_loss.seen = model.seen
    t1 = time.time()
    avg_time = torch.zeros(9)

    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        # adjust_learning_rate(optimizer, processed_batches)
        # processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')
        print("timgs type", data.type())
        if use_cuda:
            data = data.cuda()
            target= target.cuda()
        t3 = time.time()
        #print( "before", data )
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        # optimizer.zero_grad()
        t5 = time.time()
        #print( "after", data )
        #output = model( data )
        output, _= model( data )


        #print(output)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        model.seen = region_loss.seen 
        #try:
        if debug:
            print("output", output.size())
            print("label length", len(target))
            print("label[0] length", len(target[0]))
        loss = region_loss(output, target)
      
        if False:
            print("label length", len(target))
            print("label[0]", len(target[0]))
            print("label[0]", target[0])
            #print("label[0]", target[0].shape)
            #print("label[1]", target[1].shape)
            #print("label[2]", target[2].shape)
            raise "something wrong with the labels?"
      
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        # optimizer.step()
        t9 = time.time()
        if writer != None:
            writer.add_scalar('loss', loss, model.seen) 
        #writer.export_scalars_to_json("./all_scalars.json")
        
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
    #     t1 = time.time()
    # print('')
    # t1 = time.time()
    # logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    # if (epoch+1) % save_interval == 0:
    #     logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
    #     cur_model.seen = (epoch + 1) * len(train_loader.dataset)
    #     cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))



'''
def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

class Trainer:

    def __init__( self, datacfg, cfgfile, weightfile ):
        data_options  = read_data_cfg(datacfg)
        net_options   = parse_cfg(cfgfile)[0]

        trainlist     = data_options['train']
        testlist      = data_options['valid']
        backupdir     = data_options['backup']
        nsamples      = file_lines(trainlist)
        gpus          = data_options['gpus']  # e.g. 0,1,2,3
        ngpus         = len(gpus.split(','))
        num_workers   = int(data_options['num_workers'])

        batch_size    = int(net_options['batch'])
        max_batches   = int(net_options['max_batches'])
        learning_rate = float(net_options['learning_rate'])
        momentum      = float(net_options['momentum'])
        decay         = float(net_options['decay'])
        steps         = [float(step) for step in net_options['steps'].split(',')]
        scales        = [float(scale) for scale in net_options['scales'].split(',')]

        #Train parameters
        max_epochs    = max_batches*batch_size/nsamples+1
        use_cuda      = False # True
        seed          = 231 #int(time.time())
        eps           = 1e-5
        save_interval = 10  # epoches
        dot_interval  = 70  # batches

        # Test parameters
        conf_thresh   = 0.25
        nms_thresh    = 0.4
        iou_thresh    = 0.5

def new_model( cfgfile, weightfile=None ):
    model = Darknet(cfgfile)
    if weightfile != None:
        model.load_weights(weightfile)
    model.print_network()
    return model

def train_peaknet( model, trainer, imgs, labels, tmpdir ):


    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    ###############
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    region_loss = model.loss

    region_loss.seen  = model.seen
    processed_batches = model.seen/trainer.batch_size

    nsamples = imgs.shape[0]

    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen/nsamples

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    # test_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(testlist, shape=(init_width, init_height),
    #                    shuffle=False,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                    ]), train=False),
    #     batch_size=batch_size, shuffle=False, **kwargs)

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]


    # evaluate = False
    # if evaluate:
    #     logging('evaluating ...')
    #     test(0)
    # else:
    # for epoch in range(init_epoch, max_epochs):
    #     train(epoch)
        # test(epoch)
'''
