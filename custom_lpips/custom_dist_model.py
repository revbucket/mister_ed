""" Custom net to hopefully avoid memory leak """
import torch
import torch.nn as nn
import torch.nn.init as init
import utils.pytorch_utils as utils
from collections import namedtuple

from torchvision import models
from custom_lpips.base_model import BaseModel
import os

###############################################################################
#                                                                             #
#                               NN Architecture                               #
#                                                                             #
###############################################################################


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs",
                                  ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


##############################################################################
#                                                                            #
#                           NN Functional Code                               #
#                                                                            #
##############################################################################

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_tune=False, use_dropout=False,
                 manual_gpu=None):

        # HACKETY HACK -- MJ modified this file
        super(PNetLin, self).__init__()
        net_type = alexnet # ADD FREEDOM HERE LATER


        self.pnet_tune = pnet_tune
        self.chns = [64,192,384,256,256]
        if self.pnet_tune:
            self.net = net_type(requires_grad=self.pnet_tune)
        else:
            self.net = [net_type(requires_grad=self.pnet_tune),]


        # define the layers
        self.lin0 = NetLinLayer(self.chns[0],use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1],use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2],use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3],use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4],use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]

        # define transfrom to make mean 0, unit var
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))

        # cuda all the things
        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()
        if self.use_gpu:
            if self.pnet_tune:
                self.net.cuda()
            else:
                self.net[0].cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()


    def forward(self, in0, in1):

        # normalize
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)


        if self.pnet_tune:
            outs0 = self.net.forward(in0_sc)
            outs1 = self.net.forward(in1_sc)
        else:
            outs0 = self.net[0].forward(in0_sc)
            outs1 = self.net[0].forward(in1_sc)


        diffs = []
        for kk in range(len(outs0)):
            normed_0 = normalize_tensor(outs0[kk])
            normed_1 = normalize_tensor(outs1[kk])
            diffs.append((normed_0 - normed_1) ** 2)

        val = 0
        for i in range(len(self.lins)):
            val = val + torch.mean(
                            torch.mean(self.lins[i].model(diffs[i]), dim=3),
                            dim=2)
        return val.view(val.size()[0],val.size()[1],1,1)




class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


def normalize_tensor(in_feat,eps=1e-10):
    root_sum_square = torch.sqrt(torch.sum(in_feat ** 2, dim=1))
    og_size = in_feat.size()

    norm_factor = root_sum_square.view(og_size[0], 1,
                                       og_size[2], og_size[3]) + eps
    return in_feat / norm_factor


###############################################################################
#                                                                             #
#                               Distance model                                #
#                                                                             #
###############################################################################


class DistModel(BaseModel):

    def __init__(self, net='squeeze', manual_gpu=None):

        super(DistModel, self).__init__(manual_gpu=manual_gpu)

        if self.use_gpu:
            self.map_location = None
        else:
            self.map_location = lambda storage, loc: storage

        self.net = PNetLin(manual_gpu=manual_gpu, pnet_tune=False,
                           use_dropout=True)
        weight_path =  os.path.join(os.path.dirname(__file__), 'weights',
                                    '%s.pth' % net)
        self.net.load_state_dict(torch.load(weight_path,
                                            map_location=self.map_location))

        self.parameters = list(self.net.parameters())
        self.net.eval()


    def forward_var(self, input_0, input_1):
        # input_0 and input_1 are both NxCxHxW VARIABLES!
        return self.net.forward(input_0, input_1)


    def zero_grad(self):
        self.net.zero_grad()

