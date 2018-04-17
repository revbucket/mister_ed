""" File that contains various parameterizations for spatial transformation
    styles. At its simplest, spatial transforms can be affine grids,
    parameterized  by 6 values. At their most complex, for a CxHxW type image
    grids can be parameterized by CxHxWx2 parameters.

    This file will define subclasses of nn.Module that will have parameters
    corresponding to the transformation parameters and will take in an image
    and output a transformed image.

    Further we'll also want a method to initialize each set to be the identity
    initially
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import utils.pytorch_utils as utils
from torch.autograd import Variable
import numpy as np



###############################################################################
#                                                                             #
#                  FULLY PARAMETERIZED SPATIAL TRANSFORMATION NETWORK         #
#                                                                             #
###############################################################################

class FullSpatial(nn.Module):
    def __init__(self, img_shape):
        """ FullSpatial just has parameters that are the grid themselves.
            Forward then will just call grid sample using these params directly
        """
        super(FullSpatial, self).__init__()
        self.img_shape = img_shape
        self.grid_params = nn.Parameter(self.identity_params(img_shape))


    @classmethod
    def identity_params(cls, shape):
        """ Returns some grid parameters such that the minibatch of images isn't
            changed when forward is called on it
        ARGS:
            shape: torch.Size - shape of the minibatch of images we'll be
                   transforming. First index should be num examples
        RETURNS:
            torch TENSOR (not variable!!!)
            if shape arg has shape NxCxHxW, this has shape NxCxHxWx2
        """

        # Work smarter not harder -- use idenity affine transforms here
        num_examples = shape[0]


        identity_affine_transform = torch.zeros(num_examples, 2, 3)

        identity_affine_transform[:,0,0] = 1
        identity_affine_transform[:,1,1] = 1

        return F.affine_grid(identity_affine_transform, shape).data

    def clip_params(self):
        """ Clips the parameters to be between -1 and 1 as required for
            grid_sample
        """
        clamp_params = torch.clamp(self.grid_params, -1, 1).data
        self.grid_params = nn.Parameter(clamp_params)

    def project_params(self, lp, lp_bound):
        """ Projects the params to be within lp_bound (according to an lp)
            of the identity map. First thing we do is clip the params to be
            valid, too
        ARGS:
            lp : int or 'inf' - which LP norm we use. Must be an int or the
                 string 'inf'
            lp_bound : float - how far we're allowed to go in LP land
        RETURNS:
            None, but modifies self.grid_params
        """

        assert isinstance(lp, int) or lp == 'inf'

        # clip first
        self.clip_params()

        # then project back

        if lp == 'inf':
            identity_params = self.identity_params(self.img_shape)
            clamp_params = utils.clamp_ref(self.grid_params.data,
                                               identity_params, lp_bound)
            self.grid_params = nn.Parameter(clamp_params)
        else:
            raise NotImplementedError("Only L-infinity bounds working for now ")






    def forward(self, x):
        # usual forward technique
        return F.grid_sample(x, self.grid_params)


