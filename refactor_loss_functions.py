import models.dist_model as dm
import torch.nn as nn
import misc
import torch
''' Defines a loss function for perceptual distance '''

##############################################################################
#                                                                            #
#                        PARTIAL LOSS WRAPPER                                #
#                                                                            #
##############################################################################

class RegularizedLoss(object):
    """ Wrapper for multiple PartialLoss objects where we combine with
        regularization constants """
    def __init__(self, losses, scalars):
        """
        ARGS:
            losses : dict - dictionary of partialLoss objects, each is keyed
                            with a nice identifying name
            scalars : dict - dictionary of scalars, each is keyed with the
                             same identifying name as is in self.losses
        """

        assert sorted(losses.keys()) == sorted(scalars.keys())

        self.losses = losses
        self.scalars = scalars

    def forward(self, examples, labels):
        pairs = [(self.losses[k], self.scalars[k]) for k in self.losses]
        return sum([scalar * loss.forward(examples, labels)
                    for loss, scalar in pairs])



class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """
    def __init__(self, ):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()

