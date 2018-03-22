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

class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """
    def __init__(self):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()




##############################################################################
#                                                                            #
#                       CROSS ENTROPY LOSS WRAPPER                           #
#                                                                            #
##############################################################################


class PartialXentropy(PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(PartialXentropy, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)

    def forward(self, examples, labels):
        """ Returns XEntropy loss
        ARGS:
            examples: Variable (NxCxHxW) - should be same shape as
                      ctx.fix_im, is the examples we define loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - true classification
                    output for fix_im/examples
        RETURNS:
            scalar loss variable
        """

        if self.normalizer is not None:
            normed_examples = self.normalizer.forward(examples)
        else:
            normed_examples = examples

        criterion = nn.CrossEntropyLoss()
        return criterion(self.classifier.forward(normed_examples), labels)



##############################################################################
#                                                                            #
#                       PERCEPTUALLY REGULARIZED LOSS                        #
#                                                                            #
##############################################################################

class PartialXentropyPerceptual(PartialLoss):
    def __init__(self, fix_im, classifier, penalty=100.0, normalizer=None,
                 use_gpu=False):

        """ Does perceptually regularized loss
        ARGS:
            fix_im : Variable (NxCxHxW) - fixed images, should be in [0.0, 1.0]
                     range
            classifier : torch.nn - image classifier that classifies normalized
                         images
            penalty: float - regularization constant for perceptual loss
            normalizer : DifferentiableNormalize - object that converts
                         [0.0, 1.0] images into std-from-mean for image
                         classification
            use_gpu : bool - if True, we use the GPU for processing perceptual
                             distance
        """
        super(PartialXentropyPerceptual, self).__init__()
        self.fix_im = fix_im
        self.penalty = penalty
        self.classifier = classifier
        self.normalizer = normalizer

        dist_model = dm.DistModel()
        dist_model.initialize(model='net-lin',net='alex',use_gpu=use_gpu)
        self.dist_model = dist_model

        self.nets.extend([self.classifier, self.dist_model.net])

    def forward(self, examples, labels, desired_targets=None):
        """ Returns XEntropy loss
        ARGS:
            examples: Variable (NxCxHxW) - should be same shape as
                      ctx.fix_im, is the examples we define loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - true classification
                    output for fix_im/examples
        RETURNS:
            loss variable
        """

        ##############################################################
        #   Do Xentropy Loss                                         #
        ####################################################3##########

        if self.normalizer is not None:
            normed_examples = self.normalizer.forward(examples)
        else:
            normed_examples = examples

        xentropy = nn.CrossEntropyLoss()
        xentropy_loss = xentropy(self.classifier.forward(normed_examples),
                                 labels)

        ##############################################################
        # Do perceptual loss                                         #
        ##############################################################
        # first convert [0.0, 1.0] images to [-1.0, 1.0 images]
        xform = lambda im: im * 2.0 - 1.0

        perceptual_loss = self.dist_model.forward_var(xform(examples),
                                                      xform(self.fix_im))

        ##############################################################
        #   Combine and return                                       #
        ##############################################################

        return xentropy_loss - self.penalty * perceptual_loss


##############################################################################
#                                                                            #
#                               CARLINI WAGNER LOSS FXNS                     #
#                                                                            #
##############################################################################

class CWLoss(PartialLoss):
    def __init__(self, classifier, normalizer=None, use_gpu=False,
                 perceptual_params=None, kappa=0.):
        """ Computes loss for CarliniWagner attacks
        ARGS:
            classifier : torch.nn - image classifier that classifies normalized
                         images
            normalizer : DifferentiableNormalize - object that converts
                         [0.0, 1.0] images into std-from-mean for image
                         classification
            use_gpu : bool - if True we use GPU for computations
            perceptual_params : dict - if not None is a dict with keys:
                {penalty} [see docs for PerceptualLoss class]
        """

        super(CWLoss, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer or misc.IdentityNormalize()
        self.use_gpu = use_gpu
        self.nets.append(self.classifier)
        self.kappa = kappa
        self.fix_im = None
        self.perceptual_mode = False

        if perceptual_params is not None:
            self.perceptual_mode = True
            self.penalty = perceptual_params['penalty']
            dist_model = dm.DistModel()
            dist_model.initialize(model='net-lin',net='alex',use_gpu=use_gpu)
            self.dist_model = dist_model
            self.nets.append(self.dist_model.net)



    def setup_attack_batch(self, fix_im):
        """ Cleanup function to clear the fixed images after an attack batch
            has been made; also zeros grads
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        self.fix_im = fix_im
        self.zero_grad()


    def cleanup_attack_batch(self):
        """ Cleanup function to clear the fixed images after an attack batch
            has been made; also zeros grads
        """
        self.fix_im = None
        self.zero_grad()


    def forward(self, examples, labels, scale_constant=None, targeted=False):
        """
        Computes loss to be minimized:
            ||input - original||_2^2 + scale_constant * f6(examples, labels)
                                     + penalty * perceptual_distance

        ARGS:
            examples: Variable (NxCxHxW) - Is the examples we compute loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - if not targeted, this is
                    the desired classification output for fix_im, and if
                    targeted, is the desired adversarial target
            scale_constant: float - relative weighting of L2 distance to the
                            objective fxn (f_6 in CW original paper)
            targeted: bool -if True, the labels provided are the DESIRED targets
                      for classification


        RETURNS:
            loss variable
        """

        # Ensure the 'ground images' are okay
        assert self.fix_im is not None

        ######################################################################
        #   Compute L2 distance                                              #
        ######################################################################
        l2_dist = misc.nchw_l2(examples, self.fix_im, squared=True).view(-1, 1)

        ######################################################################
        #   Compute objective function f6                                    #
        ######################################################################

        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)
        num_examples, num_classes = classifier_out.shape

        # get target logits
        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))

        # get largest non-target logits
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max


        if targeted:
            # in targeted case, want to make target most likely
            f6 = torch.clamp(max_other - target_logits, min=-1 * self.kappa)
        else:
            # in NONtargeted case, want to make NONtarget most likely
            f6 = torch.clamp(target_logits - max_other, min=-1 * self.kappa)


        ######################################################################
        #   If applicable, compute perceptual distance                       #
        ######################################################################
        perceptual_loss = 0
        if getattr(self, 'penalty', None) is not None:
            xform = lambda im: im * 2.0 - 1.0

            perceptual_loss = self.dist_model.forward_var(xform(examples),
                                                          xform(self.fix_im))
            perceptual_loss *= self.penalty

        ######################################################################
        #   Combine and return                                               #
        ######################################################################

        if self.perceptual_mode:
            return scale_constant.view(-1, 1) * f6 + perceptual_loss
        else:
            return l2_dist + scale_constant.view(-1, 1)


        # return l2_dist + scale_constant.view(-1, 1) * f6 + perceptual_loss



