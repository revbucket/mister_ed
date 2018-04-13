import custom_lpips.custom_dist_model as dm
import torch.nn as nn
import torch
from numbers import Number
import utils.image_utils as img_utils

""" Loss function building blocks """

##############################################################################
#                                                                            #
#                        LOSS FUNCTION WRAPPER                               #
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

    def forward(self, examples, labels, *args, **kwargs):

        output = None
        for k in self.losses:
            loss = self.losses[k]
            scalar = self.scalars[k]

            loss_val = loss.forward(examples, labels, *args, **kwargs)

            # assert scalar is either a...
            assert (isinstance(scalar, float) or # number
                    scalar.numel() == 1 or # tf wrapping of a number
                    scalar.shape == loss_val.shape) # same as the shape of loss

            if output is None:
                output = loss_val * scalar
            else:
                output = output + loss_val * scalar

        return output


    def setup_attack_batch(self, fix_im):
        """ Setup before calling loss on a new minibatch. Ensures the correct
            fix_im for reference regularizers and that all grads are zeroed
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        for loss in self.losses.itervalues():
            if isinstance(loss, ReferenceRegularizer):
                loss.setup_attack_batch(fix_im)
            else:
                loss.zero_grad()


    def cleanup_attack_batch(self):
        """ Does some cleanup stuff after we finish on a minibatch:
        - clears the fixed images for ReferenceRegularizers
        - zeros grads
        - clears example-based scalars (i.e. scalars that depend on which
          example we're using)
        """
        for loss in self.losses.itervalues():
            if isinstance(loss, ReferenceRegularizer):
                loss.cleanup_attack_batch()
            else:
                loss.zero_grad()

        for key, scalar in self.scalars.items():
            if not isinstance(scalar, Number):
                self.scalars[key] = None


    def zero_grad(self):
        for loss in self.losses.itervalues():
            loss.zero_grad() # probably zeros the same net more than once...



class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """
    def __init__(self):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()


##############################################################################
#                                                                            #
#                                  LOSS FUNCTIONS                            #
#                                                                            #
##############################################################################

############################################################################
#                       NAIVE CORRECT INDICATOR LOSS                       #
############################################################################

class IncorrectIndicator(PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(IncorrectIndicator, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer

    def forward(self, examples, labels, *args, **kwargs):
        """ Returns either (the number | a boolean vector) of examples that
            don't match the labels when run through the
            classifier(normalizer(.)) composition.
        ARGS:
            examples: Variable (NxCxHxW) - should be same shape as
                      ctx.fix_im, is the examples we define loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - true classification
                    output for fix_im/examples
        KWARGS:
            return_type: String - either 'int' or 'vector'. If 'int', we return
                         the number of correctly classified examples,
                         if 'vector' we return a boolean length-N longtensor
                         with the indices of
        RETURNS:
            scalar loss variable or boolean vector, depending on kwargs
        """
        return_type = kwargs.get('return_type', 'int')
        assert return_type in ['int', 'vector']

        class_out = self.classifier.forward(self.normalizer.forward(examples))

        _, outputs = torch.max(class_out, 1)
        incorrect_indicator = outputs != labels

        if return_type == 'int':
            return torch.sum(incorrect_indicator)
        else:
            return incorrect_indicator



##############################################################################
#                                   Standard XEntropy Loss                   #
##############################################################################

class PartialXentropy(PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(PartialXentropy, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)

    def forward(self, examples, labels, *args, **kwargs):
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
#                           Carlini Wagner loss functions                    #
##############################################################################

class CWLossF6(PartialLoss):
    def __init__(self, classifier, normalizer=None, kappa=0.0):
        super(CWLossF6, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)
        self.kappa = kappa


    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get target logits
        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))

        # get largest non-target logits
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max


        if kwargs.get('targeted', False):
            # in targeted case, want to make target most likely
            f6 = torch.clamp(max_other - target_logits, min=-1 * self.kappa)
        else:
            # in NONtargeted case, want to make NONtarget most likely
            f6 = torch.clamp(target_logits - max_other, min=-1 * self.kappa)

        return f6





##############################################################################
#                                                                            #
#                               REFERENCE REGULARIZERS                       #
#                                                                            #
##############################################################################
""" Regularization terms that refer back to a set of 'fixed images', or the
    original images.
    example: L2 regularization which computes L2dist between a perturbed image
             and the FIXED ORIGINAL IMAGE

    NOTE: it's important that these return Variables that are scalars
    (output.numel() == 1), otherwise there's a memory leak w/ CUDA.
    See my discussion on this here:
        https://discuss.pytorch.org/t/cuda-memory-not-being-freed/15965
"""

class ReferenceRegularizer(PartialLoss):
    def __init__(self, fix_im):
        super(ReferenceRegularizer, self).__init__()
        self.fix_im = fix_im

    def setup_attack_batch(self, fix_im):
        """ Setup function to ensure fixed images are set
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
        old_fix_im = self.fix_im
        self.fix_im = None
        del old_fix_im
        self.zero_grad()


#############################################################################
#                               SOFT L_INF REGULARIZATION                   #
#############################################################################

class SoftLInfRegularization(ReferenceRegularizer):
    '''
        see page 10 of this paper (https://arxiv.org/pdf/1608.04644.pdf)
        for discussion on why we want SOFT l inf
    '''
    def __init__(self, fix_im, **kwargs):
        super(SoftLInfRegularization, self).__init__(fix_im)

    def forward(self, examples, *args, **kwargs):
        # ARGS should have one element, which serves as the tau value

        tau =  8.0 / 255.0  # starts at 1 each time?
        scale_factor = 0.9
        l_inf_dist = float(torch.max(torch.abs(examples - self.fix_im)))
        '''
        while scale_factor * tau > l_inf_dist:
            tau *= scale_factor

        assert tau > l_inf_dist
        '''
        delta_minus_taus = torch.clamp(torch.abs(examples - self.fix_im) - tau,
                                       min=0.0)

        return torch.sum(delta_minus_taus)


#############################################################################
#                               L2 REGULARIZATION                           #
#############################################################################

class L2Regularization(ReferenceRegularizer):

    def __init__(self, fix_im, **kwargs):
        super(L2Regularization, self).__init__(fix_im)

    def forward(self, examples, *args, **kwargs):
        l2_dist = img_utils.nchw_l2(examples, self.fix_im,
                                    squared=True).view(-1, 1)
        return torch.sum(l2_dist)

#############################################################################
#                         LPIPS PERCEPTUAL REGULARIZATION                   #
#############################################################################

class LpipsRegularization(ReferenceRegularizer):

    def __init__(self, fix_im, **kwargs):
        super(LpipsRegularization, self).__init__(fix_im)

        use_gpu = kwargs.get('use_gpu', False)
        self.use_gpu = use_gpu
        self.dist_model = dm.DistModel(net='alex', use_gpu=self.use_gpu)

    def forward(self, examples, *args, **kwargs):
        xform = lambda im: im * 2.0 - 1.0
        perceptual_loss = self.dist_model.forward_var(examples,
                                                      self.fix_im)
        return torch.sum(perceptual_loss)




