import loss_functions as lf
import torch
""" Examples of prebuilt loss functions """


##############################################################################
#                                                                            #
#                               BASIC CROSS-ENTROPY LOSS                     #
#                                                                            #
##############################################################################


class VanillaXentropy(lf.RegularizedLoss):
    """ Super basic Xentropy Loss """
    def __init__(self, classifier, normalizer=None):
        partial_xentropy = lf.PartialXentropy(classifier, normalizer=normalizer)
        super(VanillaXentropy, self).__init__({'xentropy': partial_xentropy},
                                              {'xentropy': 1.0})

class PerceptualXentropy(lf.RegularizedLoss):
    """ Xentropy loss with a regularization based on perceptual distance """

    def __init__(self, classifier, normalizer=None,
                 regularization_constant=-100.0, use_gpu=False):
        partial_xentropy = lf.PartialXentropy(classifier, normalizer=normalizer)
        lpips_reg = lf.LpipsRegularization(None, use_gpu=use_gpu)

        super(PerceptualXentropy, self).__init__({'xentropy': partial_xentropy,
                                                  'lpips_reg': lpips_reg},
                                                  {'xentropy': 1.0,
                                                   'lpips_reg':
                                                       regularization_constant})



##############################################################################
#                                                                            #
#                          CARLINI WAGNER L2-REGULARIZED LOSS                #
#                                                                            #
##############################################################################


class CWL2Loss(lf.RegularizedLoss):
    """
    Carlini Wagner Loss with F6 loss and l2 norms as outlined in their paper
    ( https://arxiv.org/pdf/1608.04644.pdf)
    Computes loss to be minimized:
            ||input - original||_2^2 + scale_constant * f6(examples, labels)
    """

    def __init__(self, classifier, normalizer, kappa=0.0):

        # build F6 component
        f6_component = lf.CWLossF6(classifier, normalizer=normalizer,
                                   kappa=kappa)

        # build L2 regularization component
        l2_reg = lf.L2Regularization(None)

        super(CWL2Loss, self).__init__({'f6': f6_component,
                                           'l2_reg': l2_reg},
                                          {'f6': None, 'l2_reg': 1.0})

    def forward(self, examples, labels, scale_constant=None,
                targeted=False):
        if scale_constant is not None:
            self.scalars['f6'] = scale_constant


        # if f6's constant isn't setup to be a per-example scalar, make it so
        if isinstance(self.scalars['f6'], float):
            self.scalars['f6'] = (torch.ones(examples.shape[0], 1) *
                                  self.scalars['f6]'])
            
        assert self.scalars['f6'].shape == torch.Size([examples.size(0)])


        return super(CWL2Loss, self).forward(examples, labels,
                                                targeted=targeted)


class CWLInfLoss(lf.RegularizedLoss):
    """ Carlini Wagner loss for L_infinity attacks
    See page 10 of this paper: https://arxiv.org/pdf/1608.04644.pdf
    Specifically we compute the loss
    min c * f(x + delta) + SUM_i(abs(delta_i) - tau)
    """

    def __init__(self, classifier, normalizer, kappa=0.0):

        # build F6 component
        f6_component = lf.CWLossF6(classifier, normalizer=normalizer,
                                   kappa=kappa)

        # build l_inf solf regularization component
        linf_reg = lf.SoftLInfRegularization(None)


        super(CWLInfLoss, self).__init__({'f6': f6_component,
                                          'linf_reg': linf_reg},
                                         {'f6': None, 'linf_reg': 1.0})

    def forward(self, examples, labels, scale_constant=None,
                targeted=False):
        if scale_constant is not None:
            self.scalars['f6'] = scale_constant


        # if f6's constant isn't setup to be a per-example scalar, make it so
        if isinstance(self.scalars['f6'], float):
            self.scalars['f6'] = (torch.ones(examples.shape[0], 1) *
                                  self.scalars['f6]'])

        assert self.scalars['f6'].shape == (examples.shape[0], 1)


        return super(CWLInfLoss, self).forward(examples, labels,
                                               targeted=targeted)





##############################################################################
#                                                                            #
#                       CARLINI WAGNER LPIPS-REGULARIZED LOSS                #
#                                                                            #
##############################################################################


class CWLpipsLoss(lf.RegularizedLoss):
    """
    Carlini Wagner Loss with F6 loss as outlined in their paper
    Computes loss to be minimized:
            LPIPS_DIST(input, original) + scale_constant * f6(examples, labels)
    """

    def __init__(self, classifier, normalizer, kappa=0.0, use_gpu=False):

        # build F6 component
        f6_component = lf.CWLossF6(classifier, normalizer=normalizer,
                                   kappa=kappa)

        # build L2 regularization component
        lpips_reg = lf.LpipsRegularization(None, use_gpu=use_gpu)

        super(CWLpipsLoss, self).__init__({'f6': f6_component,
                                           'lpips_reg': lpips_reg},
                                          {'f6': None, 'lpips_reg': 1.0})


    def forward(self, examples, labels, scale_constant=None,
                targeted=False):
        if scale_constant is not None:
            self.scalars['lpips_reg'] = scale_constant

        assert self.scalars['lpips_reg'] is not None

        return super(CWLpipsLoss, self).forward(examples, labels,
                                                targeted=targeted)

