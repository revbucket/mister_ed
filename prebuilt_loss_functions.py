import loss_functions as lf

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


class CWPaperLoss(lf.RegularizedLoss):
    """
    Carlini Wagner Loss with F6 loss as outlined in their paper
    Computes loss to be minimized:
            ||input - original||_2^2 + scale_constant * f6(examples, labels)
    """

    def __init__(self, classifier, normalizer, kappa=0.0):

        # build F6 component
        f6_component = lf.CWLossF6(classifier, normalizer=normalizer,
                                   kappa=kappa)

        # build L2 regularization component
        l2_reg = lf.L2Regularization(None)

        super(CWPaperLoss, self).__init__({'f6': f6_component,
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

        assert self.scalars['f6'].shape == (examples.shape[0], 1)


        return super(CWPaperLoss, self).forward(examples, labels,
                                                targeted=targeted)



##############################################################################
#                                                                            #
#                       CARLINI WAGNER LPIPS-REGULARIZED LOSS                #
#                                                                            #
##############################################################################


class CWLpipsLoss(object):
    """
    Carlini Wagner Loss with F6 loss as outlined in their paper
    Computes loss to be minimized:
            LPIPS_DIST(input, original) + scale_constant * f6(examples, labels)
    """

    def __init__(self, classifier, normalizer, kappa=0.0):

        # build F6 component
        f6_component = lf.CWLossF6(classifier, normalizer=normalizer,
                                   kappa=kappa)

        # build L2 regularization component
        lpips_reg = lf.LpipsRegularization(None)

        super(CWPaperLoss, self).__init__({'f6': f6_component,
                                           'lpips_reg': lpips_reg},
                                          {'f6': 1.0, 'lpips_reg': None})


    def forward(self, examples, labels, regularization_constant=None,
                targeted=False):
        if regularization_constant is not None:
            self.scalars['lpips_reg'] = regularization_constant

        assert self.scalars['lpips_reg'] is not None

        return super(CWPaperLoss, self).forward(examples, labels,
                                                targeted=targeted)

