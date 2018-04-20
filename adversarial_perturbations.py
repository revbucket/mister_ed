""" File that holds adversarial perturbations as torch.nn.Modules.
    An adversarial perturbation is an example-specific

TODO: this needs to be fleshed out, as a general technique to make adversarial
      perturbations.
"""

import torch
import torch.nn as nn
import spatial_transformers as st
import utils.pytorch_utils as utils
from torch.autograd import Variable

##############################################################################
#                                                                            #
#                                   SKELETON CLASS                           #
#                                                                            #
##############################################################################

class AdversarialPerturbation(nn.Module):
    """ Skeleton class to hold adversarial perturbations FOR A SINGLE MINIBATCH.
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class

        All subclasses need the following:
        - perturbation_norm() : no args -> scalar Variable
        - self.parameters() needs to iterate over params we want to optimize
        - constrain_params() : no args -> no return,
             modifies the parameters such that this is still a valid image
        - forward : no args -> Variable - applies the adversarial perturbation
                    the originals and outputs a Variable of how we got there
        - adversarial_tensors() : applies the adversarial transform to the
                                  originals and outputs TENSORS that are the
                                  adversarial images
    """

    def __init__(self):

        super(AdversarialPerturbation, self).__init__()
        # Stores parameters of the adversarial perturbation and hyperparams
        # to compute total perturbation norm here

    def __call__(self, reference=None):
        return self.forward(reference=reference)

    def instantiate(self, originals):
        self.originals = originals

    def _assert_instantiated(self):
        assert self.originals is not None

    def perturbation_norm(self):
        raise NotImplementedError("Need to call subclass method here")

    def constrain_params(self):
        raise NotImplementedError("Need to call subclass method here")

    def forward(self, reference=None):
        raise NotImplementedError("Need to call subclass method here")

    def add_to_params(self, var):
        raise NotImplementedError("Need to call subclass method here")

    def adversarial_tensors(self):
        return self.forward().data


class PerturbationParameters(dict):
    """ Object that stores parameters like a dictionary.
        This allows perturbation classes to be only partially instantiated and
        then fed various 'originals' later.
    Implementation taken from : https://stackoverflow.com/a/14620633/3837607
    """
    def __init__(self, *args, **kwargs):
        super(PerturbationParameters, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

class ThreatModel(object):
    def __init__(self, perturbation_class, *param_kwargs):
        """ Factory class to generate per_minibatch instances of Adversarial
            perturbations.
        ARGS:
            perturbation_class : class - subclass of Adversarial Perturbations
            param_kwargs : dict - dict containing named kwargs to instantiate
                           the class in perturbation class
        """
        assert issubclass(perturbation_class, AdversarialPerturbation)
        self.perturbation_class = perturbation_class
        self.param_kwargs = param_kwargs

    def __call__(self, originals):
        return self.perturbation_obj(originals)

    def perturbation_obj(self, originals):
        return self.perturbation_class(originals, *self.param_kwargs)



##############################################################################
#                                                                            #
#                            ADDITION PARAMETERS                             #
#                                                                            #
##############################################################################

class DeltaAddition(AdversarialPerturbation):

    def __init__(self, originals, perturbation_params):
        """ Maintains a delta that gets addded to the originals to generate
            adversarial images. This is the type of adversarial perturbation
            that the literature extensivey studies
        ARGS:
            originals : Tensor (NxCxHxW) - original images that get perturbed
            perturbation_params: PerturbationParameters object.
                { lp_style : None, int or 'inf' - if not None is the type of
                            Lp_bound that we apply to this adversarial example
                lp_constraint : None or float - cannot be None if lp_style is
                                not None, but if not None should be the lp bound
                                we allow for adversarial perturbations
                custom_norm : None or fxn:(NxCxHxW) -> Scalar Variable. This is
                              not implemented for now
                }
        """

        super(DeltaAddition, self).__init__()
        self.params = nn.Parameter(torch.zeros(originals.shape))
        self.originals = originals
        self.original_var = Variable(originals)
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        if perturbation_params.custom_norm is not None:
            raise NotImplementedError("Only LP norms allowed for now")


    def perturbation_norm(self, lp='inf'):
        assert isinstance(lp, int) or lp == 'inf'
        return utils.summed_lp_norm(self.params, lp=lp)


    def constrain_params(self, reference=None):
        # First throw params onto valid range of images
        reference = self.original_var if reference is None else reference
        perturbed = utils.clip_0_1(self.params + reference)
        new_delta = perturbed - self.original_var

        # Then do lp projections
        if self.lp_style == 'inf':
            new_delta = torch.clamp(new_delta, -self.lp_bound,
                                                self.lp_bound)
        else:
            batchwise_norms = utils.batchwise_norm(delta, self.lp_style, dim=0)
            # ughh..... do this later
            raise NotImplementedError("Non LInf norms not implemented")

        self.params = nn.Parameter(new_delta.data)

    def add_to_params(self, var_to_add):
        """ sets params to be self.params + var_to_add """
        self.params = nn.Parameter()


    def forward(self, reference=None):
        self.constrain_params(reference=reference)
        reference = self.original_var if reference is None else reference
        return reference + self.params


##############################################################################
#                                                                            #
#                               SPATIAL PARAMETERS                           #
#                                                                            #
##############################################################################

class ParameterizedXformAdv(AdversarialPerturbation):

    def __init__(self, originals, perturbation_params):
        super(ParameterizedXformAdv, self).__init__()
        assert issubclass(perturbation_params.xform_class,
                          st.ParameterizedTransformation)
        self.xform = perturbation_params.xform_class(shape=originals.shape)
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        self.originals = originals
        self.original_var = Variable(originals)

    def perturbation_norm(self):
        return self.xform.norm(lp=self.lp_style)

    def constrain_params(self, reference=None):
        # Do lp projections


        if isinstance(self.lp_style, int) or self.lp_style == 'inf':
            self.xform.project_params(self.lp_style, self.lp_bound)

    def forward(self, reference=None):
        reference = self.original_var if reference is None else reference
        self.constrain_params()
        return self.xform.forward(reference)




##############################################################################
#                                                                            #
#                            SPATIAL + ADDITION PARAMETERS                   #
#                                                                            #
##############################################################################

class SequentialPerturbation(AdversarialPerturbation):
    """ Takes a list of perturbations and composes them. A norm needs to
        be specified here to describe the perturbations.
    """

    def __init__(self, originals, perturbation_sequence,
                 global_parameters=PerturbationParameters()):
        """ Initializes a sequence of adversarial perturbation layers
        ARGS:
            originals : NxCxHxW tensor - original images we create adversarial
                        perturbations for
            perturbation_sequence : (Class, PerturbationParameters)[]  -
                list of ThreatModel objects
            total_parameters : PerturbationParameters - global parameters to
                               use. These contain things like how to norm this
                               sequence, how to constrain this sequence, etc
         """
        super(SequentialPerturbation, self).__init__()

        self.pipeline = []
        for layer_no, threat_model in enumerate(perturbation_sequence):
            assert isinstance(threat_model, ThreatModel)
            layer = threat_model(originals)

            self.pipeline.append(layer)
            self.add_module('layer_%02d' % layer_no, layer)

        self.originals = originals
        self.original_var = Variable(originals)

        # norm: pipeline -> Scalar Variable
        default_norm = lambda pipe: torch.sum(pipe.perturbation_norm)
        self.norm = global_parameters.norm or default_norm


    def perturbation_norm(self):
        # Need to define a nice way to describe the norm here. This can be
        # an empirical norm between input/output

        # For now, let's just say it's the sum of the norms of each constituent
        return self.norm(self.pipeline)

    def constrain_params(self, reference=None):
        # Need to do some sort of crazy projection operator for general things
        # For now, let's just constrain each thing in sequence
        reference = self.original_var if reference is None else reference
        for layer in self.pipeline:
            layer.constrain_params(reference)
            reference = layer(reference=reference)

    def forward(self, reference=None):
        self.constrain_params()
        output = self.original_var
        for layer in self.pipeline:
            output = layer(reference=output)
        return torch.clamp(output, 0, 1)




