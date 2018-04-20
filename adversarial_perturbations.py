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
import functools

# assert initialized decorator
def initialized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.initialized
        return func(self, *args, **kwargs)
    return wrapper

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
        self.initialized = False
        # Stores parameters of the adversarial perturbation and hyperparams
        # to compute total perturbation norm here

    def __call__(self, reference=None):
        return self.forward(reference=reference)

    def setup(self):
        pass

    @initialized
    def perturbation_norm(self, x=None):
        """ This returns the 'norm' of this perturbation. Optionally, for
            certain norms, having access to the images for which the
            perturbation is intended can have an effect on the output.
        ARGS:
            x : Variable or Tensor (NxCxHxW) - optionally can be the images
                that the perturbation was intended for
        RETURNS:
            Scalar Variable
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def constrain_params(self):
        """ This modifies the parameters such that the perturbation falls within
            the threat model it belongs to. E.g. for l-infinity threat models,
            this clips the params to match the right l-infinity bound.

            TODO: for non-lp norms, projecting to the nearest point in the level
                  set
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def make_valid_image(self, x):
        """ This takes in the minibatch self's parameters were tuned for and
            clips the parameters such that this is still a valid image.
        ARGS:
            x : Variable or Tensor (NxCxHxW) - the images this this perturbation
                was intended for
        RETURNS:
            None
        """
        pass # Only implement in classes that can create invalid images

    @initialized
    def forward(self, x):
        """ This takes in the minibatch self's parameters were tuned for and
            outputs a variable of the perturbation applied to the images
        ARGS:
            x : Variable (NxCxHxW) - the images this this perturbation
                was intended for
        RETURNS:
            Variable (NxCxHxW) - the perturbation applied to the input images
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def add_to_params(self, grad_data):
        """ This takes in a Tensor the same shape as self's parameters and
            adds to them. Note that this usually won't preserve gradient
            information
            (also this might have different signatures in subclasses)
        ARGS:
            x : Tensor (params-shape) - Tensor to be added to the
                parameters of self
        RETURNS:
            None, but modifies self's parameters
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def adversarial_tensors(self, x):
        """ Little helper method to get the tensors of the adversarial images
            directly
        """
        return self.forward(x).data

    @initialized
    def attach_originals(self, originals):
        """ Little helper method to tack on the original images to self to
            pass around the (images, perturbation) in a single object
        """
        self.originals = originals



class PerturbationParameters(dict):
    """ Object that stores parameters like a dictionary.
        This allows perturbation classes to be only partially instantiated and
        then fed various 'originals' later.
    Implementation taken from : https://stackoverflow.com/a/14620633/3837607
    (and then modified with the getattribute trick to return none instead of
     error for missing attributes)
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
    def __init__(self, perturbation_class, param_kwargs, *other_args):
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
        return self.perturbation_class(originals, self.param_kwargs,
                                       *other_args)



##############################################################################
#                                                                            #
#                            ADDITION PARAMETERS                             #
#                                                                            #
##############################################################################

class DeltaAddition(AdversarialPerturbation):

    def __init__(self, perturbation_params):
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
        self.delta = nn.Parameter(torch.zeros(originals.shape))
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        if perturbation_params.custom_norm is not None:
            raise NotImplementedError("Only LP norms allowed for now")
        self.initialized = True

    @initialized
    def perturbation_norm(self):
        assert isinstance(self.lp_style, int) or self.lp_style == 'inf'
        return utils.summed_lp_norm(self.params, lp=self.lp_style)


    @initialized
    def constrain_params(self):
        new_delta = self.delta.data

        if self.lp_style == 'inf':
            new_delta = torch.clamp(new_delta, -self.lp_bound,
                                                self.lp_bound)
        else:
            # ughh..... I'll do this later
            raise NotImplementedError("Non LInf norms not implemented")

        self.delta = nn.Parameter(new_delta)

    @initialized
    def make_valid_image(self, x):
        change_in_delta = utils.clamp_0_1_delta(new_delta,
                                                utils.safe_tensor(x))
        self.delta.data.add_(change_in_delta)


    @initialized
    def add_to_params(self, grad_data):
        """ sets params to be self.params + grad_data """
        self.delta.data.add_(grad_data)


    @initialized
    def forward(self, x):

        self.make_valid_image(x) # not sure which one to do first...
        self.constrain_params(x)

        return x + self.delta


##############################################################################
#                                                                            #
#                               SPATIAL PARAMETERS                           #
#                                                                            #
##############################################################################

class ParameterizedXformAdv(AdversarialPerturbation):

    def __init__(self, perturbation_params):
        super(ParameterizedXformAdv, self).__init__()
        assert issubclass(perturbation_params.xform_class,
                          st.ParameterizedTransformation)

        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound

    def setup(self, originals):
        self.xform = perturbation_params.xform_class(shape=originals.shape)
        self.initialized = True

    @initialized
    def perturbation_norm(self):
        return self.xform.norm(lp=self.lp_style)

    @initialized
    def constrain_params(self):
        # Do lp projections
        if isinstance(self.lp_style, int) or self.lp_style == 'inf':
            self.xform.project_params(self.lp_style, self.lp_bound)

    def forward(self, x):
        self.setup(x)
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

    def __init__(self, perturbation_sequence,
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
        self.norm = global_parameters.norm


    def setup(self, x):
        for layer in self.pipeline:
            layer.setup(x)
        self.initialized = True


    @initialized
    def perturbation_norm(self, x=None):
        # Need to define a nice way to describe the norm here. This can be
        # an empirical norm between input/output

        # For now, let's just say it's the sum of the norms of each constituent
        if self.norm is None:
            return self.norm(self.pipeline, x=x)
        else:
            out = None
            for layer in self.pipeline:
                if out is None:
                    out = layer.perturbation_norm(x=x)
                else:
                    out = out + layer.perturbation_norm(x=x)
            return out

    @initialized
    def make_valid_image(self, x):
        for layer in self.pipeline:
            layer.make_valid_image(x)
            x = layer(x)


    @initialized
    def constrain_params(self, x=None):
        # Need to do some sort of crazy projection operator for general things
        # For now, let's just constrain each thing in sequence

        for layer in self.pipeline:
            layer.constrain_params(x=x)
            if x is not None:
                x = layer(x)



    def forward(self, x):
        self.setup(x)
        self.constrain_params(x)
        self.make_valid_image(x)

        for layer in self.pipeline:
            x = layer(x)

        return x





