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
        assert self.initialized, ("Parameters not initialized yet. "
                                  "Call .forward(...) first")
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

    def __init__(self, perturbation_params):

        super(AdversarialPerturbation, self).__init__()
        self.initialized = False
        self.perturbation_params = perturbation_params
        # Stores parameters of the adversarial perturbation and hyperparams
        # to compute total perturbation norm here

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        print type(self.perturbation_params)
        if isinstance(self.perturbation_params, tuple):
            output_str = "[Perturbation] %s: %s" % (self.__class__.__name__,
                                                    self.perturbation_params[1])
            output_str += '\n['
            for el in self.perturbation_params[0]:
                output_str += '\n\t%s,' % el
            output_str += '\n]'
            return output_str
        else:
            return "[Perturbation] %s: %s"  % (self.__class__.__name__,
                                               self.perturbation_params)

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
    def update_params(self, step_fxn):
        """ This takes in a function step_fxn: Tensor -> Tensor that generates
            the change to the parameters that we step along. This loops through
            all parameters and updates signs accordingly.
            For sequential perturbations, this also multiplies by a scalar if
            provided
        ARGS:
            step_fxn : Tensor -> Tensor - function that maps tensors to tensors.
                       e.g. for FGSM, we want a function that multiplies signs
                       by step_size
        RETURNS:
            None, but updates the parameters
        """
        raise NotImplementedError("Need to call subclass method here")


    @initialized
    def adversarial_tensors(self, x=None):
        """ Little helper method to get the tensors of the adversarial images
            directly
        """
        assert x is not None or self.originals is not None
        if x is None:
            x = self.originals

        return self.forward(utils.safe_var(x)).data

    @initialized
    def attach_originals(self, originals):
        """ Little helper method to tack on the original images to self to
            pass around the (images, perturbation) in a single object
        """
        self.originals = originals


    @initialized
    def random_init(self):
        """ Modifies the parameters such that they're randomly initialized
            uniformly across the threat model (this is harder for nonLp threat
            models...). Takes no args and returns nothing, but modifies the
            parameters
        """
        raise NotImplementedError("Need to call subclass method here")




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
        self.other_args = other_args

    def __repr__(self):
        return "[Threat] %s: %s"  % (str(self.perturbation_class.__name__),
                                     self.param_kwargs)

    def __call__(self, *args):
        if args == ():
            return self.perturbation_obj()
        else:
            perturbation_obj = self.perturbation_obj()
            perturbation_obj.setup(*args)
            return perturbation_obj



    def perturbation_obj(self):
        return self.perturbation_class(self.param_kwargs, *self.other_args)



##############################################################################
#                                                                            #
#                            ADDITION PARAMETERS                             #
#                                                                            #
##############################################################################

class DeltaAddition(AdversarialPerturbation):

    def __init__(self, perturbation_params, *other_args):
        """ Maintains a delta that gets addded to the originals to generate
            adversarial images. This is the type of adversarial perturbation
            that the literature extensivey studies
        ARGS:
            originals : Tensor (NxCxHxW) - original images that get perturbed
            perturbation_params: PerturbationParameters object.
                { lp_style : None, int or 'inf' - if not None is the type of
                            Lp_bound that we apply to this adversarial example
                lp_bound : None or float - cannot be None if lp_style is
                           not None, but if not None should be the lp bound
                           we allow for adversarial perturbations
                custom_norm : None or fxn:(NxCxHxW) -> Scalar Variable. This is
                              not implemented for now
                }
        """

        super(DeltaAddition, self).__init__(perturbation_params)
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        if perturbation_params.custom_norm is not None:
            raise NotImplementedError("Only LP norms allowed for now")
        self.scalar_step = perturbation_params.scalar_step or 1.0

    def setup(self, x):
        self.delta = nn.Parameter(torch.zeros(x.shape))
        self.initialized = True

    @initialized
    def perturbation_norm(self):
        assert isinstance(self.lp_style, int) or self.lp_style == 'inf'
        return utils.summed_lp_norm(self.delta, lp=self.lp_style)


    @initialized
    def constrain_params(self):
        new_delta = utils.batchwise_lp_project(self.delta.data, self.lp_style,
                                               self.lp_bound)
        self.delta = nn.Parameter(new_delta)

    @initialized
    def make_valid_image(self, x):
        new_delta = self.delta.data
        try:
            change_in_delta = utils.clamp_0_1_delta(new_delta,
                                                    utils.safe_tensor(x))
        except:
            print new_delta.shape
            print x.shape
            natoeuhsntauhe
        self.delta.data.add_(change_in_delta)

    @initialized
    def update_params(self, step_fxn):
        assert self.delta.grad.data is not None
        self.add_to_params(step_fxn(self.delta.grad.data) * self.scalar_step)

    @initialized
    def add_to_params(self, grad_data):
        """ sets params to be self.params + grad_data """
        self.delta.data.add_(grad_data)


    @initialized
    def random_init(self):
        self.delta = nn.Parameter(utils.random_from_lp_ball(self.delta.data,
                                                            self.lp_style,
                                                            self.lp_bound))


    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.make_valid_image(x) # not sure which one to do first...
        self.constrain_params()
        return x + self.delta



##############################################################################
#                                                                            #
#                               SPATIAL PARAMETERS                           #
#                                                                            #
##############################################################################

class ParameterizedXformAdv(AdversarialPerturbation):

    def __init__(self, perturbation_params, *other_args):
        super(ParameterizedXformAdv, self).__init__(perturbation_params)
        assert issubclass(perturbation_params.xform_class,
                          st.ParameterizedTransformation)

        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        self.scalar_step = perturbation_params.scalar_step or 1.0

    def setup(self, originals):
        self.xform = self.perturbation_params.xform_class(shape=originals.shape)
        self.initialized = True

    @initialized
    def perturbation_norm(self):
        return self.xform.norm(lp=self.lp_style)

    @initialized
    def constrain_params(self, x=None):
        # Do lp projections
        if isinstance(self.lp_style, int) or self.lp_style == 'inf':
            self.xform.project_params(self.lp_style, self.lp_bound)

    @initialized
    def update_params(self, step_fxn):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        assert params.grad.data is not None
        self.add_to_params(step_fxn(params.grad.data) * self.scalar_step)


    @initialized
    def add_to_params(self, grad_data):
        """ Assumes only one parameters object in the Spatial Transform """
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        params.data.add_(grad_data)

    @initialized
    def random_init(self):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        param = param_list[0]
        random_perturb = utils.random.random_from_lp_ball(param.data,
                                                          self.lp_style,
                                                          self.lp_bound)

        param.data.add_(self.xform.identity_params(self.xform.img_shape) +
                        random_perturb - self.xform.xform_params.data)



    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.constrain_params()
        return self.xform.forward(x)




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
                 global_parameters=PerturbationParameters(pad=10)):
        """ Initializes a sequence of adversarial perturbation layers
        ARGS:
            originals : NxCxHxW tensor - original images we create adversarial
                        perturbations for
            perturbation_sequence : ThreatModel[]  -
                list of ThreatModel objects
            total_parameters : PerturbationParameters - global parameters to
                               use. These contain things like how to norm this
                               sequence, how to constrain this sequence, etc
         """
        super(SequentialPerturbation, self).__init__((perturbation_sequence,
                                                      global_parameters))

        self.pipeline = []
        for layer_no, threat_model in enumerate(perturbation_sequence):
            assert isinstance(threat_model, ThreatModel)
            layer = threat_model()

            self.pipeline.append(layer)
            self.add_module('layer_%02d' % layer_no, layer)


        # norm: pipeline -> Scalar Variable
        self.norm = global_parameters.norm

        # padding with black is useful to not throw information away during
        # sequential steps
        self.pad = nn.ConstantPad2d(global_parameters.pad or 0, 0)
        self.unpad = nn.ConstantPad2d(-1 * (global_parameters.pad or 0), 0)


    def setup(self, x):
        x = self.pad(x)
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
        x = self.pad(x)
        for layer in self.pipeline:
            layer.make_valid_image(x)
            x = layer(x)


    @initialized
    def constrain_params(self):
        # Need to do some sort of crazy projection operator for general things
        # For now, let's just constrain each thing in sequence

        for layer in self.pipeline:
            layer.constrain_params()

    @initialized
    def update_params(self, step_fxn):
        for layer in self.pipeline:
            layer.update_params(step_fxn)


    def forward(self, x, layer_slice=None):
        """ Layer slice here is either an int or a tuple
        If int, we run forward only the first layer_slice layers
        If tuple, we start at the

        """

        # Blocks to handle only particular layer slices (debugging)
        if layer_slice is None:
            pipeline_iter = iter(self.pipeline)
        elif isinstance(layer_slice, int):
            pipeline_iter = iter(self.pipeline[:layer_slice])
        elif isinstance(layer_slice, tuple):
            pipeline_iter = iter(self.pipeline[layer_slice[0]: layer_slice[1]])
        # End block to handle particular layer slices

        # Handle padding
        original_hw = x.shape[-2:]
        if not self.initialized:
            self.setup(x)

        self.constrain_params()
        self.make_valid_image(x)

        x = self.pad(x)
        for layer in pipeline_iter:
            x = layer(x)
        return self.unpad(x)


    @initialized
    def random_init(self):
        for layer in self.pipeline:
            layer.random_init()

    @initialized
    def attach_originals(self, originals):
        for layer in self.pipeline:
            layer.attach_originals(originals)





