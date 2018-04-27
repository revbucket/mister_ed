""" Holds the various attacks we can do """


import torch
from torch.autograd import Variable, Function
from torch import optim
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import random
import sys
import custom_lpips.custom_dist_model as dm
import loss_functions as lf
import spatial_transformers as st
import torch.nn as nn

MAXFLOAT = 1e20

class AdversarialAttack(object):
    """ Wrapper for adversarial attacks. Is helpful for when subsidiary methods
        are needed.
    """

    def __init__(self, classifier_net, normalizer, use_gpu=False):
        self.classifier_net = classifier_net
        self.normalizer = normalizer or utils.IdentityNormalize()
        self.use_gpu = use_gpu
        self.validator = lambda *args: None

    @property
    def _dtype(self):
        return torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

    def setup(self):
        self.classifier_net.eval()
        self.normalizer.differentiable_call()


    def eval(self, ground_examples, adversarials, labels, topk=1):
        """ Evaluates how good the adversarial examples are
        ARGS:
            ground_truths: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (NxCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
        RETURNS:
            tuple of (% of correctly classified original examples,
                      % of correctly classified adversarial examples)
        """
        ground_examples = utils.safe_var(ground_examples)
        adversarials = utils.safe_var(adversarials)
        labels = utils.safe_var(labels)

        normed_ground = self.normalizer.forward(ground_examples)
        ground_output = self.classifier_net.forward(normed_ground)

        normed_advs = self.normalizer.forward(adversarials)
        adv_output = self.classifier_net.forward(normed_advs)

        start_prec = utils.accuracy(ground_output.data, labels.data,
                                    topk=(topk,))
        adv_prec = utils.accuracy(adv_output.data, labels.data,
                                  topk=(topk,))

        return start_prec[0][0], adv_prec[0][0]


    def eval_attack_only(self, adversarials, labels, topk=1):
        """ Outputs the accuracy of the adv_inputs only
        ARGS:
            adv_inputs: Variable NxCxHxW - examples after we did adversarial
                                           perturbation
            labels: Variable (longtensor N) - correct labels of classification
                                              output
            topk: int - criterion for 'correct' classification
        RETURNS:
            (int) number of correctly classified examples
        """

        adversarials = utils.safe_var(adversarials)
        labels = utils.safe_var(labels)
        normed_advs = self.normalizer.forward(adversarials)

        adv_output = self.classifier_net.forward(normed_advs)
        return utils.accuracy_int(adv_output, labels, topk=topk)



    def print_eval_str(self, ground_examples, adversarials, labels, topk=1):
        """ Prints how good this adversarial attack is
            (explicitly prints out %CorrectlyClassified(ground_examples)
            vs %CorrectlyClassified(adversarials)

        ARGS:
            ground_truths: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (NxCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
        RETURNS:
            None, prints some stuff though
        """

        og, adv = self.eval(ground_examples, adversarials, labels, topk=topk)
        print "Went from %s correct to %s correct" % (og, adv)



    def validation_loop(self, examples, labels, iter_no=None):
        """ Prints out validation values interim for use in iterative techniques
        ARGS:
            new_examples: Variable (NxCxHxW) - [0.0, 1.0] images to be
                          classified and compared against labels
            labels: Variable (longTensor
            N) - correct labels for indices of
                             examples
            iter_no: String - an extra thing for prettier prints
        RETURNS:
            None
        """
        normed_input = self.normalizer.forward(examples)
        new_output = self.classifier_net.forward(normed_input)
        new_prec = utils.accuracy(new_output.data, labels.data, topk=(1,))
        print_str = ""
        if isinstance(iter_no, int):
            print_str += "(iteration %02d): " % iter_no
        elif isinstance(iter_no, basestring):
            print_str += "(%s): " % iter_no
        else:
            pass

        print_str += " %s correct" % new_prec[0][0]

        print print_str


##############################################################################
#                                                                            #
#                        Uniform Random Method (URM)                         #
#                                                                            #
##############################################################################

class URM(AdversarialAttack):
    def __init__(self, classifier_net, normalizer, loss_fxn, use_gpu=False,
                 lp_norm='linf'):
        super(URM, self).__init__(classifier_net, normalizer, use_gpu=use_gpu)

        assert lp_norm in ['l2', 'linf']

        if lp_norm == 'l2':
            raise NotImplementedError

        self.lp_norm = lp_norm

        assert isinstance(loss_fxn, lf.IncorrectIndicator)
        self.loss_fxn = loss_fxn


    @classmethod
    def _random_linf_perturbation(cls, examples_like, linf_tensor):
        """ Returns an object of the same type/shape as examples_like that
            holds a uniformly random perturbation in the l_infinity box of
            l_inf_tensor[i].

            NOTE THAT THIS JUST RETURNS THE PERTUBATION AND NOT
            PERTUBATION + EXAMPLES_LIKE
        ARGS:
            examples_like : Tensor or Variable (NxCxHxW) -
            linf_tensor : LongTensor (N) - per-example l_inf bounds we have,
        RETURNS:
            NxCxHxW object with same type/storage_device as examples_like
        """
        num_examples = examples_like.shape[0]



        linf_expanded = linf_tensor.view(num_examples,
                                         *[1] * (examples_like.dim() - 1))

        is_var = isinstance(examples_like, Variable)

        random_tensor = torch.sign(torch.rand(*examples_like.shape) - 0.5) * linf_expanded

        # random_tensor = (torch.rand(*examples_like.shape) * linf_expanded * 2 -
        #                torch.ones(*examples_like.shape) * linf_expanded)


        if random_tensor.is_cuda != examples_like.is_cuda:
            xform = lambda t: t.cuda() if examples_like.is_cuda else t.cpu()
            random_tensor = xform(random_tensor)

        if is_var:
            return Variable(random_tensor)
        else:
            return random_tensor



    def attack(self, examples, labels, lp_bound, num_tries=100, verbose=True):
        """ For a given minibatch of examples, uniformly randomly generates
            num_tries uniform elements from the lp ball of size lp_bound.
            The minimum distance incorrectly classified object is kept,
            otherwise the original is returned
        ARGS:
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)

            lp_bound: float - how far we're allowed to guess in lp-distance
            num_tries : how many random perturbations we try per example
        RETURNS:
            {'min_dist': ..., 'adv_ex': tensor output}
        """

        # NOTE: THIS IS SUPER NAIVE AND WE CAN DO UNIFORM RANDOM WAAAAY BETTER


        num_examples = examples.shape[0]
        dim = examples.dim()
        expand_to_dim = lambda t: t.view(num_examples, *([1] * dim))

        var_examples = Variable(examples) # no grads needed
        var_labels = Variable(labels)

        if self.lp_norm == 'linf':
            random_guesser = self._random_linf_perturbation
            lp_type = 'inf'
        else:
            lp_type = int(self.lp_norm[1:])
            raise NotImplementedError



        lp_vec = torch.ones(num_examples).type(self._dtype) * lp_bound
        outputs = {'best_dist': torch.ones(num_examples).type(self._dtype) *\
                                MAXFLOAT,
                   'best_adv_images': examples.clone(),
                   'original_images': examples.clone()}

        ######################################################################
        #   Loop through each try                                            #
        ######################################################################

        for try_no in xrange(num_tries):
            if verbose and try_no % (num_tries / 10) == 0 and try_no > 1:
                print "Completed %03d random guesses..." % try_no

            # get random perturbation
            random_guess = var_examples + random_guesser(var_examples,
                                                         lp_vec)
            random_guess = utils.clip_0_1(random_guess)

            loss = self.loss_fxn.forward(random_guess, var_labels,
                                         return_type='vector')

            loss = loss.type(self._dtype)
            converse_loss = 1 - loss

            # get distances per example


            batchwise_norms = utils.batchwise_norm(random_guess - var_examples,
                                                   lp_type, dim=0)

            # Reflect this iteration in outputs and lp_tensor

            # build incorrect index vector
            # comp_1[i] = batchwise_norm[i] if i is adversarial, MAXFLOAT o.w.
            comp_1 = (loss * batchwise_norms + converse_loss * (MAXFLOAT)).data

            # figure out which random guesses to keep
            to_keep = (comp_1 < outputs['best_dist']).type(self._dtype)
            to_keep = expand_to_dim(to_keep)
            to_keep_converse = 1 - to_keep

            # compute new best_dists and best_adv_images
            new_bests = (random_guess.data * to_keep +
                         outputs['best_adv_images'] * to_keep_converse)
            new_best_dists = torch.min(comp_1, outputs['best_dist'])

            outputs['best_dist'] = new_best_dists
            outputs['best_adv_images'] = new_bests

        if verbose:
            num_successful = len([_ for _ in outputs['best_dist']
                                  if _ < MAXFLOAT])
            print "\n Ending attack"
            print "Successful attacks for %03d/%03d examples in CONTINUOUS" %\
                   (num_successful, num_examples)

        return outputs









##############################################################################
#                                                                            #
#                         Fast Gradient Sign Method (FGSM)                   #
#                                                                            #
##############################################################################

class FGSM(AdversarialAttack):
    def __init__(self, classifier_net, normalizer, loss_fxn, use_gpu=False):
        super(FGSM, self).__init__(classifier_net, normalizer, use_gpu=use_gpu)
        self.loss_fxn = loss_fxn

    def attack(self, examples, labels, l_inf_bound=0.05, verbose=True):

        """ Builds FGSM examples for the given examples with l_inf bound
        ARGS:
            classifier: Pytorch NN
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)
            l_inf_bound: float between 0.0 and 1.0 of maximum l_infinity bound
            normalizer: DifferentiableNormalize object to prep objects into
                        classifier
            evaluate: boolean, if True will validation results
            loss_fxn:  RegularizedLoss object - partially applied loss fxn that
                         takes [0.0, 1.0] image Variables and labels and outputs
                         a scalar loss variable. Also has a zero_grad method
        RETURNS:
            adv_examples: NxCxWxH tensor with adversarial examples

        """

        assert 0 < l_inf_bound < 1.0
        self.classifier_net.eval() # ALWAYS EVAL FOR BUILDING ADV EXAMPLES

        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        ######################################################################
        #   Build adversarial examples                                       #
        ######################################################################

        # Fix the 'reference' images for the loss function
        self.loss_fxn.setup_attack_batch(var_examples)

        # take gradients and step
        loss = self.loss_fxn.forward(var_examples, var_labels)
        torch.autograd.backward(loss)

        # add adversarial noise and clamp to 0.0, 1.0 range
        signs = l_inf_bound * torch.sign(var_examples.grad.data)
        adversarial_examples = torch.clamp(examples + signs, 0, 1)


        # output tensor with the data
        self.loss_fxn.cleanup_attack_batch()
        return adversarial_examples



##############################################################################
#                                                                            #
#                               Basic Iterative Method (BIM)                 #
#                                                                            #
##############################################################################

class BIM(AdversarialAttack):
    def __init__(self, classifier_net, normalizer, loss_fxn, use_gpu=False):
        super(BIM, self).__init__(classifier_net, normalizer, use_gpu=use_gpu)
        self.loss_fxn = loss_fxn

    def attack(self, examples, labels, l_inf_bound=0.05, step_size=1 / 255.,
               num_iterations=None, verbose=True):
        """ Builds BIM examples for the given examples with l_inf bound, and
            given step_size
        ARGS:
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)
            l_inf_bound: float between 0.0 and 1.0 of maximum l_infinity bound
            step_size: how much to step each time. 1/256 is a nice value here
            normalizer: DifferentiableNormalize object to prep objects into
                        classifier
            loss_fxn:  RegularizedLoss object - partially applied loss fxn that
                         takes [0.0, 1.0] image Variables and labels and outputs
                         a scalar loss variable. Also has a zero_grad method
            num_iterations: int - if not None should be an integer number of
                                  iterations. Defaults to the default from the
                                  original paper
        RETURNS:
            Nxcxwxh tensor with adversarial examples,
        """

        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        assert 0 < l_inf_bound < 1.0

        # use original paper to figure out num_iterations
        # https://arxiv.org/pdf/1607.02533.pdf
        num_iterations = num_iterations or int(min([l_inf_bound * 255 + 4,
                                                 l_inf_bound * 255 * 1.25]) + 1)

        if not verbose:
            self.validator = lambda ex, label, iter_no: None
        else:
            self.validator = self.validation_loop

        self.classifier_net.eval() # ALWAYS EVAL FOR BUILDING ADV EXAMPLES
        var_examples = Variable(examples,requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        # Fix the 'reference' images for the loss function
        self.loss_fxn.setup_attack_batch(var_examples.clone())
        ######################################################################
        #   Build adversarial examples                                       #
        ######################################################################

        intermed_images = var_examples
        self.validator(intermed_images, var_labels, iter_no="START")

        # Start iterating...
        for iter_no in xrange(num_iterations):

            # Reset gradients, then take another gradient
            self.loss_fxn.zero_grad()
            loss = self.loss_fxn.forward(intermed_images, var_labels)

            torch.autograd.backward(loss)

            # Take a step and then clamp
            signs = torch.sign(intermed_images.grad.data) * step_size
            clamp_inf = utils.clamp_ref(intermed_images.data + signs, examples,
                                       l_inf_bound)
            clamp_box = torch.clamp(clamp_inf, 0., 1.)

            # Setup for next
            intermed_images = Variable(clamp_box, requires_grad=True)
            self.validator(intermed_images, var_labels, iter_no=iter_no)


        return intermed_images.data



##############################################################################
#                                                                            #
#                           Projected Gradient Descent (PGD)                 #
#                                                                            #
##############################################################################

class LInfPGD(AdversarialAttack):
    def __init__(self, classifier_net, normalizer, loss_fxn, use_gpu=False):
        super(LInfPGD, self).__init__(classifier_net, normalizer,
                                      use_gpu=use_gpu)
        self.loss_fxn = loss_fxn

    def _do_iteration(self, intermed_images, var_labels, signed, step_size,
                      l_inf_bound, reference_var):

        self.loss_fxn.zero_grad()
        loss = self.loss_fxn.forward(intermed_images, var_labels)
        torch.autograd.backward(loss)

        # Take a step and 'project'
        if signed:
            perturbation = torch.sign(intermed_images.grad.data) * step_size
        else:
            perturbation = intermed_images.grad.data * step_size

        clamp_inf = utils.clamp_ref(intermed_images.data + perturbation,
                                   reference_var.data, l_inf_bound)
        clamp_box = torch.clamp(clamp_inf, 0., 1.)
        intermed_images = Variable(clamp_box, requires_grad=True)
        return Variable(intermed_images.data, requires_grad=True)


    def _random_init(self, var_examples, l_inf_bound):
        """ Returns a tensor with a random perturbation within the l_inf
            bound of the original
        ARGS:
            examples : NxCxHxW Variable - original images
            l_inf_bound : float - how much we can perturb each pixel by
        RETURNS:
            NxCxHxW Varialbe on the same device as the original,
        """
        rand_noise = (torch.rand(*var_examples.shape) * l_inf_bound * 2 -
                      torch.ones(*var_examples.shape) * l_inf_bound)
        rand_noise = rand_noise.type(self._dtype)

        clipped_init = torch.clamp(rand_noise + var_examples.data,
                                   0.0, 1.0)

        var_examples = Variable(clipped_init, requires_grad=True)
        return var_examples


    def attack(self, examples, labels, l_inf_bound=0.05, step_size=1/255.0,
               num_iterations=None, random_init=False, signed=True,
               verbose=True):
        """ Builds PGD examples for the given examples with l_inf bound and
            given step size. Is almost identical to the BIM attack, except
            we take steps that are proportional to gradient value instead of
            just their sign
        ARGS:
            examples: NxCxHxW tensor - for N examples, is NOT NORMALIZED
                      (i.e., all values are in between 0.0 and 1.0)
            labels: N longTensor - single dimension tensor with labels of
                    examples (in same order as examples)
            l_inf_bound : float - how much we're allowed to perturb each pixel
                          (relative to the 0.0, 1.0 range)
            step_size : float - how much of a step we take each iteration
            num_iterations: int - how many iterations we take
            random_init : bool - if True, we randomly pick a point in the
                               l-inf epsilon ball around each example
            signed : bool - if True, each step is
                            adversarial = adversarial + sign(grad)
                            [this is the form that madry et al use]
                            if False, each step is
                            adversarial = adversarial + grad
        RETURNS:
            NxCxHxW tensor with adversarial examples
        """

        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        assert 0 < l_inf_bound < 1.0

        self.classifier_net.eval() # ALWAYS EVAL FOR BUILDING ADV EXAMPLES

        if not verbose:
            self.validator = lambda ex, label, iter_no: None
        else:
            self.validator = self.validation_loop

        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        reference_var = Variable(examples.clone(), requires_grad=False)

        self.loss_fxn.setup_attack_batch(reference_var)

        ##################################################################
        #   Build adversarial examples                                   #
        ##################################################################

        self.validator(var_examples, var_labels, iter_no="START")

        # random initialization if necessary
        if random_init:
            var_examples = self._random_init(var_examples, l_inf_bound)
            self.validator(var_examples, var_labels, iter_no="RANDOM")

        # Start iterating...
        for iter_no in xrange(num_iterations):
            # Reset gradients, then take another gradient
            var_examples = self._do_iteration(var_examples, var_labels, signed,
                                              step_size, l_inf_bound,
                                              reference_var)
            self.validator(var_examples, var_labels, iter_no=iter_no)
        return var_examples.data


############################################################################
#                                                                          #
#                           Carlini Wagner attacks                         #
#                                                                          #
############################################################################

class CW(AdversarialAttack):

    def __init__(self, classifier_net, normalizer, loss_fxn,
                 scale_constant, num_bin_search_steps=5, num_optim_steps=1000,
                 distance_metric_type='l2', confidence=0.0, use_gpu=False):
        """ Most effective implementation of Carlini/Wagner's L2 attack as
            outlined in their paper: https://arxiv.org/pdf/1608.04644.pdf
            Reference Implementations:
                - https://github.com/rwightman/pytorch-nips2017-attack-example
                - https://github.com/tensorflow/cleverhans/
        ARGS:
            classifier_net: Pytorch NN
            normalizer: DifferentiableNormalize object to prep objects into
                        classifier
            loss_fxn:  RegularizedLoss object - partially applied loss fxn that
                         [0.0, 1.0] image Variables and labels and outputs a
                         scalar loss variable. Also has a zero_grad method

            REFACTOR WITH BETTER SPECS!
            distance_metric: function that takes two arguments and returns the
                             'distance' between two arguments. Defaults to
                             l2 distance between [0.0, 1.0] images
        """

        super(CW, self).__init__(classifier_net, normalizer, use_gpu=use_gpu)
        self.loss_fxn = loss_fxn
        self.scale_constant = scale_constant
        self.num_bin_search_steps = num_bin_search_steps
        self.num_optim_steps = num_optim_steps
        self.confidence = confidence
        self.use_gpu = use_gpu

        if distance_metric_type == 'l2':
            # x, y should be in [0., 1.0] range
            distance_metric = lambda x, y: torch.norm(x - y, 2)
        elif distance_metric_type == 'linf':
            # x, y should be in [0.0, 1.0] range
            distance_metric = lambda x, y: torch.max(torch.abs(x - y))

        elif distance_metric_type == 'lpips':
            # Perceptual distance is a little more involved... not defined here
            # x, y should be in [0., 1.0] range
            dist_model = dm.DistModel(net='alex', use_gpu=self.use_gpu)
            def distance_metric(x, y, dist_model=dist_model):
                xform = lambda im: im * 2.0 - 1.0
                dist = dist_model.forward_var(Variable(xform(x.unsqueeze(0))),
                                              Variable(xform(y.unsqueeze(0))))
                return float(dist)
        self.distance_metric = distance_metric





    @classmethod
    def filter_outputs(cls, output, cutoff_metric=None):
        """ Takes in the dict that is the output of attack and returns a
            tensor that is same shape as output['best_adv_images'] but for each
            image is either the adv image (or if the best distance is higher
            than the supplied cutoff_metric) is the original image
        ARGS:
            output : dict - output of an instance of this.attack(...)
            cutoff_metric: float - if not None is the metric we force cutoffs
                                   for. An attack that has a best distance
                                   higher than this cutoff is not a valid attack
        RETURNS:
            tensor of shape (NxCxHxW) that is combination of
            output['best_adv_images'] and output['original_images']
        """

        output_examples = output['best_adv_images'].clone()
        if cutoff_metric is None:
            return output_examples

        for idx, dist in enumerate(output['best_dist']):
            if dist > cutoff_metric:
                output_examples[idx] = output['original_images'][idx]

        return output_examples






    def _tanh_transform(self, examples, forward=True):
        """ Converts [0., 1.] examples -> tanh space examples and vice versa
        ARGS:
            examples: Variable/Tensor (NxCxHxW) - either var OR tensor to be
                      transformed
            forward: bool - if True we convert [0., 1.] examples to tanh space
                            and if False we convert tanh -> [0., 1.]
        RETURNS:
            output: <same type as examples>
        """
        if forward:
            # scale to -1, 1
            temp = (examples * 2 - 1) * (1 - 1e-6)
            return torch.log((1 + temp) / (1 - temp)) / 2.
        else:
            return (torch.tanh(examples) + 1) / 2.



    def _optimize_step(self, optimizer, intermed_adv_ex,
                       var_target, var_scale, targeted=False):
        """ Does one step of optimization """

        # Convert back into [0, 1] space and then into classifier space
        optimizer.zero_grad()
        intermed_images = self._tanh_transform(intermed_adv_ex, forward=False)

        # compute loss
        loss = self.loss_fxn.forward(intermed_images, var_target,
                                         scale_constant=var_scale,
                                         targeted=targeted)

        # backprop one step
        if torch.numel(loss) > 1:
            loss = loss.sum()

        loss.backward()
        optimizer.step()

        # return a loss 'average' to determine if we need to stop early
        return loss.data[0]


    def attack(self, examples, original_labels, target_labels=None,
               verbose=True):

        if self.use_gpu:
            self.classifier_net.cuda()
            examples = examples.cuda()
            original_labels = original_labels.cuda()


        self.classifier_net.eval() # ALWAYS EVAL FOR BUILDING ADV EXAMPLES

        self.loss_fxn.setup_attack_batch(Variable(examples,
                                                  requires_grad=False))


        num_examples = examples.shape[0]
        best_results = {'best_dist': torch.ones(num_examples) * MAXFLOAT, # NOT SQUARED
                        'best_adv_images': examples.clone(), # [0,1] CWH TENSOR
                        'original_images': examples.clone()
                       }

        ###################################################################
        #   First transform the input to tanh space                       #
        ###################################################################

        # [0,1] images -> [-inf, inf] space through tanh transform
        tanh_examples = self._tanh_transform(examples, forward=True)

        var_intermeds = Variable(tanh_examples, requires_grad=True)


        targeted = target_labels is not None
        if targeted:
            var_targets = Variable(target_labels, requires_grad=False)
        else:
            var_targets = Variable(original_labels, requires_grad=False)

        ####################################################################
        #   Binary search over possible scale constants                    #
        ####################################################################
        var_scale_lo = Variable(torch.ones(num_examples).type(self._dtype) *
                       self.scale_constant).squeeze()

        var_scale = Variable(torch.ones(num_examples, 1).type(self._dtype) *
                             self.scale_constant).squeeze()
        var_scale_hi = Variable(torch.ones(num_examples).type(self._dtype)
                                * 16).squeeze() # HARDCODED UPPER LIMIT
        for bin_search_step in xrange(self.num_bin_search_steps):

            ##############################################################
            #   Optimize with a given scale constant                     #
            ##############################################################
            if verbose:
                print "Starting binary_search_step %02d..." % bin_search_step
            prev_loss = MAXFLOAT
            optimizer = optim.Adam([var_intermeds], lr=0.0005)

            for optim_step in xrange(self.num_optim_steps):


                if verbose and optim_step > 0 and optim_step % 25 == 0:
                    print "Optim search: %s, %s" % (optim_step, prev_loss)
                loss_sum = self._optimize_step(optimizer, var_intermeds,
                                               var_targets, var_scale,
                                               targeted=targeted)

                if loss_sum + 1e-10 > prev_loss * 0.9999:
                    if verbose:
                        print ("...stopping early on binary_search_step %02d "
                               " after %03d iterations" ) % (bin_search_step,
                                                             optim_step)
                    break
                prev_loss = loss_sum
            # End inner optimize loop


            ##########################################################
            #   Update with results from optimization                #
            ##########################################################
            bin_search_output = self.classifier_net.forward(
                                self.normalizer.forward(
                                self._tanh_transform(var_intermeds,
                                                     forward=False)))

            successful_attack_idxs = set(self._batch_compare(bin_search_output,
                                                             var_targets,
                                                             targeted=targeted))

            for example_idx in xrange(num_examples):
                # if successful attack, record best-keepers
                og_example = examples[example_idx]
                adv_example = self._tanh_transform(var_intermeds.data[example_idx],
                                                   forward=False)


                # compute distance between intermeds and originals:


                dist = self.distance_metric(og_example, adv_example)


                if (example_idx in successful_attack_idxs and
                    best_results['best_dist'][example_idx] > dist):
                    best_results['best_dist'][example_idx] = dist

                    #CHECK VOLATILITY HERE
                    best_results['best_adv_images'][example_idx] = adv_example


                # do binary search adjustments
                hi = float(var_scale_hi[example_idx])
                lo = float(var_scale_lo[example_idx])
                current_scale = float(var_scale[example_idx])

                if example_idx in successful_attack_idxs:
                    var_scale_hi[example_idx] = min([hi, current_scale])
                    var_scale[example_idx] = (current_scale + lo) / 2.
                else:
                    var_scale_lo[example_idx] = max([lo, current_scale])
                    var_scale[example_idx] = (current_scale + hi) / 2
            # End update loop

        # End binary search loop
        if verbose:
            num_successful = len([_ for _ in best_results['best_dist']
                                  if _ < MAXFLOAT])
            print "\n Ending attack"
            print "Successful attacks for %03d/%03d examples in CONTINUOUS" %\
                   (num_successful, num_examples)


        self.loss_fxn.cleanup_attack_batch()
        return best_results



##############################################################################
#                                                                            #
#                           Spatial Transformation Attacks                   #
#                                                                            #
##############################################################################

"""
This is an experimental section. Here we optimize spatial transformation
parameters to maximize loss on a per-example basis. Later we can also fold in
l-infinity perturbations, but for now, let's just use STN modules

The most basic thing to do would be to do a 'PGD/FGSM^k/BIM' type attack across
a FULLY paramaterized spatial transformation network
"""


class SpatialPGDLp(AdversarialAttack):
    def __init__(self, classifier_net, normalizer, loss_fxn, lp, use_gpu=False):
        super(SpatialPGDLp, self).__init__(classifier_net, normalizer,
                                     use_gpu=use_gpu)
        self.loss_fxn = loss_fxn
        assert (isinstance(lp, int) or lp == 'inf')
        self.lp = lp


    def attack(self, examples, labels, lp_bound, num_iter, step_size=1/320.,
               signed=True, verbose=True):
        """ Builds tensors with adversarially attacked examples
        ARGS:
            examples : NxCxHxW tensor - original examples
            labels : N-length tensor - labels for the original examples
            lp_bound: float - maximum allowable Lp distance the transformation
                              can attain
            num_iter : int - number of PGD iterations we do
            step_size : float - the step size we take for signed perturbations
            signed : bool - if True, we take an L-infinity step of magnitude
                            step_size based on the signs of the gradients,
                            otherwise we do true gradient ascent
            verbose: bool - if True, we print things
        RETURNS
            adversarial_examples : NxCxHxW tensor of adversarially perturbed
                                   examples
        """


        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        if self.lp == 'inf':
            assert 0 < lp_bound < 1.0

        self.classifier_net.eval()

        if not verbose:
            self.validator = lambda ex, label, iter_no: None
        else:
            self.validator = self.validation_loop

        spatial_transformer = st.FullSpatial(shape=examples.shape)

        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)


        self.loss_fxn.setup_attack_batch(Variable(examples,
                                                  requires_grad=False))

        #####################################################################
        #   Build adversarial examples                                      #
        #####################################################################

        # iterate and modify the spatial_transformation bound
        optimizer = optim.Adam(spatial_transformer.parameters(), lr=step_size)
        for iter_no in xrange(num_iter):
            if verbose:
                print "Optim step: %03d" % iter_no

            spatial_transformer.zero_grad()

            xformed_examples = spatial_transformer(var_examples)
            normed_xformed = self.normalizer.forward(xformed_examples)
            normed_xformed_out = self.classifier_net(normed_xformed)


            loss_val = self.loss_fxn.forward(xformed_examples, var_labels,
                                             spatial=spatial_transformer)


            # if signed make steps according to signs of grads
            if signed:
                loss_val.backward()
                signed_grads = torch.sign(spatial_transformer.grid_params.grad)
                new_params = nn.Parameter((spatial_transformer.grid_params +
                                           signed_grads * step_size).data)
                spatial_transformer.grid_params = new_params
            else: # use Adam
                loss_val = loss_val * -1 # want to maximize loss
                loss_val.backward()
                optimizer.step()


            spatial_transformer.project_params(self.lp, lp_bound)
            if iter_no % 10 == 0:
                self.validator(spatial_transformer(var_examples), var_labels,
                               iter_no)


        return spatial_transformer(var_examples).data







