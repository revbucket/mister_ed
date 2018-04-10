""" Holds the various attacks we can do """
import torch
from torch.autograd import Variable, Function
from torch import optim
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import random
import sys
import custom_lpips.custom_dist_model as dm

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
#                           Alternating PGD                                #
#                                                                          #
############################################################################
class AltPGD(LInfPGD):

    def __init__(self, classifier_net, normalizer, to_loss_fxn, fro_loss_fxn,
                 use_gpu=False):
        """ HACKETY HACK:

        For some reason, my regularization isn't working the way I want it to.
        What I'll do instead is a hokey-pokey game:
        I have two alternating losses and I try to minimize them in alternating
        iterations.

        to_loss refers to a loss that we want to MAXIMIZE (xentropy, say)
        fro_loss refers to a loss that we want to MINIMIZE (lpips, say)
        """
        super(AltPGD, self).__init__(classifier_net, normalizer,
                                     use_gpu=use_gpu)
        self.to_loss_fxn = to_loss_fxn
        self.fro_loss_fxn = fro_loss_fxn
        self.losses = [self.to_loss_fxn, self.fro_loss_fxn]


    def attack(self, examples, labels, l_inf_bound=8.0/255.0,
               step_size=1.0/255.0, fro_scale=0.9, num_iterations=20,
               random_init=True, signed=True, verbose=True):

        assert 0.0 <= l_inf_bound <= 1.0

        if not verbose:
            self.validator = lambda ex, label, iter_no: None
        else:
            self.validator = self.validation_loop


        self.classifier_net.eval()
        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        [loss.setup_attack_batch(var_examples.clone()) for loss in self.losses]

        #####################################################################
        #   Build adversarial examples                                      #
        #####################################################################

        intermed_images = var_examples

        if random_init:
            var_examples = self._random_init(var_examples, l_inf_bound)
            self.validator(var_examples, var_labels, iter_no="RANDOM")

        # start iterating
        for iter_no in xrange(num_iterations):
            # Reset gradients
            [loss.zero_grad() for loss in self.losses()]

            # to step
            to_loss = self.to_loss_fxn(intermed_images, var_labels)

            torch.autograd.backward(to_loss)

            signs = torch.sign(intermed_images.grad.data) * step_size
            clamp_inf = utils.clamp_ref(intermed_images.data + signs, examples,
                                        l_inf_bound)
            clamp_box = torch.clamp(clamp_inf, 0., 1.)

            intermed_images = Variable(clamp_box, requires_grad=True)
            [loss.zero_grad() for loss in self.losses()]

            # fro step

            fro_loss = self.fro_loss_fxn(intermed_images, var_labels)
            torch.autograd.backward(fro_loss)

            signs = (torch.sign(intermed_images.grad.data) * step_size *
                     fro_scale)
            clamp_inf = utils.clamp_ref(intermed_images.data - signs, examples,
                                        l_inf_bound)
            clamp_box = torch.clamp(clamp_inf, 0., 1.0)

            intermed_images = Variable(clamp_box, requires_grad=True)
            self.validator(intermed_images, var_labels, iter_no=iter_no)


        return intermed_images.data








############################################################################
#                                                                          #
#                           Carlini Wagner attacks                         #
#                                                                          #
############################################################################

class CWL2(AdversarialAttack):

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

        super(CWL2, self).__init__(classifier_net, normalizer, use_gpu=use_gpu)
        self.loss_fxn = loss_fxn
        self.scale_constant = scale_constant
        self.num_bin_search_steps = num_bin_search_steps
        self.num_optim_steps = num_optim_steps
        self.confidence = confidence

        if distance_metric_type == 'l2':
            # x, y should be in [0., 1.0] range
            distance_metric = lambda x, y: torch.norm(x - y, 2)
        elif distance_metric_type == 'lpips':
            # Perceptual distance is a little more involved... not defined here
            # x, y should be in [0., 1.0] range
            dist_model = dm.DistModel()
            dist_model.initialize(model='net-lin', net='alex', use_gpu=use_gpu)
            def distance_metric(x, y, dist_model=dist_model):
                xform = lambda im: im * 2.0 - 1.0
                dist = dist_model.forward(xform(x.unsqueeze(0)),
                                          xform(y.unsqueeze(0)))
                return float(dist)
        self.distance_metric = distance_metric

        self.use_gpu = use_gpu



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



    def _batch_compare(self, example_logits, targets, targeted=False):
        """ Returns a list of indices of valid adversarial examples
        ARGS:
            example_logits: Variable/Tensor (Nx#Classes) - output logits for a
                            batch of images
            targets: Variable/Tensor (N) - each element is a class index for the
                     target class for the i^th example.
            targeted: bool - if True, the 'targets' arg should be the targets
                             we want to hit. If False, 'targets' arg should be
                             the targets we do NOT want to hit
        RETURNS:
            list of indices into example_logits for which the desired confidence
            bound/targeting holds
        """
        # check if the max val is the targets
        target_vals = example_logits.gather(1, targets.view(-1, 1))
        max_vals, max_idxs = torch.max(example_logits, 1)
        max_eq_targets = torch.eq(targets, max_idxs)

        # check margins between max and target_vals
        if targeted:
            max_2_vals, _ = example_logits.kthvalue(2, dim=1)
            good_confidence = torch.gt(max_vals - self.confidence, max_2_vals)
            one_hot_indices = max_eq_targets * good_confidence
        else:
            good_confidence = torch.gt(max_vals.view(-1, 1),
                                       target_vals + self.confidence)
            one_hot_indices = ((1 - max_eq_targets.data).view(-1, 1) *
                               good_confidence.data)

        return [idx for idx, el in enumerate(one_hot_indices) if el[0] == 1]


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


        original_output = self.classifier_net.forward(
                            self.normalizer.forward(Variable(examples)))

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

        var_intermeds = Variable(tanh_examples, requires_grad=True) # VOLATILE


        targeted = target_labels is not None
        if targeted:
            var_targets = Variable(target_labels, requires_grad=False)
        else:
            var_targets = Variable(original_labels, requires_grad=False)

        ####################################################################
        #   Binary search over possible scale constants                    #
        ####################################################################
        var_scale_lo = Variable(torch.ones(num_examples).type(self._dtype) *
                       self.scale_constant)

        var_scale = Variable(torch.ones(num_examples, 1).type(self._dtype) *
                             self.scale_constant)
        var_scale_hi = Variable(torch.ones(num_examples).type(self._dtype)
                                * 16) # HARDCODED UPPER LIMIT


        for bin_search_step in xrange(self.num_bin_search_steps):

            ##############################################################
            #   Optimize with a given scale constant                     #
            ##############################################################
            if verbose:
                print "Starting binary_search_step %02d..." % bin_search_step
            prev_loss = MAXFLOAT
            optimizer = optim.Adam([var_intermeds], lr=0.0005)

            for optim_step in xrange(self.num_optim_steps):
                loss_sum = self._optimize_step(optimizer, var_intermeds,
                                               var_targets, var_scale,
                                               targeted=targeted)

                if loss_sum > prev_loss * 0.9999:
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







