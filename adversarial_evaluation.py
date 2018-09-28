""" Code to better evaluate the efficacy of our attacks/models.
    AdversarialEvaluation class contains two main things:
    1) evaluate_ensemble: convenient wrapper to try an ensemble of
                          attacks on a single pretrained model and
                          output the average accuracy of each at the
                          end (compared to the ground set too)
    2) full_attack function: function that attacks each example in a DataLoader
                             and outputs all attacked examples to a numpy file
                             [USEFUL FOR GENERATING FILES NEEDED BY MADRY
                              CHALLENGE]
"""



import torch
from torch.autograd import Variable
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import custom_lpips.custom_dist_model as dm
import os
import config
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim
import math

###########################################################################
#                                                                         #
#                               EVALUATION RESULT OBJECT                  #
#                                                                         #
###########################################################################

class EvaluationResult(object):
    """ Stores results of adversarial evaluations, will be used in the
        output of AdversarialEvaluation
    """

    def __init__(self, attack_params, classifier_net, normalizer, to_eval=None,
                 use_gpu=False):
        """ to_eval is a dict of {str : toEval methods}.
        """
        self.attack_params = attack_params
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        self.use_gpu = use_gpu

        # First map shorthand strings to methods
        shorthand_evals = {'top1': self.top1_accuracy,
                           'avg_successful_lpips': self.avg_successful_lpips,
                           'avg_successful_ssim': self.avg_successful_ssim,
                           'stash_perturbations': self.stash_perturbations}
        if to_eval is None:
            to_eval = {'top1': 'top1'}

        for key, val in list(to_eval.items()):
            if val in shorthand_evals:
                to_eval[key] = shorthand_evals[val]
        self.to_eval = to_eval
        self.results = {k: None for k in self.to_eval}
        self.params = {k: None for k in self.to_eval}

    def set_gpu(self, use_gpu):
        self.attack_params.set_gpu(use_gpu)

    def eval(self, examples, labels):
        attack_out = self.attack_params.attack(examples, labels)

        for k, v in self.to_eval.items():
            v(k, attack_out, examples, labels)


    def _get_successful_attacks(self, attack_out):
        ''' Gets the (successful, corresponding-original) attacks '''
        perturbation = attack_out[4]
        return perturbation.collect_successful(self.classifier_net,
                                               self.normalizer)




    def top1_accuracy(self, eval_label, attack_out, ground_examples,
                      labels):

        ######################################################################
        #  First set up evaluation result if doesn't exist:                  #
        ######################################################################
        if self.results[eval_label] is None:
            self.results[eval_label] = utils.AverageMeter()

        result = self.results[eval_label]

        ######################################################################
        #  Computes the top 1 accuracy and updates the averageMeter          #
        ######################################################################
        attack_examples = utils.safe_var(attack_out[0])
        pre_adv_labels = utils.safe_var(attack_out[1])
        num_examples = float(attack_examples.shape[0])

        attack_accuracy_int = self.attack_params.eval_attack_only(
                                                attack_examples,
                                                pre_adv_labels, topk=1)
        result.update(attack_accuracy_int / num_examples, n=int(num_examples))


    def avg_successful_lpips(self, eval_label, attack_out, ground_examples,
                             labels):
        ######################################################################
        #  First set up evaluation result if doesn't exist:                  #
        ######################################################################
        if self.results[eval_label] is None:
            self.results[eval_label] = utils.AverageMeter()
            self.dist_model = dm.DistModel(net='alex', use_gpu=self.use_gpu)

        result = self.results[eval_label]

        if self.params[eval_label] is None:
            dist_model = dm.DistModel(net='alex', use_gpu=self.use_gpu)
            self.params[eval_label] = {'dist_model': dist_model}

        dist_model = self.params[eval_label]['dist_model']

        ######################################################################
        #  Compute which attacks were successful                             #
        ######################################################################
        successful_pert, successful_orig = self._get_successful_attacks(
                                                                     attack_out)

        if successful_pert is None or successful_pert.numel() == 0:
            return

        successful_pert = Variable(successful_pert)
        successful_orig = Variable(successful_orig)
        num_successful = successful_pert.shape[0]
        xform = lambda im: im * 2.0 - 1.0

        lpips_dist = self.dist_model.forward_var(xform(successful_pert),
                                                 xform(successful_orig))
        avg_lpips_dist = float(torch.mean(lpips_dist))

        result.update(avg_lpips_dist, n=num_successful)

    def avg_successful_ssim(self, eval_label, attack_out, ground_examples,
                            labels):
        # We actually compute (1-ssim) to match better with notion of a 'metric'
        ######################################################################
        #  First set up evaluation result if doesn't exist:                  #
        ######################################################################
        if self.results[eval_label] is None:
            self.results[eval_label] = utils.AverageMeter()

        ######################################################################
        #  Compute which attacks were successful                             #
        ######################################################################
        successful_pert, successful_orig = self._get_successful_attacks(
                                                                     attack_out)
        if successful_pert is None or successful_pert.numel() == 0:
            return


        successful_pert = Variable(successful_pert)
        successful_orig = Variable(successful_orig)

        count = 0
        runsum = 0
        for og, adv in zip(successful_orig, successful_pert):
            count += 1
            runsum += ssim(og.transpose(0, 2).cpu().numpy(),
                           adv.transpose(0, 2).cpu().numpy(), multichannel=True)


        avg_minus_ssim = 1 - (runsum / float(count))
        result.update(avg_minus_ssim, n=num_successful)


    def stash_perturbations(self, eval_label, attack_out, ground_examples,
                            labels):
        """ This will store the perturbations.
           (TODO: make these tensors and store on CPU)
        """

        ######################################################################
        #   First set up evaluation result if it doesn't exist               #
        ######################################################################
        if self.results[eval_label] is None:
            self.results[eval_label] = []

        result = self.results[eval_label]
        perturbation_obj = attack_out[4]
        result.append(perturbation_obj)





class IdentityEvaluation(EvaluationResult):
    """ Subclass of evaluation result that just computes top1 accuracy for the
        ground truths (attack perturbation is the identity)
    """
    def __init__(self, classifier_net, normalizer, use_gpu=False):
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        self.use_gpu = use_gpu

        self.results = {'top1': utils.AverageMeter()}

    def set_gpu(self, use_gpu):
        pass

    def eval(self, examples, labels):
        assert self.results.keys() == ['top1']
        ground_avg = self.results['top1']
        ground_output = self.classifier_net(self.normalizer(Variable(examples)))
        minibatch = float(examples.shape[0])

        ground_accuracy_int = utils.accuracy_int(ground_output,
                                                Variable(labels), topk=1)
        ground_avg.update(ground_accuracy_int / minibatch,
                          n=int(minibatch))





############################################################################
#                                                                          #
#                                   EVALUATION OBJECT                      #
#                                                                          #
############################################################################

class AdversarialEvaluation(object):
    """ Wrapper for evaluation of NN's against adversarial examples
    """

    def __init__(self, classifier_net, normalizer, use_gpu=False):
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        self.use_gpu = use_gpu


    def evaluate_ensemble(self, data_loader, attack_ensemble,
                          skip_ground=False, verbose=True,
                          num_minibatches=None):
        """ Runs evaluation against attacks generated by attack ensemble over
            the entire training set
        ARGS:
            data_loader : torch.utils.data.DataLoader - object that loads the
                          evaluation data
            attack_ensemble : dict {string -> EvaluationResult}
                             is a dict of attacks that we want to make.
                             None of the strings can be 'ground'
            skip_ground : bool - if True we don't evaluate the no-attack case
            verbose : bool - if True, we print things
            num_minibatches: int - if not None, we only validate on a fixed
                                   number of minibatches
        RETURNS:
            a dict same keys as attack_ensemble, as well as the key 'ground'.
            The values are utils.AverageMeter objects
        """

        ######################################################################
        #   Setup input validations                                          #
        ######################################################################

        self.classifier_net.eval()
        assert isinstance(data_loader, torch.utils.data.DataLoader)


        if attack_ensemble is None:
            attack_ensemble = {}

        if not skip_ground:
            assert 'ground' not in attack_ensemble
            # Build ground result
            ground_result = IdentityEvaluation(self.classifier_net,
                                               self.normalizer,
                                               use_gpu=self.use_gpu)
            attack_ensemble['ground'] = ground_result

        # Do GPU checks
        utils.cuda_assert(self.use_gpu)
        if self.use_gpu:
            self.classifier_net.cuda()

        for eval_result in attack_ensemble.values():
            eval_result.set_gpu(self.use_gpu)


        ######################################################################
        #   Loop through validation set and attack efficacy                  #
        ######################################################################

        for i, data in enumerate(data_loader, 0):
            if num_minibatches is not None and i >= num_minibatches:
                break
            if verbose:
                print "Starting minibatch %s..." % i


            inputs, labels = data
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            for k, result in attack_ensemble.items():
                if verbose:
                    print "\t (mb: %s) evaluating %s..." % (i, k)
                result.eval(inputs, labels)

        return attack_ensemble



    def full_attack(self, data_loader, attack_parameters,
                    output_filename, num_minibatches=None,
                    continue_attack=True, checkpoint_minibatch=10,
                    verbose=True, save_xform=img_utils.nhwc255_xform):

        """ Builds an attack on the data and outputs the resulting attacked
            images into a .numpy file
        ARGS:
            data_loader : torch.utils.data.DataLoader - object that loads the
                          evaluation data.
                          NOTE: for Madry challenge this shouldn't be shuffled
            attack_parameters : AdversarialAttackParameters object - wrapper to
                                contain the attack
            output_filename : string - name of the file we want to output.
                              should just be the base name (extension is .npy)
            num_minibatches : int - if not None, we only build attacks for this
                                    many minibatches of data
            continue_attack : bool - if True, we do the following :
                              1) check if output_filename exists. If it doesn't
                                 exist, proceed to make full attack as usual.
                              2) if output_filename exists, figure out how many
                                 minibatches it went through and skip to the
                                 next minibatch in the data loader
                           This is kinda like a checkpointing system for attacks
            checkpoint_minibatch: int - how many minibatches until we checkpoint
            verbose: bool - if True, we print out which minibatch we're in out
                            of total number of minibatches
            save_xform: fxn, np.ndarray -> np.ndarray - function that
                        transforms our adv_example.data.numpy() to the form that
                        we want to store it in in the .npy output file
        RETURNS:
            numpy array of attacked examples
        """
        raise NotImplementedError("BROKEN!!!")
        ######################################################################
        #   Setup and assert things                                          #
        ######################################################################

        self.classifier_net.eval()

        # Check if loader is shuffled. print warning if random
        assert isinstance(data_loader, torch.utils.data.DataLoader)
        if isinstance(data_loader.batch_sampler.sampler,
                      torch.utils.data.sampler.RandomSampler):
            print "WARNING: data loader is shuffled!"
        total_num_minibatches = int(math.ceil(len(data_loader.dataset) /
                                              data_loader.batch_size))
        minibatch_digits = len(str(total_num_minibatches))

        # Do cuda stuff
        utils.cuda_assert(self.use_gpu)
        attack_parameters.set_gpu(self.use_gpu)
        if self.use_gpu:
            self.classifier_net.cuda()

        # Check attack is attacking everything
        assert attack_parameters.proportion_attacked == 1.0

        # handle output_file + continue_attack stuff
        assert os.path.basename(output_filename) == output_filename, \
               "Provided output_filename was %s, should have been %s" % \
               (output_filename, os.path.basename(output_filename))

        output_file = os.path.join(config.OUTPUT_IMAGE_PATH,
                                   output_filename + '.npy')

        minibatch_attacks = [] # list of 4d numpy arrays
        num_prev_minibatches = 0
        if continue_attack and len(glob.glob(output_file)) != 0:
            # load file and see how many minibatches we went through
            saved_data = np.load(output_file)
            saved_num_examples = saved_data.shape[0]
            loader_batch_size = data_loader.batch_size
            if saved_num_examples % loader_batch_size != 0:
                print "WARNING: incomplete minibatch in previously saved attack"

            minibatch_attacks.append(saved_data)
            num_prev_minibatches = saved_num_examples / loader_batch_size

        if verbose:
            def printer(num):
                print ("Minibatch %%0%dd/%s" % (minibatch_digits,
                                                   total_num_minibatches) % num)
        else:
            printer = lambda num: None


        ######################################################################
        #   Start attacking and saving                                       #
        ######################################################################
        for minibatch_num, data in enumerate(data_loader):

            # Handle skippy cases
            if minibatch_num < num_prev_minibatches: # CAREFUL ABOUT OBOEs HERE
                continue

            if num_minibatches is not None and minibatch_num >= num_minibatches:
                break

            printer(minibatch_num)

            # Load data and build minibatch of attacked images
            inputs, labels = data
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            adv_examples = attack_parameters.attack(inputs, labels)[0]

            # Convert to numpy and append to our save buffer
            adv_data = utils.safe_tensor(adv_examples).cpu().numpy()
            minibatch_attacks.append(save_xform(adv_data))

            # Perform checkpoint if necessary
            if minibatch_num > 0 and minibatch_num % checkpoint_minibatch == 0:
                minibatch_attacks = utils.checkpoint_incremental_array(
                                                 output_file, minibatch_attacks,
                                                 return_concat=True)


        return utils.checkpoint_incremental_array(output_file,
                                                  minibatch_attacks,
                                                  return_concat=True)[0]





if __name__ == '__main__':
    import interact
