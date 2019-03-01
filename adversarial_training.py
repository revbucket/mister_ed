""" Contains training code for adversarial training """
from __future__ import print_function
import torch
import torchvision
import torch.cuda as cuda
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from torch.autograd import Variable

import random

import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import adversarial_attacks as attacks
import utils.checkpoints as checkpoints


##############################################################################
#                                                                            #
#                               ATTACK PARAMETERS OBJECT                     #
#                                                                            #
##############################################################################

class AdversarialAttackParameters(object):
    """ Wrapper to store an adversarial attack object as well as some extra
        parameters for how to use it in training
    """
    def __init__(self, adv_attack_obj, proportion_attacked=1.0,
                 attack_specific_params=None):
        """ Stores params for how to use adversarial attacks in training
        ARGS:
            adv_attack_obj : AdversarialAttack subclass -
                             thing that actually does the attack
            proportion_attacked: float between [0.0, 1.0] - what proportion of
                                 the minibatch we build adv examples for
            attack_specific_params: possibly None dict, but possibly dict with
                                    specific parameters for attacks

        """
        self.adv_attack_obj = adv_attack_obj
        self.proportion_attacked = proportion_attacked

        attack_specific_params = attack_specific_params or {}
        self.attack_specific_params = attack_specific_params
        self.attack_kwargs = attack_specific_params.get('attack_kwargs', {})


    def set_gpu(self, use_gpu):
        """ Propagates changes of the 'use_gpu' parameter down to the attack
        ARGS:
            use_gpu : bool - if True, the attack uses the GPU, ow it doesn't
        RETURNS:
            None
        """
        self.adv_attack_obj.use_gpu = use_gpu


    def attack(self, inputs, labels):
        """ Builds some adversarial examples given real inputs and labels
        ARGS:
            inputs : torch.Tensor (NxCxHxW) - tensor with examples needed
            labels : torch.Tensor (N) - tensor with the examples needed
        RETURNS:
            some sample of (self.proportion_attacked * N ) examples that are
            adversarial, and the corresponding NONADVERSARIAL LABELS

            output is a tuple with three tensors:
             (adv_examples, pre_adv_labels, selected_idxs, coupled )
             adv_examples: Tensor with shape (N'xCxHxW) [the perturbed outputs]
             pre_adv_labels: Tensor with shape (N') [original labels]
             selected_idxs : Tensor with shape (N') [idxs selected]
             adv_inputs : Tensor with shape (N'xCxHxW)
                          [examples used to make advs]
             perturbation: Adversarial Perturbation Object
        """
        num_elements = inputs.shape[0]

        selected_idxs = sorted(random.sample(list(range(num_elements)),
                                int(self.proportion_attacked * num_elements)))

        selected_idxs = inputs.new(selected_idxs).long()
        if selected_idxs.numel() == 0:
            return (None, None, None)

        adv_inputs = Variable(inputs.index_select(0, selected_idxs))
        pre_adv_labels = labels.index_select(0, selected_idxs)

        perturbation = self.adv_attack_obj.attack(adv_inputs.data,
                                                  pre_adv_labels,
                                                  **self.attack_kwargs)
        adv_examples = perturbation(adv_inputs)

        return (adv_examples, pre_adv_labels, selected_idxs, adv_inputs,
                perturbation)


    def eval(self, ground_inputs, adv_inputs, labels, idxs, topk=1):
        """ Outputs the accuracy of the adversarial examples

            NOTE: notice the difference between N and N' in the argument
        ARGS:
            ground_inputs: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (N'xCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
            idxs: Variable (longtensor N') - indices of ground_inputs/labels
                  used for adversarials.
        RETURNS:
            tuple of (% of correctly classified original examples,
                      % of correctly classified adversarial examples)

        """

        selected_grounds = ground_inputs.index_select(0, idxs)
        selected_labels = labels.index_select(0, idxs)
        return self.adv_attack_obj.eval(selected_grounds, adv_inputs,
                                        selected_labels, topk=topk)


    def eval_attack_only(self, adv_inputs, labels, topk=1):
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

        return self.adv_attack_obj.eval_attack_only(adv_inputs, labels,
                                                    topk=topk)

    def switch_model(self, new_classifier, new_normalizer=None):
        """ Builds a new AdversarialAttackParameters object with a new model but
            everything else the same
        ARGS:
            new_model : nn.Module subclass - neural
        RETURNS:
            AdversarialAttackParameters instance with this new model
        """
        new_attack_obj = self.adv_attack_obj.switch_model(new_classifier,
                                                  new_normalizer=new_normalizer)
        return AdversarialAttackParameters(new_attack_obj,
                             proportion_attacked=self.proportion_attacked,
                             attack_specific_params=self.attack_specific_params)



##############################################################################
#                                                                            #
#                               TRAINING OBJECT                              #
#                                                                            #
##############################################################################

class TrainingProtocol(object):
    """ Helper class to get passed in at training. Contains information about
        how 'adversarial training' is done. There's lots of schools of thoughts
        on best protocols here, so this just helps keep it clean what exactly
        we're doing
    """

    def __init__(self, minibatch_protocol, test_percentage):

        """ Minibatch protocol parameter: string
        Given an adversarial training with many potential attackParameters, we
        have three options:
            adv_only: the loss is computed wrt ONLY the generated adversarial
                      examples
            adv_and_orig: the loss is computed wrt ALL the adversarials, but
                          only one total copy of the originals
            duplicate_originals: the loss is computed wrt ALL the adversarials,
                                 and one copy of each original example per
                                 adversarial (i.e, there'll be duplicated clean
                                 examples in the minibatch if multiple attack
                                 styles are provided)

        """
        assert minibatch_protocol in ['adv_only',
                                      'adv_and_orig',
                                      'duplicate_originals']
        self.minibatch_protocol = minibatch_protocol

        """ Test percentage: float
            Value between [0.0, 1.0] for how much of
        """
        assert 0.0 <= test_percentage < 1.0
        self.test_percentage = test_percentage





class AdversarialTraining(object):
    """ Wrapper for training of a NN with adversarial examples cooked in
    """

    def __init__(self, classifier_net, normalizer,
                 experiment_name, architecture_name, protocol=None,
                 manual_gpu=None):

        """
        ARGS:
        classifier_net : nn.Module subclass - instance of neural net to classify
                         images. Can have already be trained, or not
        normalizer : DifferentiableNormalize - object to convert to zero-mean
                     unit-variance domain
        experiment_name : String - human-readable name of the 'trained_model'
                          (this is helpful for identifying checkpoints later)
        manual_gpu : None or bool - if not None is a manual override of whether
                     or not to use the GPU. If left None, we use the GPU if we
                     can

        ON NOMENCLATURE:
        Depending on verbosity levels, training checkpoints are saved after
        some training epochs. These are saved as
        '<experiment_name>/<architecture_name>/<epoch>.path.tar'

        Best practice is to keep architecture_name consistent across
        adversarially trained models built off the same architecture and having
        a descriptive experiment_name for each training instance
        """
        self.classifier_net =classifier_net
        self.normalizer = normalizer
        self.experiment_name = experiment_name
        self.architecture_name = architecture_name


        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

        self.verbosity_level = None
        self.verbosity_minibatch = None
        self.verbosity_adv = None
        self.verbosity_epoch = None

        self.logger = utils.TrainingLogger()
        self.log_level = None
        self.log_minibatch = None
        self.log_adv = None
        self.log_epoch = None

        if protocol is None:
            protocol = TrainingProtocol('adv_and_orig', 0.1)
        self.protocol = protocol


    def reset_logger(self):
        """ Clears the self.logger instance - useful occasionally """
        self.logger = utils.TrainingLogger()


    def set_verbosity_loglevel(self, level,
                               verbosity_or_loglevel='verbosity'):
        """ Sets the verbosity or loglevel for training.
            Is called in .train method so this method doesn't need to be
            explicitly called.

            Verbosity is mapped from a string to a comparable int 'level'.
            <val>_level : int - comparable value of verbosity
            <val>_minibatch: int - we do a printout every this many
                                       minibatches
            <val>_adv: int - we evaluate the efficacy of our attack every
                                 this many minibatches
            <val>_epoch: int - we printout/log and checkpoint every this many
                               epochs
        ARGS:
            level : string ['low', 'medium', 'high', 'snoop'],
                        varying levels of verbosity/logging in increasing order

        RETURNS: None
        """
        assert level in ['low', 'medium', 'high', 'snoop']
        assert verbosity_or_loglevel in ['verbosity', 'loglevel']
        setattr(self, verbosity_or_loglevel, level)


        _level = {'low': 0,
                  'medium': 1,
                  'high': 2,
                  'snoop': 420}[level]
        setattr(self, verbosity_or_loglevel + '_level', _level)


        _minibatch = {'medium': 2000,
                      'high': 100,
                      'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_minibatch', _minibatch)


        _adv = {'medium': 2000,
                'high': 100,
                'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_adv', _minibatch)


        _epoch = {'low': 100,
                  'medium': 10,
                  'high': 1,
                  'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_epoch', _epoch)


    def _get_regularize_adv_criterion(self):
        return nn.L1Loss()

    def _attack_subroutine(self, attack_parameters, inputs, labels,
                           epoch_num, minibatch_num, adv_saver,
                           logger, is_xvalidate=False):
        """ Subroutine to run the specified attack on a minibatch and append
            the results to inputs/labels.

        NOTE: THIS DOES NOT MUTATE inputs/labels !!!!

        ARGS:
            attack_parameters:  {k: AdversarialAttackParameters} (or None) -
                                if not None, contains info on how to do adv
                                attacks. If None, we don't train adversarially
            inputs : Tensor (NxCxHxW) - minibatch of data we build adversarial
                                        examples for
            labels : Tensor (longtensor N) - minibatch of labels
            epoch_num : int - number of which epoch we're working on.
                        Is helpful for printing
            minibatch_num : int - number of which minibatch we're working on.
                            Is helpful for printing
            adv_saver : None or checkpoints.CustomDataSaver -
                        if not None, we save the adversarial images for later
                        use, else we don't save them.
            logger : utils.TrainingLogger instance -  logger instance to keep
                     track of logging data if we need data for this instance

            duplicate_originals: boolean - if True we include one copy of the
                                 originals per adversarial attack, otherwise we
                                 just include one copy total
            is_xvalidate: boolean - True if called by _cross_validate, is
                          is useful for printing
        RETURNS:
            (inputs, labels, adv_inputs, coupled_inputs)
            where inputs = <arg inputs> ++ adv_inputs
                  labels is original labels
                  adv_inputs is the (Variable) adversarial examples generated,
                  coupled_inputs is the (Variable) inputs used to generate the
                                 adversarial examples (useful for when we don't
                                 augment 1:1).
        """
        if attack_parameters is None:
            return inputs, labels, None, None

        assert isinstance(attack_parameters, dict)

        adv_examples_total, adv_labels_total, coupled_inputs = [], [], []

        ####################################################################
        #   Build adversarial examples as per attack_parameter objects     #
        ####################################################################
        for (key, param) in attack_parameters.items():
            adv_data = param.attack(inputs, labels)
            adv_examples, adv_labels, adv_idxs, og_adv_inputs, _ = adv_data

            adv_examples_total.append(adv_examples)
            adv_labels_total.append(adv_labels)
            coupled_inputs.append(og_adv_inputs)


            ################################################################
            #   Internal evaluations, prints, and logging stuff            #
            ################################################################

            needs_print = (self.verbosity_level >= 1 and
                   minibatch_num % self.verbosity_adv == self.verbosity_adv - 1)
            needs_log = (self.loglevel_level >= 1 and
                     minibatch_num % self.loglevel_adv == self.loglevel_adv - 1)

            if needs_print or needs_log:
                accuracy = param.eval(inputs, adv_examples, labels, adv_idxs)

            if needs_print:
                print_str = ('[%d, %5d] accuracy: (%.3f, %.3f)' %
                    (epoch_num, minibatch_num + 1, accuracy[1], accuracy[0]))
                if is_xvalidate:
                    print_str = 'TEST: ' + print_str
                print(print_str)

            if needs_log and logger is not None:
                logger.log(key, epoch_num, minibatch_num + 1,
                           (accuracy[1], accuracy[0]))

            if adv_saver is not None: # Save the adversarial examples
                adv_saver.save_minibatch(adv_examples, adv_labels)

            ################################################################
            #   /Internal evaluations prints                               #
            ################################################################
        ####################################################################
        #   End building of adversarial examples                           #
        ####################################################################


        ####################################################################
        #   Now handle how we want to compute loss, as per protocol        #
        ####################################################################

        '''
        The inputs should depend on value of self.protocol.minibatch_protocol

        If minibatch_protocol is "adv_only":
            - output[0] should be [adversarial_ex1 :: ... :: adversarial_ex_k]
            - output[1] should be [adversarial_lab1 :: ... :: adversarial_lab_k]

        If minibatch_protocol is "adv_and_orig":
            - output[0] should be
              [original_mb :: adversarial_input_1 :: ... :: adversarial_input_k]
              and output[1] is
              [original_label :: adversarial_label_1 :: adversarial_label_k]


        If minibatch_protocol is "duplicate_originals":
          - output[0] should be
          [originals_1 :: ... :: originals_k ::
           atk_1(originals_1) :: ... :: atk_k(originals_k)]
          - output[1] should be
          [labels_1 :: ... :: labels_k ::
           labels_1 :: ... :: labels_k]

        Regardless output[2], output[3] are the adversarial_examples and their
        coupled inputs
        '''
        minibatch_protocol = self.protocol.minibatch_protocol

        if minibatch_protocol == 'adv_only':
            inputs = torch.cat(adv_examples_total, dim=0)
            labels = torch.cat(adv_labels_total, dim=0)

        elif minibatch_protocol == 'adv_and_orig':
            inputs = torch.cat([inputs] + [_.data for _ in adv_examples_total],
                               dim=0)
            labels = torch.cat([labels] + [_.data for _ in adv_labels_total],
                               dim=0)

        elif minibatch_protocol == 'duplicate_originals':
            inputs = torch.cat(coupled_inputs + adv_examples_total, dim=0)
            labels = torch.cat(adv_labels_total + adv_labels_total, dim = 0)
        else:
            raise NotImplementedError("Unknown protocol argument: %s" %
                                      minibatch_protocol)

        return (inputs, labels, torch.cat(adv_examples_total, dim=0),
                torch.cat(coupled_inputs, dim=0))


    def _minibatch_loss(self, inputs, labels, train_loss, attack_parameters,
                        epoch, minibatch_no, adv_saver, regularize_adv_scale,
                        regularize_adv_criterion, logger,
                        is_xvalidate=False):
        """ Subroutine to compute the loss for a single minibatch """

        # Build adversarial examples
        attack_out = self._attack_subroutine(attack_parameters,
                                             inputs, labels,
                                             epoch, minibatch_no, adv_saver,
                                             logger, is_xvalidate=is_xvalidate)
        inputs, labels, adv_examples, adv_inputs = attack_out
        # Now proceed with standard training
        self.normalizer.differentiable_call()
        self.classifier_net.train()
        inputs, labels = Variable(inputs), Variable(labels)
        # forward step
        outputs = self.classifier_net.forward(self.normalizer(inputs))
        loss = train_loss.forward(outputs, labels)

        if regularize_adv_scale is not None:
            # BE SURE TO 'DETACH' THE ADV_INPUTS!!!
            reg_adv_loss = regularize_adv_criterion(adv_examples,
                                                    adv_inputs.data)
            loss = loss + regularize_adv_scale * reg_adv_loss

        return loss


    def _cross_validate(self, test_loader, train_loss, attack_parameters, epoch,
                        regularize_adv_scale, regularize_adv_criterion):
        """ Performs cross-validation and returns the average/minibatch test
            loss
        """
        test_loss = 0
        test_minibatches = 0.0
        for i, data in enumerate(test_loader):
            inputs, labels = utils.cudafy(self.use_gpu, data)
            mb_loss = self._minibatch_loss(inputs, labels, train_loss,
                                           attack_parameters,
                                           epoch, i, None,
                                           regularize_adv_scale,
                                           regularize_adv_criterion,
                                           None,
                                        is_xvalidate=True)
            test_loss += float(mb_loss.data)
            test_minibatches += 1.0

        return test_loss / test_minibatches


    def train(self, data_loader, num_epochs, train_loss,
              optimizer=None, attack_parameters=None,
              verbosity='medium', loglevel='medium', logger=None,
              starting_epoch=0, adversarial_save_dir=None,
              regularize_adv_scale=None, best_test_loss=None):
        """ Modifies the NN weights of self.classifier_net by training with
            the specified parameters s
        ARGS:
            data_loader: torch.utils.data.DataLoader OR
                         checkpoints.CustomDataLoader - object that loads the
                         data
            num_epoch: int - number of epochs to train on
            train_loss: ????  - TBD
            optimizer: torch.Optimizer subclass - defaults to Adam with some
                       decent default params. Pass this in as an actual argument
                       to do anything different
            attack_parameters:  AdversarialAttackParameters obj | None |
                                {key: AdversarialAttackParameters} -
                                if not None, is either an object or dict of
                                objects containing names and info on how to do
                                adv attacks. If None, we don't train
                                adversarially
            verbosity : string - must be 'low', 'medium', 'high', which
                        describes how much to print
            loglevel : string - must be 'low', 'medium', 'high', which
                        describes how much to log
            logger : if not None, is a utils.TrainingLogger instance. Otherwise
                     we use this instance's self.logger object to log
            starting_epoch : int - which epoch number we start on. Is useful
                             for correct labeling of checkpoints and figuring
                             out how many epochs we actually need to run for
                             (i.e., num_epochs - starting_epoch)
            adversarial_save_dir: string or None - if not None is the name of
                                  the directory we save adversarial images to.
                                  If None, we don't save adversarial images
            regularize_adv_scale : float > 0 or None - if not None we do L1 loss
                                   between the logits of the adv examples and
                                   the inputs used to generate them. This is the
                                   scale constant of that loss
            stdout_prints: bool - if True we print out using stdout so we don't
                                  spam logs like crazy
            test_percentage: float - value between [0.0, 1.0] for how much of
                             the train set should be reserved as a test set for
                             each epoch. This is randomly selected at each epoch
            best_test_loss : float - for when restarting from checkpoint, this
                             stores the best loss on a test set, so we know
                             which the best model in future epochs is
            duplicate_originals: boolean - When training with multiple attack
                                 parameters, this includes one copy of originals
                                 per attack, thereby balancing loss-gradient
                                 direction. Else we just incorporate one copy of
                                 the originals only

        RETURNS:
            None, but modifies the classifier_net's weights
        """


        ######################################################################
        #   Setup/ input validations                                         #
        ######################################################################
        self.classifier_net.train() # in training mode
        assert isinstance(num_epochs, int)


        # Validate the attack parameters
        if attack_parameters is not None:
            if not isinstance(attack_parameters, dict):
                attack_parameters = {'attack': attack_parameters}
            # assert that the adv attacker uses the NN that's being trained
            for param in attack_parameters.values():
                assert (param.adv_attack_obj.classifier_net ==
                        self.classifier_net)


        # Validate the GPU parameters
        assert not (self.use_gpu and not cuda.is_available())
        if self.use_gpu:
            self.classifier_net.cuda()
        if attack_parameters is not None:
            for param in attack_parameters.values():
                param.set_gpu(self.use_gpu)


        # Validate the verbosity parameters
        assert verbosity in ['low', 'medium', 'high', 'snoop', None]
        self.set_verbosity_loglevel(verbosity,
                                    verbosity_or_loglevel='verbosity')
        verbosity_level = self.verbosity_level
        verbosity_minibatch = self.verbosity_minibatch
        verbosity_epoch = self.verbosity_epoch


        # Validate Loglevel parameters and initialize logger
        assert loglevel in ['low', 'medium', 'high', 'snoop', None]
        if logger is None:
            logger = self.logger
        if logger.data_count() > 0:
            print("WARNING: LOGGER IS NOT EMPTY! BE CAREFUL!")
        logger.add_series('training_loss')
        for key in (attack_parameters or {}).keys():
            logger.add_series(key)
        self.set_verbosity_loglevel(loglevel, verbosity_or_loglevel='loglevel')
        loglevel_level = self.loglevel_level
        loglevel_minibatch = self.loglevel_minibatch
        loglevel_epoch = self.loglevel_epoch


        # Adversarial image saver:
        adv_saver = None
        if adversarial_save_dir is not None and attack_parameters is not None:
            adv_saver = checkpoints.CustomDataSaver(adversarial_save_dir)


        # setup loss fxn, optimizer
        optimizer = optimizer or optim.Adam(self.classifier_net.parameters(),
                                            lr=0.001)

        # setup regularize adv object
        regularize_adv_criterion = None
        if regularize_adv_scale is not None:
            regularize_adv_criterion = self._get_regularize_adv_criterion()


        # Setup best test loss
        if best_test_loss is None:
            best_test_loss = float('inf')



        ######################################################################
        #   Training loop                                                    #
        ######################################################################
        for epoch in range(starting_epoch + 1, num_epochs + 1):
            # Build train/test sets
            if self.protocol.test_percentage > 0:
                train_loader, test_loader = utils.split_training_data(
                                                                data_loader,
                                                  self.protocol.test_percentage)
                logger.add_series('test_loss')
            else:
                train_loader = data_loader

            running_loss_print, running_loss_print_mb = 0.0, 0
            running_loss_log, running_loss_log_mb = 0.0, 0

            # Loop through minibatches
            for i, data in enumerate(train_loader):
                inputs, labels = utils.cudafy(self.use_gpu, data)
                optimizer.zero_grad()
                loss = self._minibatch_loss(inputs, labels, train_loss,
                                            attack_parameters,
                                            epoch, i, adv_saver,
                                            regularize_adv_scale,
                                            regularize_adv_criterion,
                                            logger)
                loss.backward()
                optimizer.step()

                # print things
                running_loss_print += float(loss.data)
                running_loss_print_mb +=1
                if (verbosity_level >= 1 and
                    i % verbosity_minibatch == verbosity_minibatch - 1):
                    print('[%d, %5d] loss: %.6f' %
                          (epoch, i + 1, running_loss_print /
                                 float(running_loss_print_mb)))
                    running_loss_print = 0.0
                    running_loss_print_mb = 0


                # log things
                running_loss_log += float(loss.data)
                running_loss_log_mb += 1
                if (loglevel_level >= 1 and
                    i % loglevel_minibatch == loglevel_minibatch - 1):
                    logger.log('training_loss', epoch, i + 1,
                               running_loss_log / float(running_loss_log_mb))
                    running_loss_log = 0.0
                    running_loss_log_mb = 0


            # end_of_epoch: Do validation on reserved test set and save best
            if self.protocol.test_percentage > 0:
                test_loss = self._cross_validate(test_loader, train_loss,
                                                 attack_parameters, epoch,
                                                 regularize_adv_scale,
                                                 regularize_adv_criterion)
                # Print test loss
                if (verbosity_level >= 1):
                    print('TEST: [%d] loss: %.6f' % (epoch, test_loss))


                # Log test loss
                if (loglevel_level >= 1):
                    logger.log('test_loss', epoch, 0, test_loss)

                # Save best model
                if test_loss <= best_test_loss:
                    print("Old best test loss:", best_test_loss)
                    print("New best test loss:", test_loss)
                    best_test_loss = test_loss
                    checkpoints.save_state_dict(self.experiment_name,
                                               self.architecture_name,
                                               'best', self.classifier_net,
                                               k_highest=None)


            if epoch % verbosity_epoch == 0:
                print("COMPLETED EPOCH %04d... checkpointing here" % epoch)
                checkpoints.save_state_dict(self.experiment_name,
                                            self.architecture_name,
                                            epoch, self.classifier_net,
                                            k_highest=3)

        if verbosity_level >= 1:
            print('Finished Training')

        return logger


    def train_from_checkpoint(self, data_loader, num_epochs, loss_fxn,
                              optimizer=None, attack_parameters=None,
                              verbosity='medium', loglevel='medium',
                              starting_epoch='max',
                              adversarial_save_dir=None,
                              regularize_adv_scale=None):
        """ Resumes training from a saved checkpoint with the same architecture.
            i.e. loads weights from specified checkpoint, figures out which
                 epoch we checkpointed on and then continues training until
                 we reach num_epochs epochs
        ARGS:
            same as in train
            starting_epoch: 'max' or int - which epoch we start training from.
                             'max' means the highest epoch we can find,
                             an int means a specified int epoch exactly.
                             'best' means we load the best-test loss model

        RETURNS:
            None
        """
        if attack_parameters is not None:
            if not isinstance(attack_parameters, dict):
                attack_parameters = {'attack': attack_parameters}
        ######################################################################
        #   Checkpoint handling block                                        #
        ######################################################################
        # which epoch to load
        valid_epochs = checkpoints.list_saved_epochs(self.experiment_name,
                                                     self.architecture_name)
        assert valid_epochs != []
        if starting_epoch == 'max':
            epoch = max([_ for _ in valid_epochs if _ != 'best'])
        elif starting_epoch == 'best':
            epoch = 'best'
        else:
            assert starting_epoch in valid_epochs
            epoch = starting_epoch

        # modify the classifer to use these weights
        self.classifier_net = checkpoints.load_state_dict(self.experiment_name,
                                                         self.architecture_name,
                                                         epoch,
                                                         self.classifier_net)
        if self.use_gpu:
            self.classifier_net.cuda()
        ######################################################################
        #   Compute the best test loss value for use in restart              #
        ######################################################################

        best_test_loss = None
        if self.protocol.test_percentage > 0:
            _, test_loader = utils.split_training_data(data_loader,
                                                  self.protocol.test_percentage)
            regularize_adv_criterion = None
            if regularize_adv_scale is not None:
                regularize_adv_criterion = self._get_regularize_adv_criterion()
            try:
                best_net = checkpoints.load_state_dict(self.experiment_name,
                                                       self.architecture_name,
                                                       'best',
                                                       self.classifier_net)
                best_net_computer = AdversarialTraining(best_net,
                                                        self.normalizer,
                                                        self.experiment_name,
                                                        self.architecture_name)
                best_net_computer.set_verbosity_loglevel('low', 'verbosity')
                best_net_computer.set_verbosity_loglevel('low', 'loglevel')
                test_loss_best = best_net_computer._cross_validate(test_loader,
                                                              loss_fxn,
                                                              attack_parameters,
                                                        0, regularize_adv_scale,
                                                       regularize_adv_criterion)
            except:
                print("NO SAVED BEST AVAILABLE")
                test_loss_best = float('inf')


            if epoch != 'best':
                loaded_net = checkpoints.load_state_dict(
                                                   self.experiment_name,
                                                   self.architecture_name,
                                                   epoch,
                                                   self.classifier_net)
                loaded_net_computer = AdversarialTraining(loaded_net,
                                                          self.normalizer,
                                                          self.experiment_name,
                                                          self.architecture_name)
                loaded_net_computer.set_verbosity_loglevel('low', 'verbosity')
                loaded_net_computer.set_verbosity_loglevel('low', 'loglevel')
                test_loss_loaded = loaded_net_computer._cross_validate(
                                                          test_loader, loss_fxn,
                                                          attack_parameters,
                                                    0, regularize_adv_scale,
                                                   regularize_adv_criterion)
            else:
                test_loss_loaded = float('inf')
            print("Best test loss:", test_loss_best)
            print("Max test loss:", test_loss_loaded)
            best_test_loss = min([test_loss_best, test_loss_loaded])



        ######################################################################
        #   Training block                                                   #
        ######################################################################

        self.train(data_loader, num_epochs, loss_fxn,
                   optimizer=optimizer,
                   attack_parameters=attack_parameters,
                   verbosity=verbosity,
                   loglevel=loglevel,
                   starting_epoch=epoch,
                   adversarial_save_dir=adversarial_save_dir,
                   regularize_adv_scale=regularize_adv_scale,
                   best_test_loss=best_test_loss)




