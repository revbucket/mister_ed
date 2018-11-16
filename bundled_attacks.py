
""" Holds the various attacks we can do """
from __future__ import print_function
from six import string_types
import torch
from torch.autograd import Variable, Function
from torch import optim
import functools
import inspect

import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import random
import sys
import custom_lpips.custom_dist_model as dm
import loss_functions as lf
import spatial_transformers as st
import torch.nn as nn
import adversarial_perturbations as ap
import adversarial_attacks as aa
import adversarial_training as advtrain

MAXFLOAT = 1e20



class AttackBundle(aa.AdversarialAttack):
    def __init__(self, classifier_net, normalizer, threat_model,
                 bundled_attacks, goal='misclassify', goal_params=None,
                 manual_gpu=None):
        """ These need to be on the same threat_model.
            It's a bit more complicated to compute mixed-threat-model attacks,
            so for the time being let's just assume all attacks need to have
            the same threat model
        ARGS:
            classifier_net
            normalizer
            threat_model
            goal
            bundled_attacks: dict
        """
        super(AttackBundle, self).__init__(classifier_net, normalizer,
                                           threat_model, manual_gpu=manual_gpu)

        assert isinstance(bundled_attacks, dict)
        for k, v in bundled_attacks.items():
            assert isinstance(k, basestring)
            assert isinstance(v, advtrain.AdversarialAttackParameters)
            assert v.adv_attack_obj.threat_model == threat_model
            assert v.proportion_attacked == 1.0
        self.bundled_attacks = bundled_attacks
        self.set_goal(goal, goal_params)

    def set_goal(self, goal, goal_params):
        """ Sets the goal. There's two types of goals: to cause a
            misclassification or maximize loss. If our goal is to misclassify,
            we'll stop upon the first attack that misclassifies
        """
        assert goal in ['misclassify', 'max_loss', 'min_perturbation',
                        'min_successful_perturbation']
        self.goal = goal
        self.goal_params = goal_params



    def attack_lazy(self, examples, labels):
        """ Lazy version of attack method: useful for when we want to stop under
            certain 'goal' conditions: i.e., if we want to run until we
            misclassify, then we can just run attack (i + 1) on examples that
            haven't been successful in the first i attacks
        ARGS:
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)
        """
        assert self.goal == 'misclassify' # only misclassify for now



        if self.goal_params is None:
            order = self.goal_params
        else:
            order = self.bundled_attacks.keys()
        ######################################################################
        #   Loop through attacks                                             #
        ######################################################################

        for k in order:
            pass
            ##################################################################
            #   First compute the attack on the remaining exmaples           #
            ##################################################################

            ##################################################################
            #   Figure out which exmaples still need attacking               #
            ##################################################################

            ##################################################################
            #   Build the attack batch for the next side                     #
            ##################################################################

        ######################################################################
        #   Now merge completely successful attacks                          #
        ######################################################################



    def attack(self, examples, labels):

        ######################################################################
        #   Handle 'lazy' goals                                              #
        ######################################################################

        if self.goal == 'misclassify':
            return self.attack_lazy(examples, labels)

        ######################################################################
        #   Handle 'nonlazy' goals                                           #
        ######################################################################
        batchwise_values = []
        name_to_order = {}
        order_to_name = {}
        batchwise_fxn = self.goal_params

        # make batchwise fxn only take in adversarial tensors
        if self.goal == 'max_loss':
            label_arg_name = inspect.getfullargspec(batchwise_fxn).args[1]

            batchwise_fxn = functools.partial(batchwise_fxn,
                                              eval(label_arg_name)=labels)
            comparator = torch.max
        elif self.goal == 'min_perturbation':
            comparator = torch.min

        elif self.goal == 'min_successful_perturbation':
            raise NotImplementedError("TODO")
            pass

        # Make perturbations and compute their batchwise values
        perturbations = {}
        for i, (name, attack_param) in enumerate(self.bundled_attacks.items()):
            name_to_order[name] = i
            order_to_name[i] = name

            _, _, _, _, perturbation = attack_param.attack(examples, labels)
            perturbations[name] = perturbation
            adversarial_tensors = perturbation.adversarial_tensors()
            batchwise_values.append(batchwise_fxn(adversarial_tensors))

        # Get max/min (as desired)
        selection_idxs = comparator(batchwise_values, dim=1)[1]

        # Iteratively build the combined perturbation
        running_perturbation = perturbations[order_to_name[0]]
        for i in xrange(len(batchwise_losses) - 1):
            next_perturbation = perturbations[order_to_name[i + 1]]
            running_perturbation.merge_perturbation(next_perturbation,
                                                    selection_idxs <= i)
        return running_perturbation

        for i, (name, perturbation) in enumerate(perturbation)

