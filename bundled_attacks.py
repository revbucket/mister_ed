
""" Holds the various attacks we can do """
from __future__ import print_function
from six import string_types
import torch

import utils.pytorch_utils as utils
import prebuilt_loss_functions as plf
import adversarial_perturbations as ap
import adversarial_attacks as aa
import adversarial_training as advtrain




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

        # Safety checks:
        assert isinstance(bundled_attacks, dict)
        for k, v in bundled_attacks.items():
            assert isinstance(k, string_types)
            assert isinstance(v, advtrain.AdversarialAttackParameters)
            assert v.adv_attack_obj.threat_model == threat_model #need eq method
            assert v.proportion_attacked == 1.0
        self.bundled_attacks = bundled_attacks
        self.set_goal(goal, goal_params)

    def set_goal(self, goal, goal_params):
        """ Sets the goal. There's two types of goals: to cause a
            misclassification or maximize loss. If our goal is to misclassify,
            we'll stop upon the first attack that misclassifies.

        ARGS:
            goal: string - what we aim to achieve by bundling attacks
            goal_params: None or goal-specific object: if None we do a standard
                         default thing. If not, we use the params depending on
                         the goal:
                    misclassify - the order to apply attacks
                                  (defaults to arbitrary dictionary key-order)
                    max_loss - the loss function to maximize
                               NEEDS SIGNATURE: (perturbation, labels)
                               RETURNS: tensor of size [N] (for N examples)
                    min_perturbation - the perturbation norm to minimize
                               (defaults to the builtin perturbation norm)
        RETURNS:
            None
        """
        assert goal in ['misclassify', 'max_loss', 'min_perturbation',
                        'min_successful_perturbation']
        self.goal = goal

        if goal_params is not None:
            self.goal_params = goal_params
            return

        #if goal is to _____, then params are ____
        if self.goal == 'misclassify':
            # the order in which we try attacks
            self.goal_params = self.bundled_attacks.keys()
        elif self.goal == 'max_loss':
            # the loss function we want to maximize
            self.goal_params = self._default_loss_for_max_loss
        elif self.goal in ['min_perturbation',
                           'min_successful_perturbation']:
            # the perturbation norm we want to minimize
            self.goal_params = self._default_norm_for_min_perturbation
        else:
            raise NotImplementedError("Goal: %s not supported" % self.goal)

    def _default_loss_for_max_loss(self, perturbation, labels):
        """ Is a loss function that takes (perturbations, labels) in and outputs
            a batchwise loss_fxn. Defaults to crossEntropyLoss
        ARGS:
            perturbation : ap.AdversarialPerturbation instance - needs an
                           adversarial_tensors() method, which should return an
                           NxCxHxW dimension tensor
            labels : tensor (N) - labels for the given examples
        RETURNS:
            tensor of shape (N) with loss per each label
        """
        loss_object = plf.VanillaXentropy(self.classifier_net, self.normalizer)
        return loss_object(perturbation, labels, output_per_example=True)

    def _default_norm_for_min_perturbation(self, perturbation):
        """ Is a norm for the perturbation and outputs batchwise norm vals
        ARGS:
            perturbation : ap.AdversarialPerturbation instance - needs an
                           adversarial_tensors() method, which should return an
                           NxCxHxW dimension tensor
        RETURNS:
            tensor of shape (N) with perturbation norm per example
        """
        return perturbation.perturbation_norm(perturbation)


    def attack_lazy(self, examples, labels, verbose=False):
        """ Lazy version of attack method: useful for when we want to stop under
            certain 'goal' conditions: i.e., if we want to run until we
            misclassify, then we can just run attack (i + 1) on examples that
            haven't been successful in the first i attacks
        ARGS:
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)
        RETURNS:  a single perturbation object

        """
        assert self.goal == 'misclassify' # only misclassify for now
        if self.goal_params is None:
            order = self.goal_params
        else:
            order = self.bundled_attacks.keys()

        num_examples = examples.shape[0]
        ######################################################################
        #   Loop through attacks                                             #
        ######################################################################

        remaining_examples, remaining_labels = examples, labels
        success_list = []
        perturbations = []
        total_success = 0
        for k in order:
            if verbose:
                print("Running attack %s on %s examples" % \
                      (k, remaining_examples.shape[0]))
            attack_param = self.bundled_attacks[k]
            # First compute the attack on the remaining exmaples
            _, _, _, _, perturbation = attack_param.attack(remaining_examples,
                                                           remaining_labels)
            perturbations.append(perturbation)

            # Figure out which examples still need attacking
            success_idxs = perturbation.collect_adversarially_successful(
                                                           self.classifier_net,
                                                           self.normalizer,
                                                           remaining_labels)
            success_idxs = success_idxs['idxs']

            success_list.append(success_idxs)
            success_set = set(success_idxs.cpu().numpy())
            total_success += len(success_set)
            if total_success == num_examples:
                break

            num_remaining = remaining_examples.shape[0]
            fail_idxs = torch.LongTensor([_ for _ in range(num_remaining)
                                          if _ not in success_set])
            if examples.is_cuda:
                fail_idxs = fail_idxs.cuda()

            # Build the attack batch for the next attack

            remaining_examples = remaining_examples.index_select(0, fail_idxs)
            remaining_labels = remaining_labels.index_select(0, fail_idxs)

        # If not all successful, have a 'null perturbation' for the remainder
        if remaining_examples.numel() > 0:
            fail_perturbation = self.threat_model(remaining_examples)
            fail_perturbation.attach_originals(remaining_examples)
            perturbations.append(fail_perturbation)
            success_list.append(range(len(fail_idxs)))
        ######################################################################
        #   Now merge completely successful attacks                          #
        ######################################################################
        # Compute the expand masks
        unseen_idxs = set(range(num_examples))
        expand_masks = []
        for idxs in success_list:
            sorted_unseen = sorted(unseen_idxs)
            expand_mask = []
            for j in idxs:
                expand_mask.append(sorted_unseen[j])
                unseen_idxs.remove(sorted_unseen[j])
            expand_masks.append(expand_mask)


        # Backtrack to build suffixes to expand from
        mask_suffixes = []
        running_suffix = []
        for j in range(len(expand_masks)-1, -1, -1):
            running_suffix.extend(expand_masks[j])
            mask_suffixes.append(running_suffix[:])
        mask_suffixes = [sorted(suffix) for suffix in mask_suffixes][::-1]

        scattered_perturbations = []
        for i, (pert, mask) in enumerate(zip(perturbations, mask_suffixes)):
            if i == 0:
                scattered_perturbations.append(pert)
                continue
            scatter_pert = pert.scatter_perturbation(num_examples, mask)
            scattered_perturbations.append(scatter_pert)

        # Finally, merge all perturbations together using the expand masks
        running_perturbation = scattered_perturbations[0]
        for j in range(len(scattered_perturbations) - 1):
            next_pert = scattered_perturbations[j + 1]
            mask = torch.LongTensor(expand_masks[j])

            self_mask = torch.zeros(num_examples).index_fill_(0, mask, 1) > 0
            if examples.is_cuda:
                self_mask = self_mask.cuda()
            running_perturbation = running_perturbation.merge_perturbation(
                                            next_pert, self_mask)

        running_perturbation.attach_originals(examples)
        return running_perturbation


    def attack_nonlazy(self, examples, labels):
        """ Nonlazy version of attack method: for when we want to compute
            adversarial examples for each attack in the bundle. We'll then
            merge the perturbations to take the 'best' (according to the goal)
            attack for each provided example
        ARGS:
            examples: Nxcxwxh tensor for N examples. NOT NORMALIZED (i.e. all
                      vals are between 0.0 and 1.0 )
            labels: single-dimension tensor with labels of examples (in same
                    order)
        RETURNS: a single perturbation object
        """

        ####################################################################
        #   Setup shared variables                                         #
        ####################################################################

        batchwise_values = []
        name_to_order = {}
        order_to_name = {}
        param_fxn = self.goal_params

        # make batchwise fxn take in adversarial tensors as only arg
        if self.goal == 'max_loss':
            # loss better have labels as second arg and nothing else
            batchwise_fxn = lambda ex: param_fxn(ex, labels)
            comparator = torch.max
        elif self.goal == 'min_perturbation':
            comparator = torch.min
            batchwise_fxn = param_fxn
        else: # elif self.goal == 'min_successful_perturbation':
            # this is more complicated b/c we need to ensure success.
            # Left as TODO
            raise NotImplementedError("TODO")

        ######################################################################
        #   Loop through attacks and compute perturbations for each          #
        ######################################################################

        perturbations = {}
        for i, (name, attack_param) in enumerate(self.bundled_attacks.items()):
            if verbose:
                print("Running attack %s" % name)
            name_to_order[name] = i
            order_to_name[i] = name
            _, _, _, _, perturbation = attack_param.attack(examples, labels)
            perturbations[name] = perturbation
            batchwise_values.append(batchwise_fxn(perturbation))


        ######################################################################
        #   Compute 'best' examples and merge perturbations                  #
        ######################################################################
        batchwise_values = torch.stack(batchwise_values, dim=0)
        selection_idxs = comparator(batchwise_values, dim=0)[1]

        # Iteratively build the combined perturbation
        running_perturbation = perturbations[order_to_name[0]]
        for i in range(len(order_to_name) - 1):
            next_perturbation = perturbations[order_to_name[i + 1]]
            running_perturbation.merge_perturbation(next_perturbation,
                                                    selection_idxs <= i)
        return running_perturbation



    def attack(self, examples, labels, verbose=False):
        """ See docs for subsidiary methods, depending on which goal we have """

        if self.goal in ['misclassify']:
            return self.attack_lazy(examples, labels, verbose=verbose)
        elif self.goal in ['max_loss', 'min_perturbation']:
            return self.attack_nonlazy(examples, labels, verbose=verbose)
        else:
            raise NotImplementedError("Goal: %s not supported" % self.goal)

