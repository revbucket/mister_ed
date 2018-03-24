#!/usr/bin/env python

""" Main file that I'll run for experiments and such. VERY VOLATILE!!! """

import torch
import os

import prebuilt_loss_functions as plf

import utils.pytorch_utils as utils
import utils.image_utils as img_utils

import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets

import adversarial_attacks as aa
import adversarial_training as advtrain
import loss_functions as lf
import discretization as disc
from torch.autograd import Variable
import pickle
import config

BATCH_SIZE = config.DEFAULT_BATCH_SIZE
WORKERS = config.DEFAULT_WORKERS

##############################################################################
#                                                                            #
#                                   ATTACK EXAMPLES                          #
#                                                                            #
##############################################################################


'''
    STEPS TO MAKE A BATCH OF ATTACKS:
    1) Load the classifier
'''

def main_attack_script(attack_examples=None,
                       show_images=False):

    # Which attacks to do...
    attack_examples = attack_examples or ['FGSM', 'BIM', 'PGD', 'CW2']

    ########################################################################
    #   SHARED BLOCK                                                       #
    ########################################################################

    # Initialize CIFAR classifier
    classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32,
                                                               use_gpu=False)
    classifier_net.eval()

    # Collect one minibatch worth of data/targets
    val_loader = cifar_loader.load_cifar_data('val', normalize=False)
    ex_minibatch, ex_targets = next(iter(val_loader))

    # Differentiable normalizer needed for classification
    cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                                 std=config.CIFAR10_STDS)



    #########################################################################
    #   FGSM ATTACK BLOCK                                                   #
    #########################################################################
    if 'FGSM' in attack_examples:
        # Example FGSM attack on a single minibatch
        # steps:
        #   0) initialize hyperparams
        #   1) setup loss object
        #   2) build attack object
        #   3) setup examples to attack
        #   4) perform attack
        #   5) evaluate attack (accuracy + display a few images )

        FGSM_L_INF = 8.0 / 255.0

        fgsm_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                                 normalizer=cifar_normer)

        fgsm_attack_obj = aa.FGSM(classifier_net, cifar_normer,
                                  fgsm_xentropy_loss)

        fgsm_original_images = ex_minibatch
        fgsm_original_labels = ex_targets

        fgsm_adv_images = fgsm_attack_obj.attack(fgsm_original_images,
                                                 fgsm_original_labels,
                                                 FGSM_L_INF)

        fgsm_accuracy = fgsm_attack_obj.eval(fgsm_original_images,
                                             fgsm_adv_images,
                                             fgsm_original_labels)
        print "FGSM ATTACK ACCURACY: "
        print "\t Original %% correct:    %s" % fgsm_accuracy[0]
        print "\t Adversarial %% correct: %s" % fgsm_accuracy[1]

        if show_images:
            img_utils.display_adversarial_2row(classifier_net, cifar_normer,
                                               fgsm_original_images,
                                               fgsm_adv_images, 4)


    ##########################################################################
    #   BIM ATTACK BLOCK                                                     #
    ##########################################################################

    if 'BIM' in attack_examples:
        # Example BIM attack on a single minibatch
        # steps:
        #   0) initialize hyperparams
        #   1) setup loss object
        #   2) build attack object
        #   3) setup examples to attack
        #   4) perform attack
        #   5) evaluate attack

        BIM_L_INF = 8.0 / 255.0
        BIM_STEP_SIZE = 1.0 / 255.0
        BIM_NUM_ITER = 16

        bim_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                                normalizer=cifar_normer)

        bim_attack_obj = aa.BIM(classifier_net, cifar_normer,
                                bim_xentropy_loss)

        bim_original_images = ex_minibatch
        bim_original_labels = ex_targets

        bim_adv_images = bim_attack_obj.attack(bim_original_images,
                                               bim_original_labels,
                                               l_inf_bound=BIM_L_INF,
                                               step_size=BIM_STEP_SIZE,
                                               num_iterations=BIM_NUM_ITER)

        bim_accuracy = bim_attack_obj.eval(bim_original_images,
                                           bim_adv_images,
                                           bim_original_labels)
        print "BIM ATTACK ACCURACY: "
        print "\t Original %% correct:    %s" % bim_accuracy[0]
        print "\t Adversarial %% correct: %s" % bim_accuracy[1]

        if show_images:
            img_utils.display_adversarial_2row(classifier_net, cifar_normer,
                                               bim_original_images,
                                               bim_adv_images, 4)

    ##########################################################################
    #   PGD ATTACK BLOCK                                                     #
    ##########################################################################

    if 'PGD' in attack_examples:
        # Example BIM attack on a single minibatch
        # steps:
        #   0) initialize hyperparams
        #   1) setup loss object
        #   2) build attack object
        #   3) setup examples to attack
        #   4) perform attack
        #   5) evaluate attack

        PGD_L_INF = 8.0 / 255.0
        PGD_STEP_SIZE = 1.0 / 255.0
        PGD_NUM_ITER = 16

        pgd_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                                normalizer=cifar_normer)

        pgd_attack_obj = aa.LInfPGD(classifier_net, cifar_normer,
                                    pgd_xentropy_loss)

        pgd_original_images = ex_minibatch
        pgd_original_labels = ex_targets

        pgd_adv_images = pgd_attack_obj.attack(pgd_original_images,
                                               pgd_original_labels,
                                               l_inf_bound=PGD_L_INF,
                                               step_size=PGD_STEP_SIZE,
                                               num_iterations=PGD_NUM_ITER)

        pgd_accuracy = pgd_attack_obj.eval(pgd_original_images,
                                           pgd_adv_images,
                                           pgd_original_labels)
        print "PGD ATTACK ACCURACY: "
        print "\t Original %% correct:    %s" % pgd_accuracy[0]
        print "\t Adversarial %% correct: %s" % pgd_accuracy[1]

        if show_images:
            img_utils.display_adversarial_2row(classifier_net, cifar_normer,
                                               pgd_original_images,
                                               pgd_adv_images, 4)


    ##########################################################################
    #   CW ATTACK                                                            #
    ##########################################################################

    if 'CWL2' in attack_examples:

        # Example Carlini Wagner L2 attack on a single minibatch
        # steps:
        #   0) initialize hyperparams
        #   1) setup loss object
        #   2) build attack object
        #   3) setup examples to attack
        #   4) perform attack
        #   5) evaluate attack

        CWL2_INITIAL_SCALE_CONSTANT = 0.1
        CWL2_NUM_BIN_SEARCH_STEPS = 5
        CWL2_NUM_OPTIM_STEPS = 1000
        CWL2_DISTANCE_METRIC = 'l2'
        CWL2_CONFIDENCE = 0.0

        cwl2_loss = plf.CWPaperLoss(classifier_net, cifar_normer, kappa=0.0)
        cwl2_obj = aa.CWL2(classifier_net, cifar_normer, cwl2_loss,
                           CWL2_INITIAL_SCALE_CONSTANT,
                           num_bin_search_steps=CWL2_NUM_BIN_SEARCH_STEPS,
                           num_optim_steps=CWL2_NUM_OPTIM_STEPS,
                           distance_metric_type=CWL2_DISTANCE_METRIC,
                           confidence=CWL2_CONFIDENCE)


        cwl2_original_images = ex_minibatch
        cwl2_original_labels = ex_targets

        cwl2_output = cwl2_obj.attack(ex_minibatch, ex_targets,
                                      verbose=True)

        cwl2_adv_images = cwl2_output['best_adv_images']


        cwl2_accuracy = cwl2_obj.eval(cwl2_original_images,
                                      cwl2_adv_images,
                                      cwl2_original_labels)
        print "CWL2 ATTACK ACCURACY: "
        print "\t Original %% correct:    %s" % cwl2_accuracy[0]
        print "\t Adversarial %% correct: %s" % cwl2_accuracy[1]

        if show_images:
            img_utils.display_adversarial_2row(classifier_net, cifar_normer,
                                               cwl2_original_images,
                                               cwl2_adv_images, 4)




##############################################################################
#                                                                            #
#                               DEFENSE EXAMPLES                             #
#                                                                            #
##############################################################################


def main_defense_script():


    ########################################################################
    #   SHARED BLOCK                                                       #
    ########################################################################

    # Initialize CIFAR classifier
    classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32,
                                                               use_gpu=False)
    classifier_net.eval()

    # Collect one minibatch worth of data/targets
    val_loader = cifar_loader.load_cifar_data('val', normalize=False)
    ex_minibatch, ex_targets = next(iter(val_loader))

    # Differentiable normalizer needed for classification
    cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                                 std=config.CIFAR10_STDS)


    ######################################################################
    #                           NO ATTACK EXAMPLE                        #
    ######################################################################

    # DEBUGGING PHASE
    if True:

        xentropy_loss = lf.PartialXentropy(classifier_net, normalizer=cifar_normer)
        fgsm_xentropy_obj = aa.FGSM(classifier_net, cifar_normer, xentropy_loss)
        params = advtrain.AdversarialAttackParameters(fgsm_xentropy_obj,
                                                      0.5)
        vanilla_train = advtrain.AdversarialTraining(classifier_net,
                                                     cifar_normer)

        vanilla_train.train(train_loader, 2,
                            torch.nn.CrossEntropyLoss(),
                            attack_parameters=params, verbosity='snoop')







if __name__ == '__main__':
    main_attack_script(['CWL2'], True)
