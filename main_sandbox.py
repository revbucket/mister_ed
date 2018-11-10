#!/usr/bin/env python

""" Main file that I'll run for experiments and such. VERY VOLATILE!!! """
from __future__ import print_function
import warnings
import os
import torch
import torch.nn as nn

import config
import prebuilt_loss_functions as plf
import loss_functions as lf
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import adversarial_attacks as aa
import adversarial_training as advtrain
import adversarial_perturbations as ap

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
    attack_examples = attack_examples or ['FGSM', 'BIM', 'PGD', 'CW2', 'CWLInf']

    ########################################################################
    #   SHARED BLOCK                                                       #
    ########################################################################

    # Initialize CIFAR classifier
    classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
    classifier_net.eval()

    # Collect one minibatch worth of data/targets
    val_loader = cifar_loader.load_cifar_data('val', normalize=False,
                                              batch_size=16)
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

        delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                 'lp_bound': 8.0 / 255})

        fgsm_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                                 normalizer=cifar_normer)

        fgsm_attack_obj = aa.FGSM(classifier_net, cifar_normer,
                                  delta_threat, fgsm_xentropy_loss)

        fgsm_original_images = ex_minibatch
        fgsm_original_labels = ex_targets

        fgsm_adv_images = fgsm_attack_obj.attack(fgsm_original_images,
                                                 fgsm_original_labels,
                                                 FGSM_L_INF).adversarial_tensors()

        fgsm_accuracy = fgsm_attack_obj.eval(fgsm_original_images,
                                             fgsm_adv_images,
                                             fgsm_original_labels)
        print("FGSM ATTACK ACCURACY: ")
        print("\t Original %% correct:    %s" % fgsm_accuracy[0])
        print("\t Adversarial %% correct: %s" % fgsm_accuracy[1])

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
        print("BIM ATTACK ACCURACY: ")
        print("\t Original %% correct:    %s" % bim_accuracy[0])
        print("\t Adversarial %% correct: %s" % bim_accuracy[1])

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

        delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                 'lp_bound': 8.0 / 255})

        pgd_attack_obj = aa.PGD(classifier_net, cifar_normer,
                                delta_threat, pgd_xentropy_loss)

        pgd_original_images = ex_minibatch
        pgd_original_labels = ex_targets

        pgd_adv_images = pgd_attack_obj.attack(pgd_original_images,
                                               pgd_original_labels,
                                               step_size=PGD_STEP_SIZE,
                                               num_iterations=PGD_NUM_ITER).adversarial_tensors()

        pgd_accuracy = pgd_attack_obj.eval(pgd_original_images,
                                           pgd_adv_images,
                                           pgd_original_labels)
        print("PGD ATTACK ACCURACY: ")
        print("\t Original %% correct:    %s" % pgd_accuracy[0])
        print("\t Adversarial %% correct: %s" % pgd_accuracy[1])

        if show_images:
            img_utils.display_adversarial_2row(classifier_net, cifar_normer,
                                               pgd_original_images,
                                               pgd_adv_images, 4)


    ##########################################################################
    #   CW L2 ATTACK                                                         #
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

        CW_INITIAL_SCALE_CONSTANT = 0.1
        CW_NUM_BIN_SEARCH_STEPS = 5
        CW_NUM_OPTIM_STEPS = 1000
        CW_DISTANCE_METRIC = 'l2'
        CW_CONFIDENCE = 0.0

        cw_f6loss = lf.CWLossF6
        delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 2,
                                                         'lp_bound': 3072.0})
        cwl2_obj = aa.CarliniWagner(classifier_net, cifar_normer, delta_threat,
                                    lf.L2Regularization, cw_f6loss)


        cwl2_original_images = ex_minibatch
        cwl2_original_labels = ex_targets

        cwl2_output = cwl2_obj.attack(ex_minibatch, ex_targets,
                                   num_bin_search_steps=CW_NUM_BIN_SEARCH_STEPS,
                                   num_optim_steps=CW_NUM_OPTIM_STEPS,
                                   verbose=True)

        print(cwl2_output['best_dist'])
        cwl2_adv_images = cwl2_output['best_adv_images']


        cwl2_accuracy = cwl2_obj.eval(cwl2_original_images,
                                      cwl2_adv_images,
                                      cwl2_original_labels)
        print("CWL2 ATTACK ACCURACY: ")
        print("\t Original %% correct:    %s" % cwl2_accuracy[0])
        print("\t Adversarial %% correct: %s" % cwl2_accuracy[1])

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
    classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
    classifier_net.eval()

    # Differentiable normalizer needed for classification
    cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                                 std=config.CIFAR10_STDS)



    ######################################################################
    #                     SIMPLE FGSM TRAINING EXAMPLE                   #
    ######################################################################
    if True:
        # Steps
        # 0) initialize hyperparams for attack/training
        # 1) setup attack loss object
        # 2) build attack and parameters for attack
        # 3) build training object, training loss, data loader
        # 4) train

        # 0
        FGSM_L_INF = 8.0 / 255.0
        FGSM_TRAINING_ATTACK_PROPORTION = 0.5
        FGSM_TRAINING_EPOCHS = 10

        # 1
        fgsm_attack_loss = plf.VanillaXentropy(classifier_net, cifar_normer)

        # 2
        fgsm_xentropy_attack_obj = aa.FGSM(classifier_net, cifar_normer,
                                           fgsm_attack_loss)
        fgsm_xentropy_attack_params = advtrain.AdversarialAttackParameters(
                                        fgsm_xentropy_attack_obj,
                                        FGSM_TRAINING_ATTACK_PROPORTION,
                                        {'attack_kwargs':
                                         {'l_inf_bound': FGSM_L_INF}})

        # 3
        half_fgsm_cifar = advtrain.AdversarialTraining(classifier_net,
                                                       cifar_normer,
                                                       'half_fgsm_cifar',
                                                       'cifar_resnet32')
        train_loss = nn.CrossEntropyLoss()
        train_loader = cifar_loader.load_cifar_data('train', normalize=False)

        # 4
        half_fgsm_cifar.train(train_loader, FGSM_TRAINING_EPOCHS, train_loss,
                              attack_parameters=fgsm_xentropy_attack_params,
                              verbosity='snoop')


def main_evaluation_script():
    """ Here's a little script to show how to evaluate a trained model
        against varying attacks (on the fly, without saving adv examples)
    """

    # Steps
    # 0) Initialize a classifier/normalizer/evaluation loader
    # 1) Build some attack objects to try
    # 2) Run the evaluation and print results

    # 0
    classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
    cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                                 std=config.CIFAR10_STDS)
    val_loader = cifar_loader.load_cifar_data('val', normalize=False)

    # 1
    L_INF_BOUND = 8.0 / 255.0
    # --- FGSM attack
    fgsm_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                             normalizer=cifar_normer)

    fgsm_attack_obj = aa.FGSM(classifier_net, cifar_normer,
                              fgsm_xentropy_loss)
    fgsm_spec_params = {'attack_kwargs': {'l_inf_bound': L_INF_BOUND}}
    fgsm_attack_params = advtrain.AdversarialAttackParameters(
                                fgsm_attack_obj, 0.5, fgsm_spec_params)

    # --- BIM attack
    BIM_L_INF = 8.0 / 255.0

    BIM_STEP_SIZE = 1.0 / 255.0
    BIM_NUM_ITER = 16

    bim_xentropy_loss = plf.VanillaXentropy(classifier_net,
                                            normalizer=cifar_normer)

    bim_attack_obj = aa.BIM(classifier_net, cifar_normer,
                            bim_xentropy_loss)
    bim_spec_params = {'attack_kwargs': {'l_inf_bound': L_INF_BOUND,
                                         'step_size': BIM_STEP_SIZE,
                                         'num_iterations': BIM_NUM_ITER}}
    bim_attack_params = advtrain.AdversarialAttackParameters(
                            bim_attack_obj, 0.5, bim_spec_params)

    attack_ensemble = {'fgsm': fgsm_attack_params,
                       'bim': bim_attack_params}


    # 2
    eval_obj = advtrain.AdversarialEvaluation(classifier_net, cifar_normer)
    eval_out = eval_obj.evaluate(val_loader, attack_ensemble,
                                 num_minibatches=5)



if __name__ == '__main__':
    warnings.warn("This file is no longer actively maintained. \n"
                   "Please use a Jupyter notebook for interactive sessions",
                  DeprecationWarning)
    main_attack_script(['FGSM'], show_images=True)
