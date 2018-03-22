#!/usr/bin/env python

""" Main file that I'll run for experiments and such. VERY VOLATILE!!! """

import torch
import os
import misc
import adversarial_attacks as aa
import adversarial_training as advtrain
import loss_functions as lf
import discretization as disc
from torch.autograd import Variable
import pickle


BATCH_SIZE = misc.DEFAULT_BATCH_SIZE
WORKERS = misc.DEFAULT_WORKERS

##############################################################################
#                                                                            #
#                                   ATTACK EXAMPLES                          #
#                                                                            #
##############################################################################


def main_attack_script():

    # Initialize CIFAR classifier
    classifier_net = misc.load_pretrained_cifar_resnet(flavor=32, use_gpu=False)
    classifier_net.eval()


    # Collect one minibatch worth of data/targets
    val_loader = misc.load_cifar_data('val', normalize=False)
    ex_minibatch, ex_targets = next(iter(val_loader))


    # Differentiable normalizer needed for classification
    cifar_normer = misc.DifferentiableNormalize(mean=misc.CIFAR10_MEANS,
                                                std=misc.CIFAR10_STDS)
    # Set up loss functions
    xentropy_loss = lf.PartialXentropy(classifier_net, normalizer=cifar_normer)

    var_ex_minibatch = Variable(ex_minibatch, requires_grad=False)
    perceptual_loss = lf.PartialXentropyPerceptual(var_ex_minibatch,
                                                   classifier_net,
                                                   normalizer=cifar_normer)

    # DEFINE BOUNDS
    L_INF_BOUND = 0.1 # on nonnormalized [0.0, 1.0] data


    #########################################################################
    #   FGSM ATTACK BLOCK                                                   #
    #########################################################################
    if False:
        fgsm_xentropy_obj = aa.FGSM(classifier_net, cifar_normer, xentropy_loss)
        fgsm_xentropy_examples = fgsm_xentropy_obj.attack(ex_minibatch,
                                                          ex_targets,
                                                          L_INF_BOUND)



        fgsm_perceptual_obj = aa.FGSM(classifier_net, cifar_normer,
                                      perceptual_loss)
        fgsm_perceptual_examples = fgsm_perceptual_obj.attack(ex_minibatch,
                                                              ex_targets,
                                                              L_INF_BOUND)



    ##########################################################################
    #   BIM ATTACK BLOCK                                                     #
    ##########################################################################

    if False:
        '''
        bim_xentropy_obj = aa.BIM(classifier_net, cifar_normer, xentropy_loss)
        bim_xentropy_examples = bim_xentropy_obj.attack(ex_minibatch,
                                                        ex_targets,
                                                        L_INF_BOUND,
                                                        1./256,
                                                        num_iterations=3)
        '''

        bim_perceptual_obj = aa.BIM(classifier_net, cifar_normer,
                                    perceptual_loss)
        bim_perceptual_examples = bim_perceptual_obj.attack(ex_minibatch,
                                                            ex_targets,
                                                            L_INF_BOUND,
                                                            1./256,
                                                            num_iterations=3)






    ##########################################################################
    #   CW ATTACK                                                            #
    ##########################################################################

    if False:
        cw_loss = lf.CWLoss(classifier_net, cifar_normer)
        cwl2_obj = aa.CWL2(classifier_net, cifar_normer, cw_loss, 0.1,
                           num_bin_search_steps=10)

        cwl2_examples = cwl2_obj.attack(ex_minibatch, ex_targets, verbose=True)
        return
        disc.discretized_adversarial(cwl2_examples['best_adv_images'],
                                     classifier_net,
                                     cifar_normer)

        cw_loss_p = lf.CWLoss(Variable(ex_minibatch, requires_grad=False),
                              classifier_net, cifar_normer,
                              perceptual_params={'penalty': 100.0 })
        cwl2_obj_p = aa.CWL2(classifier_net, cifar_normer, cw_loss, 0.1,
                           num_bin_search_steps=10)

        cwl2_examples_p = cwl2_obj.attack(ex_minibatch, ex_targets,
                                          verbose=True)

    ##########################################################################
    #   PGD ATTACK                                                           #
    ##########################################################################

    if True:
        pgd_xentropy_obj = aa.LInfPGD(classifier_net, cifar_normer, xentropy_loss)
        pgd_xentropy_examples = pgd_xentropy_obj.attack(ex_minibatch,
                                                        ex_targets,
                                                        step_size=2.0/255,
                                                        num_iterations=10)





##############################################################################
#                                                                            #
#                               DEFENSE EXAMPLES                             #
#                                                                            #
##############################################################################



def main_defense_script():

    # Initialize CIFAR classifier
    classifier_net = misc.load_pretrained_cifar_resnet(flavor=32, use_gpu=False)
    classifier_net.eval()


    # Collect one minibatch worth of data/targets
    train_loader = misc.load_cifar_data('train', normalize=False)


    # Differentiable normalizer needed for classification
    cifar_normer = misc.DifferentiableNormalize(mean=misc.CIFAR10_MEANS,
                                                std=misc.CIFAR10_STDS)
    # Set up loss functions
    xentropy_loss = lf.PartialXentropy(classifier_net, normalizer=cifar_normer)


    # DEF DEBUG CHECKPOINT
    _, (examples, labels) = next(enumerate(train_loader))
    import interact


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
    main_attack_script()
