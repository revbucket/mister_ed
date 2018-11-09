""" Adversarial Training script for training through tmux/screen --
Heavily borrowed from https://github.com/meliketoy/wide-resnet.pytorch

To use this:
copy this file into a script called advtrain_<experiment_name>.py
and modify the build_attack_parameters(...) function below to modify the
attack parameters used in adversarial training.

Also needs to be run from mister_ed home directory like
$ python -m scripts.advtrain_<experiment_name>.py

"""

##############################################################################
#                                                                            #
#   IMPORTS                                                                  #
#                                                                            #
##############################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

import re
import copy
import math


# Mister_ed imports
import adversarial_perturbations as ap
import prebuilt_loss_functions as plf
import adversarial_attacks as aa
import adversarial_training as advtrain
import spatial_transformers as st
import loss_functions as lf
import cifar10.cifar_loader as cl

###############################################################################
#                                                                             #
#   SAFETY METHODS                                                            #
#                                                                             #
###############################################################################

def validate_architecture(architecture_name):
    """ We only allow a few types of architectures for now """
    assert architecture_name in ['resnet32', 'resnet110', 'wide-resnet28x10'],\
           "INVALID ARCHITECTURE: %s" % architecture_name

def validate_filenaming(experiment_name):
    """ This file should be named advtrain_<experiment_name>.py """
    filename = os.path.basename(__file__)
    file_exp = re.sub(r'\.py$', '', re.sub(r'advtrain_', '', filename))
    assert file_exp == experiment_name,\
           "Filename needs to match provided experiment name"

##############################################################################
#                                                                            #
#   ATTACK PARAMETER GENERATOR                                               #
#                                                                            #
##############################################################################

GLOBAL_ATK_KWARGS = {'num_iterations': 100,
                     'optimizer': optim.Adam,
                     'optimizer_kwargs': {'lr': 0.01},
                     'signed': False,
                     'verbose': False}
L_INF_BOUND = 8.0 / 255.0
FLOW_LINF_BOUND = 0.05



# FGSM ONLY
def build_fgsm_attack(classifier_net, normalizer):
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=L_INF_BOUND))
    attack_loss = plf.VanillaXentropy(classifier_net, normalizer)
    fgsm_attack = aa.FGSM(classifier_net, cifar_normer, delta_threat,
                           attack_loss)
    attack_kwargs = {'verbose': GLOBAL_ATK_KWARGS['verbose']}
    params = advtrain.AdversarialAttackParameters(fgsm_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': attack_kwargs})
    return params

# STANDARD PGD LINF ATTACK
def build_pgd_linf_attack(classifier_net, normalizer):
    # PREBUILT LOSS FUNCTION
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=L_INF_BOUND,
                                                            ))
    attack_loss = plf.VanillaXentropy(classifier_net, normalizer=normalizer)
    pgd_attack = aa.PGD(classifier_net, normalizer, delta_threat, attack_loss)
    pgd_kwargs = copy.deepcopy(GLOBAL_ATK_KWARGS)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                            attack_specific_params={'attack_kwargs': pgd_kwargs})
    return params


# LINF + STADV
def build_pgd_linf_stadv_att(classifier_net, normalizer):
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=L_INF_BOUND))
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=FLOW_LINF_BOUND,
                                                           xform_class=st.FullSpatial,
                                                           use_stadv=True))
    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [delta_threat, flow_threat],
                                ap.PerturbationParameters(norm_weights=[0.00, 1.00]))
    adv_loss = lf.CWLossF6(classifier_net, normalizer)
    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss, 'st':st_loss},
                                  {'adv': 1.0, 'st': 0.05},
                                  negate=True)
    pgd_kwargs = copy.deepcopy(GLOBAL_ATK_KWARGS)
    pgd_kwargs['optimizer_kwargs']['lr'] = 0.001

    pgd_attack = aa.PGD(classifier_net, normalizer, sequence_threat, loss_fxn)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                            attack_specific_params={'attack_kwargs': pgd_kwargs})
    return params



# STADV ATTACK
def build_stadv_linf_attack(classifier_net, normalizer):
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=FLOW_LINF_BOUND,
                                                           xform_class=st.FullSpatial,
                                                           use_stadv=True))
    adv_loss = lf.CWLossF6(classifier_net, normalizer)
    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss, 'st':st_loss},
                                  {'adv': 1.0, 'st': 0.05},
                                  negate=True)

    pgd_kwargs = copy.deepcopy(GLOBAL_ATK_KWARGS)
    pgd_kwargs['optimizer_kwargs']['lr'] = 0.001

    pgd_attack = aa.PGD(classifier_net, normalizer, flow_threat, loss_fxn)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                            attack_specific_params={'attack_kwargs': pgd_kwargs})
    return params


# Delta + R + T ATTACK
def build_rotation_translation_attack(classifier_net, normalizer):
    # L_inf + flow style attack
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=L_INF_BOUND,
                                                            ))

    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style=1,
                                                            lp_bound=0.05,
                                                            xform_class=st.TranslationTransform,
                                                            ))
    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(xform_class=st.RotationTransform,
                                                              lp_style='inf', lp_bound=math.pi / 24.,
                                                              ))

    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [delta_threat, trans_threat, rotation_threat])
    pgd_kwargs = copy.deepcopy(GLOBAL_ATK_KWARGS)
    pgd_kwargs['optimizer_kwargs']['lr'] = 0.001

    loss_fxn = plf.VanillaXentropy(classifier_net, normalizer)
    pgd_attack = aa.PGD(classifier_net, normalizer, sequence_threat,
                         loss_fxn)

    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': pgd_kwargs})
    return params



def build_full_attack(classifier_net, normalizer):
    # L_inf + flow style attack
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=L_INF_BOUND,
                                                            ))

    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style=1,
                                                            lp_bound=0.05,
                                                            xform_class=st.TranslationTransform,
                                                            ))
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=FLOW_LINF_BOUND,
                                                           xform_class=st.FullSpatial,
                                                           use_stadv=True))

    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(xform_class=st.RotationTransform,
                                                              lp_style='inf', lp_bound=math.pi / 24.,
                                                              ))

    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [delta_threat, flow_threat, trans_threat, rotation_threat],
                                 ap.PerturbationParameters(norm_weights=[0.0, 1.0, 1.0, 1.0]))
    pgd_kwargs = copy.deepcopy(GLOBAL_ATK_KWARGS)
    pgd_kwargs['optimizer_kwargs']['lr'] = 0.001

    adv_loss = lf.CWLossF6(classifier_net, normalizer)
    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss, 'st':st_loss},
                                  {'adv': 1.0, 'st': 0.05},
                                  negate=True)

    pgd_attack = aa.PGD(classifier_net, normalizer, sequence_threat,
                         loss_fxn)

    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': pgd_kwargs})
    return params



def build_attack_params(classifier_net, normalizer):
    return build_pgd_linf_attack(classifier_net, normalizer)


##############################################################################
#                                                                            #
#   MAIN TRAINING LOOP                                                       #
#                                                                            #
##############################################################################

def main(architecture_name, experiment_name, num_epochs, batch_size=128,
         resume=False, verbosity='high'):
    validate_architecture(architecture_name)
    validate_filenaming(experiment_name)


    ##########################################################################
    #   Load the model + data loader                                         #
    ##########################################################################

    if architecture_name.startswith('resnet'):
        flavor = int(re.sub('^resnet', '', architecture_name))
        model, normalizer = cl.load_pretrained_cifar_resnet(flavor=flavor,
                                                        return_normalizer=True)

    elif architecture_name.startswith('wide-resnet'):
        model, normalizer = cl.load_pretrained_cifar_wide_resnet(
                                                         return_normalizer=True)
    else:
        raise Exception("INVALID ARCHITECTURE")

    cifar_dataset = cl.load_cifar_data('train', batch_size=batch_size)

    #########################################################################
    #   Build the training object + Train                                   #
    #########################################################################

    train_obj = advtrain.AdversarialTraining(model, normalizer,
                                             experiment_name, architecture_name)

    if resume:
        train_fxn = train_obj.train_from_checkpoint
    else:
        train_fxn = train_obj.train


    attack_params = build_attack_params(model, normalizer)
    criterion = nn.CrossEntropyLoss()
    train_fxn(cifar_dataset, num_epochs, criterion,
              attack_parameters=attack_params,
              verbosity=verbosity)


##############################################################################
#                                                                            #
#   ARGPARSER BLOCK                                                          #
#                                                                            #
##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument('--arch', type=str, help='architecture name')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='dropout rate for wide-resnets')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--verbosity', default='high', type=str,
                        choices=['low', 'medium', 'high', 'snoop'],
                        help='verbosity of training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="number of epochs trained")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="number of examples/minibatch")


    args = parser.parse_args()

    main(args.arch, args.exp, args.num_epochs, batch_size=args.batch_size,
         resume=args.resume, verbosity=args.verbosity)


