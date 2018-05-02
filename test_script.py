import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.autograd import Variable
import numpy as np


# Universal import block
# Block to get the relative imports working
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import config
import matplotlib.pyplot as plt
import prebuilt_loss_functions as plf
import loss_functions as lf
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import adversarial_attacks as aa
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import utils.checkpoints as checkpoints
import adversarial_perturbations as ap
import adversarial_attacks_refactor as aar
import spatial_transformers as st

# Load up dataLoader, classifier, normer
use_gpu = torch.cuda.is_available()
classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32,
                                                           use_gpu=use_gpu)
classifier_net.eval()

val_loader = cifar_loader.load_cifar_data('val', normalize=False,
                                          batch_size=32, use_gpu=use_gpu)

cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                             std=config.CIFAR10_STDS)

examples, labels = next(iter(val_loader))

# example attack
delta_threat = ap.ThreatModel(ap.DeltaAddition,
                              ap.PerturbationParameters(lp_style='inf',
                                                        lp_bound=8.0/255))
attack_loss = plf.VanillaXentropy(classifier_net, cifar_normer)

fgsm_attack = aar.FGSM(classifier_net, cifar_normer, delta_threat, attack_loss)
attack_params = advtrain.AdversarialAttackParameters(fgsm_attack, 0.25)

reload(adveval)
eval_result = adveval.EvaluationResult(attack_params, classifier_net, cifar_normer,
                                       to_eval={'lpips': 'avg_successful_lpips'})
eval_result.eval(examples, labels)

