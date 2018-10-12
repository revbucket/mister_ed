
# coding: utf-8

# In[1]:


""" File to generate an LPIPS attack in the infinity norm
(specifically to beat the Madry challenge)

Steps to get this done:
-1) Import a buncha things
 0) Load up my dataset, normalizer, adversarially trained net
 1) Build attack parameters
 2) Check efficacy on small dataset
 3) Build madry dataset
"""



# In[2]:


# Universal import block
# Block to get the relative imports working
from __future__ import print_function
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

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
import adversarial_evaluation as adveval
import checkpoints


# In[3]:


# Block 0: load dataset, normalizer, adversarially trained net
val_loader = cifar_loader.load_cifar_data('val', normalize=False, batch_size=8)

cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                           std=config.CIFAR10_STDS)

base_model = cifar_resnets.resnet32()
adv_trained_net = checkpoints.load_state_dict_from_filename('half_trained_madry.th', base_model)


# In[4]:


# Block 1: build attack parameters
ATTACK_KWARGS = {'l_inf_bound': 8.0/255.0,
                 'step_size': 0.5/255.0,
                 'num_iterations': 20,
                 'random_init': True,
                 'signed': True,
                 'verbose': False}
ATTACK_SPECIFIC_PARAMS = {'attack_kwargs': ATTACK_KWARGS}

def build_attack_loss(classifier, normalizer, lpips_penalty):
    """ Builds a regularized loss function for use in PGD
    Takes in (perturbed_examples, labels) and returns
    XEntropy(perturbed_examples, labels) + hyperparam * LPIPS(examples, perturbed_examples)
    """
    return plf.PerceptualXentropy(classifier, normalizer=normalizer,
                                  regularization_constant=lpips_penalty)

attack_params = {}
penalties = [0.01, 0.1, 1.0, 10.0, 100.0]
for penalty in penalties:
    loss_obj = build_attack_loss(adv_trained_net, cifar_normer, penalty)
    attack_obj = aa.LInfPGD(adv_trained_net, cifar_normer, loss_obj)
    attack_param = advtrain.AdversarialAttackParameters(attack_obj, 1.0,
                                                        attack_specific_params=ATTACK_SPECIFIC_PARAMS)
    attack_params[str(penalty)] = attack_param


# In[8]:


# Eval over just one
particular_param = attack_params['1.0']
eval_obj = adveval.AdversarialEvaluation(adv_trained_net, cifar_normer)
torch.cuda.empty_cache()
out = eval_obj.evaluate_ensemble(val_loader, {'partic': particular_param},
                                 num_minibatches=20)

print(out)

