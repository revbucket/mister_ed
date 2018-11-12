""" Script to ensure that:
1) all dependencies are installed correctly
2) CIFAR data can be accessed locally
3) a functional classifier for CIFAR has been loaded.

"""


##############################################################################
#                                                                            #
#                       STEP ONE: DEPENDENCIES ARE INSTALLED                 #
#                                                                            #
##############################################################################
from __future__ import print_function
print("Checking imports...")
import sys
import os
sys.path.append(os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]))

import torch
import glob
import numpy as np
import math
import config
import torchvision.datasets as datasets

try: #This block from: https://stackoverflow.com/a/17510727
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import hashlib
print("...imports look okay!")


##############################################################################
#                                                                            #
#                       STEP TWO: CIFAR DATA HAS BEEN LOADED                 #
#                                                                            #
##############################################################################

def check_cifar_data_loaded():
    print("Checking CIFAR10 data loaded...")
    dataset_dir = config.DEFAULT_DATASETS_DIR

    train_set = datasets.CIFAR10(root=dataset_dir, train=True, download=True)
    val_set = datasets.CIFAR10(root=dataset_dir, train=False, download=True)

    print("...CIFAR10 data looks okay!")


check_cifar_data_loaded()

##############################################################################
#                                                                            #
#                       STEP THREE: LOAD CLASSIFIER FOR CIFAR10              #
#                                                                            #
##############################################################################


# https://stackoverflow.com/a/44873382
def file_hash(filename):
  h = hashlib.sha256()
  with open(filename, 'rb', buffering=0) as f:
    for b in iter(lambda : f.read(128*1024), b''):
      h.update(b)
  return h.hexdigest()



def load_cifar_classifiers():
    print("Checking CIFAR10 classifier exists...")

    # NOTE: pretrained models are produced by Yerlan Idelbayev
    # https://github.com/akamaster/pytorch_resnet_cifar10
    # I'm just hosting these on my dropbox for stability purposes

    # Check which models already exist in model directory
    resnet_name = lambda flavor: 'cifar10_resnet%s.th' % flavor
    total_cifar_files = set([resnet_name(flavor) for flavor in
                             [1202, 110, 56, 44, 32, 20]])
    total_cifar_files.add('Wide-Resnet28x10')

    try:
        os.makedirs(config.MODEL_PATH)
    except OSError as err:
        if not os.path.isdir(config.MODEL_PATH):
            raise err

    extant_models = set([_.split('/')[-1] for _ in
                         glob.glob(os.path.join(*[config.MODEL_PATH, '*']))])

    lacking_models = total_cifar_files - extant_models

    LINK_DEPOT = {resnet_name(20)  :  'https://www.dropbox.com/s/glchyr9ljnpgvb5/cifar10_resnet20.th?dl=1',
                  resnet_name(32)  :  'https://www.dropbox.com/s/kis991c5w2qtgpq/cifar10_resnet32.th?dl=1',
                  resnet_name(44)  :  'https://www.dropbox.com/s/sigj56ysrti6s6a/cifar10_resnet44.th?dl=1',
                  resnet_name(56)  :  'https://www.dropbox.com/s/3p6d5tkvdgcbru5/c7ifar10_resnet56.th?dl=1',
                  resnet_name(110) :  'https://www.dropbox.com/s/sp172x5vjlypfw6/cifar10_resnet110.th?dl=1',
                  resnet_name(1202):  'https://www.dropbox.com/s/4qxfa6dmdliw9ko/cifar10_resnet1202.th?dl=1',
                  'Wide-Resnet28x10': 'https://www.dropbox.com/s/5ln2gow7mnxub29/cifar10_wide-resnet28x10.th?dl=1'
                }


    HASH_DEPOT = {resnet_name(20)  :  '12fca82f0bebc4135bf1f32f6e3710e61d5108578464b84fd6d7f5c1b04036c8',
                  resnet_name(32)  :  'd509ac1820d7f25398913559d7e81a13229b1e7adc5648e3bfa5e22dc137f850',
                  resnet_name(44)  :  '014dd6541728a1c700b1642ab640e211dc6eb8ed507d70697458dc8f8a0ae2e4',
                  resnet_name(56)  :  '4bfd97631478d6b638d2764fd2baff3edb1d7d82252d54439343b6596b9b5367',
                  resnet_name(110) :  '1d1ed7c27571399c1fef66969bc4df68d6a92c8e6c41170f444e120e5354e3bc',
                  resnet_name(1202):  'f3b1deed382cd4c986ff8aa090c805d99a646e99d1f9227d7178183648844f62',
                  'Wide-Resnet28x10': 'd6a68ec2135294d91f9014abfdb45232d07fda0cdcd67f8c3b3653b28f08a88f'}

    for name in lacking_models:
        link = LINK_DEPOT[name]
        print("Downloading %s..." % name)
        u = urlopen(link)
        data = u.read()
        u.close()
        filename = os.path.join(config.MODEL_PATH, name)
        with open(filename, 'wb') as f:
            f.write(data)

        try:
            assert file_hash(filename) == HASH_DEPOT[name]
        except AssertionError as err:
            print("Something went wrong downloading %s" % name)
            os.remove(filename)
            raise err

    # Then load up all that doesn't already exist

    print("...CIFAR10 classifier looks okay")



load_cifar_classifiers()


print("\n Okay, you should be good to go now! ")
print("Try running tutorial_{1,2,3}.ipynb in notebooks/")

