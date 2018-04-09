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
print "Checking imports..."
import sys
import os
sys.path.append(os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]))

import torch
import glob
import numpy as np
import math
import config
import torchvision.datasets as datasets
import urllib2
import hashlib
print "...imports look okay!"


##############################################################################
#                                                                            #
#                       STEP TWO: CIFAR DATA HAS BEEN LOADED                 #
#                                                                            #
##############################################################################

def check_cifar_data_loaded():
    print "Checking CIFAR10 data loaded..."
    dataset_dir = config.DEFAULT_DATASETS_DIR

    train_set = datasets.CIFAR10(root=dataset_dir, train=True, download=True)
    val_set = datasets.CIFAR10(root=dataset_dir, train=False, download=True)

    print "...CIFAR10 data looks okay!"


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
    print "Checking CIFAR10 classifier exists..."

    # NOTE: pretrained models are produced by Yerlan Idelbayev
    # https://github.com/akamaster/pytorch_resnet_cifar10
    # I'm just hosting these on my dropbox for stability purposes

    # Check which models already exist in model directory
    resnet_name = lambda flavor: 'cifar10_resnet%s.th' % flavor
    total_cifar_files = set([resnet_name(flavor) for flavor in
                             [1202, 110, 56, 44, 32, 20]])

    try:
        os.makedirs(config.MODEL_PATH)
    except OSError, err:
        if not os.path.isdir(config.MODEL_PATH):
            raise err

    extant_models = set([_.split('/')[-1] for _ in
                         glob.glob(os.path.join(*[config.MODEL_PATH, '*']))])

    lacking_models = total_cifar_files - extant_models
    dropbox_linker = lambda s: 'https://www.dropbox.com/s/jce5t3ysr555wqo/%s?dl=1' % s

    LINK_DEPOT = {k: dropbox_linker(k) for k in lacking_models}

    HASH_DEPOT = {resnet_name(20)  : '12fca82f0bebc4135bf1f32f6e3710e61d5108578464b84fd6d7f5c1b04036c8',
                  resnet_name(32)  : 'd509ac1820d7f25398913559d7e81a13229b1e7adc5648e3bfa5e22dc137f850',
                  resnet_name(44)  : 'f3b1deed382cd4c986ff8aa090c805d99a646e99d1f9227d7178183648844f62',
                  resnet_name(56)  : '4bfd97631478d6b638d2764fd2baff3edb1d7d82252d54439343b6596b9b5367',
                  resnet_name(110) : '1d1ed7c27571399c1fef66969bc4df68d6a92c8e6c41170f444e120e5354e3bc',
                  resnet_name(1202): 'f3b1deed382cd4c986ff8aa090c805d99a646e99d1f9227d7178183648844f62'}


    for name, link in LINK_DEPOT.iteritems():
        print "Downloading %s..." % name
        u = urllib2.urlopen(link)
        data = u.read()
        u.close()
        filename = os.path.join(config.MODEL_PATH, name)
        with open(filename, 'wb') as f:
            f.write(data)

        assert file_hash(filename) == HASH_DEPOT[name]


    # Then load up all that doesn't already exist

    print "...CIFAR10 classifier looks okay"



load_cifar_classifiers()


print "\n Okay, you should be good to go now! "
print "Try running main_sandbox.py for a quick example"
