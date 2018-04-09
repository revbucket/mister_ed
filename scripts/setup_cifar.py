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


def load_cifar_classifiers():
    print "Checking CIFAR10 classifier exists..."

    # NOTE: pretrained models are produced by Yerlan Idelbayev
    # https://github.com/akamaster/pytorch_resnet_cifar10
    # I'm just hosting these on my dropbox for stability purposes

    # Check which models already exist in model directory
    total_cifar_files = set(['cifar10_resnet%s.th' % flavor for flavor in
                             [1202, 110, 56, 44, 32, 20]])

    extant_models = set([_.split('/')[-1] for _ in
                         glob.glob(os.path.join(*[config.MODEL_PATH, '*']))])

    lacking_models = total_cifar_files - extant_models
    dropbox_linker = lambda s: 'https://www.dropbox.com/s/jce5t3ysr555wqo/%s?dl=1' % s

    LINK_DEPOT = {k: dropbox_linker(k) for k in lacking_models}



    for name, link in LINK_DEPOT.iteritems():
        print "Downloading %s..." % name
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
            raise

        u = urllib2.urlopen(link)
        data = u.read()
        u.close()
        with open(os.path.join(config.MODEL_PATH, name), 'wb') as f:
            f.write(data)



    # Then load up all that doesn't already exist

    print "...CIFAR10 classifier looks okay"



load_cifar_classifiers()


print "\n Okay, you should be good to go now! "
print "Try running main_sandbox.py for a quick example"
