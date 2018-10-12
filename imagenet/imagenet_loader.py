""" Code to build a cifar10 data loader.

This leverages this github repo:
https://github.com/Cadene/pretrained-models.pytorch

"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import config
import imagenet.pretrainedmodels as ptm
import utils.pytorch_utils as utils

###############################################################################
#                           PARSE CONFIGS                                     #
###############################################################################

DEFAULT_DATASETS_DIR = config.DEFAULT_DATASETS_DIR
DEFAULT_BATCH_SIZE   = config.DEFAULT_BATCH_SIZE
IMAGENET_WEIGHT_PATH = os.path.join(config.MODEL_PATH)

DEFAULT_WORKERS      = config.DEFAULT_WORKERS
DEFAULT_MEANS        = config.IMAGENET_MEANS
DEFAULT_STDS         = config.IMAGENET_STDS


##############################################################################
#                                                                            #
#                                MODEL LOADER                                #
#                                                                            #
##############################################################################

def load_pretrained_imagenet(arch='nasnetalarge',
                             return_normalizer=False,
                             manual_gpu=None):

    assert arch in ['fbresnet152', 'bninception', 'resnext101_32x4d',
                      'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2',
                      'alexnet', 'densenet121', 'densenet169', 'densenet201',
                      'densenet161', 'resnet18', 'resnet34', 'resnet50',
                      'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0',
                      'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                      'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetalarge',
                      'nasnetamobile', 'cafferesnet101', 'senet154',
                      'se_resnet50', 'se_resnet101', 'se_resnet152',
                      'se_resnext50_32x4d', 'se_resnext101_32x4d']


    model = ptm.__dict__[arch](num_classes=1000,
                               pretrained='imagenet')
    model.eval()

    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    if use_gpu:
        model.cuda()

    if return_normalizer:
      normalizer = normalizer_from_imagenet_model(model)
      return model, normalizer
    return model

def normalizer_from_imagenet_model(model):
    """ Imagenet models taken from the git repo are not normalized and have
        different normalization means/stds. This builds a
        DifferentiableNormalizer object
    ARGS:
        model : output of load_pretrained_imagenet
    """
    if hasattr(model, 'mean'):
        mean = model.mean
    else:
        mean = DEFAULT_MEANS

    if hasattr(model, 'std'):
        std = model.std
    else:
        std = DEFAULT_STDS

    return utils.DifferentiableNormalize(mean, std)


###############################################################################
#                                                                             #
#                                DATA LOADER                                  #
#                                                                             #
###############################################################################

def load_imagenet_data(train_or_val, extra_args=None, dataset_dir=None,
                       normalize=False, batch_size=None, manual_gpu=None,
                       means=None, stds=None,
                       shuffle=True, no_transform=False):

    ######################################################################
    #   DEFAULTS                                                         #
    ######################################################################



    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR

    assert train_or_val in ['train', 'val']

    image_folder = {'val': 'ILSVRC2012_img_val'}[train_or_val] # error on train
    full_image_dir = os.path.join(dataset_dir, image_folder)

    transform_list = [transforms.RandomResizedCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor()]

    if normalize:
        means = means or DEFAULT_MEANS
        stds = stds or DEFAULT_STDS
        normalizer = transforms.Normalize(mean=means, std=stds)
        transform_list.append(normalizer)

    if no_transform:
        transform = transforms.Compose([])
    else:
        transform = transforms.Compose(transform_list)

    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    dataset = datasets.ImageFolder(
            full_image_dir, transform)

    ######################################################################
    #   Build DataLoader                                                 #
    ######################################################################
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size, shuffle=shuffle,
                                       num_workers=DEFAULT_WORKERS,
                                       pin_memory=use_gpu)

