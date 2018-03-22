""" Miscellaneous helpers (similar to util/util.py, but actually written by me)
"""
import torch
import numpy as np
import cifar_resnets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.cuda as cuda

from torch.autograd import Variable, Function

import matplotlib.pyplot as plt
import os
import re
import scipy.misc as smp
import random
import json



###############################################################################
#                           PARSE CONFIGS                                     #
###############################################################################

config_dict = json.loads(open('config.json', 'rb').read())

unexpanded_dataset_dir = config_dict['dataset_path']
unexpanded_model_path = config_dict['model_path']

DEFAULT_DATASETS_DIR = os.path.expanduser(unexpanded_dataset_dir)
RESNET_WEIGHT_PATH = os.path.expanduser(unexpanded_resnet_path)


DEFAULT_BATCH_SIZE = config_dict['batch_size']
DEFAULT_WORKERS = config_dict['default_workers']
CIFAR10_MEANS = config_dict['cifar10_means']
CIFAR10_STDS = config_dict['cifar10_stds']

###############################################################################
#                          END PARSE CONFIGS                                  #
###############################################################################




##############################################################################
#                                                                            #
#                               HELPFUL CLASSES                              #
#                                                                            #
##############################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


###############################################################################
#                            Normalizer classes                               #
###############################################################################

class IdentityNormalize(object):
    def __init__(self):
        pass

    def forward(self, var):
        return var


class DifferentiableNormalize(Function):

    def __init__(self, mean, std):
        super(DifferentiableNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.differentiable = True
        self.nondiff_normer = transforms.Normalize(mean, std)


    def __call__(self, var):
        if self.differentiable:
            return self.forward(var)
        else:
            return self.nondiff_normer(var)


    def _setter(self, c, mean, std):
        """ Modifies params going forward """
        if mean is not None:
            self.mean = mean
        assert len(self.mean) == c

        if std is not None:
            self.std = std
        assert len(self.std) == c

        if mean is not None or std is not None:
            self.nondiff_normer = transforms.Normalize(self.mean, self.std)


    def differentiable_call(self):
        """ Sets the __call__ method to be the differentiable version """
        self.differentiable = True


    def nondifferentiable_call(self):
        """ Sets the __call__ method to be the torchvision.transforms version"""
        self.differentiable = False


    def forward(self, var, mean=None, std=None):
        """ Normalizes var by subtracting the mean of each channel and then
            dividing each channel by standard dev
        ARGS:
            self - stores mean and std for later
            var - Variable of shape NxCxHxW
            mean - if not None is a list of length C for channel-means
            std - if not None is a list of length C for channel-stds
        RETURNS:
            variable of normalized var
        """
        c = var.shape[1]
        self._setter(c, mean, std)

        mean_var = Variable(var.data.new(self.mean).view(1, c, 1, 1))
        std_var = Variable(var.data.new(self.std).view(1, c, 1, 1))
        return (var - mean_var) / std_var


##############################################################################
#                                                                            #
#                               HELPFUL FUNCTIONS                            #
#                                                                            #
##############################################################################

def cuda_assert(use_cuda):
    assert not (use_cuda and not cuda.is_available())


def safe_var(entity, **kwargs):
    """ Returns a variable of an entity, which may or may not already be a
        variable
    """
    if isinstance(entity, Variable):
        return entity
    elif isinstance(entity, torch.tensor._TensorBase):
        return Variable(entity, **kwargs)
    else:
        raise Exception("Can't cast %s to a Variable" %
                        entity.__class__.__name__)


def safe_tensor(entity):
    """ Returns a tensor of an entity, which may or may not already be a
        tensor
    """
    if isinstance(entity, Variable):
        return entity.data
    elif isinstance(entity, torch.tensor._TensorBase):
        return entity
    elif isinstance(entity, np.ndarray):
        return torch.Tensor(entity) # UNSAFE CUDA CASTING
    else:
        raise Exception("Can't cast %s to a Variable" %
                        entity.__class__.__name__)


def tuple_getter(tensor, idx_tuple):
    """ access a tensor by a tuple """
    tensor_ = tensor
    for el in idx_tuple:
        tensor_ = tensor_[el]
    return tensor_


def tuple_setter(tensor, idx_tuple, val):
    """ Sets a tensor element while indexing by a tuple"""

    tensor_ = tensor
    for el in idx_tuple[:-1]:
        tensor_ = tensor_[el]

    tensor_[idx_tuple[-1]] = val
    return tensor


def torch_argmax(tensor):
    """ Returns the idx tuple that corresponds to the max value in the tensor"""

    flat_tensor = tensor.view(tensor.numel())
    _, argmax = flat_tensor.max(0)
    return np.unravel_index(int(argmax), tensor.shape)


def torch_argmin(tensor):
    """ Returns the idx tuple that corresponds to the min value in the tensor"""
    flat_tensor = tensor.view(tensor.numel())
    _, argmin = flat_tensor.min(0)
    return np.unravel_index(int(argmin), tensor.shape)


def random_element_index(tensor):
    """ Uniformly randomly selects an element from a tensor, outputs the index
        and the element
    ARGS:
        tensor : Torch.tensor (any shape):
    RETURNS:
        ((index_tuple), element)

    """
    shape = tensor.shape
    idx_tuple = tuple(random.randrange(s) for s in shape)
    el = tensor
    for idx in idx_tuple:
        el = el[idx]

    return idx_tuple, el


def nchw_l2(x, y, squared=True):
    """ Computes l2 norm between two NxCxHxW images
    ARGS:
        x, y: Tensor/Variable (NxCxHxW) - x, y must be same type & shape.
        squared : bool - if True we return squared loss, otherwise we return
                         square root of l2
    RETURNS:
        ||x - y ||_2 ^2 (no exponent if squared == False),
        shape is (Nx1x1x1)
    """
    temp = torch.pow(x - y, 2) # square diff


    for i in xrange(1, temp.dim()): # reduce on all but first dimension
        temp = torch.sum(temp, i, keepdim=True)

    if not squared:
        temp = torch.pow(temp, 0.5)

    return temp.squeeze()


def clamp_ref(x, y, l_inf):
    """ Clamps each element of x to be within l_inf of each element of y """
    return torch.clamp(x - y , -l_inf, l_inf) + y


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


def accuracy_int(output, target, topk=1):
    """ Computes the number of correct examples in the output.
    RETURNS an int!
    """
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return int(correct.data.sum())


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def show_images(img_tensor, normalize=None, ipython=True, num_rows=None):
    """ quick method to show list of cifar images in a single row
    ARGS:
        img_tensor: Nx3x32x32 floatTensor where N is the number of images.
        num_rows: if not specified, will try to make the output as square as
                  possible (erring on the side of being wider )
    RETURNS:
        None, but interactively shows an image
    """
    if normalize is not None:
        img_tensor = UnnormalizeImg(normalize)(img_tensor)

    # convert to numpy
    np_tensor = np.dstack(img_tensor.cpu().numpy())
    if ipython:
        plt.imshow(np_tensor.transpose(1, 2, 0))
    else:
        transforms.ToPILImage()(torch.from_numpy(np_tensor)).show()



def load_cifar_data(train_or_val, extra_args=None, dataset_dir=None,
                    normalize=False, batch_size=None):
    """ Builds a CIFAR10 data loader for either training or evaluation of
        CIFAR10 data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': True,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': False}
    constructor_kwargs.update(extra_args or {})

    # transform chain

    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.RandomCrop(32, 4),
                     transforms.ToTensor()]
    if normalize:
        normalizer = transforms.Normalize(mean=CIFAR10_MEANS,
                                          std=CIFAR10_STDS)
        transform_list.append(normalizer)


    transform_chain = transforms.Compose(transform_list)
    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_dir, train=train_or_val=='train',
                             transform=transform_chain, download=True),
            **constructor_kwargs)


def load_pretrained_cifar_resnet(flavor=32, use_gpu=False):
    """ Helper fxn to initialize/load the pretrained cifar resnet """

    # Resolve load path
    valid_flavor_numbers = [110, 1202, 20, 32, 44, 56]
    assert flavor in valid_flavor_numbers
    weight_path = os.path.join(RESNET_WEIGHT_PATH, 'resnet%s.th' % flavor)


    # Resolve CPU/GPU stuff
    if use_gpu:
        map_location = None
    else:
        map_location = (lambda s, l: s)


    # need to modify the resnet state dict to be proper

    bad_state_dict = torch.load(weight_path, map_location=map_location)
    correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                          bad_state_dict['state_dict'].items()}


    classifier_net = eval("cifar_resnets.resnet%s" % flavor)()
    classifier_net.load_state_dict(correct_state_dict)

    return classifier_net





