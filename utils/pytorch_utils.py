""" Utilities for general pytorch helpfulness """

import torch
import numpy as np
import torchvision.transforms as transforms
import torch.cuda as cuda
import gc

from torch.autograd import Variable, Function
import subprocess


###############################################################################
#                                                                             #
#                                     SAFETY DANCE                            #
#                                                                             #
###############################################################################
# aka things for safer pytorch usage


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

##############################################################################
#                                                                            #
#                               CONVENIENCE STORE                            #
#                                                                            #
##############################################################################
# aka convenient things that are not builtin to pytorch

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


def clamp_ref(x, y, l_inf):
    """ Clamps each element of x to be within l_inf of each element of y """
    return torch.clamp(x - y , -l_inf, l_inf) + y


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


def checkpoint_incremental_array(output_file, numpy_list,
                                 return_concat=True):
    """ Takes in a string of a filename and a list of numpy arrays and
        concatenates them along first axis, saves them to a file, and then
        outputs a list containing only that single concatenated array
    ARGS:
        output_file : string ending in .npy - full path location of the
                      place we're saving this numpy array
        numpy_list : list of numpy arrays (all same shape except for the first
                     axis) - list of arrays we concat and save to file
        return_concat : boolean - if True, we return these concatenated arrays
                        in a list, else we return nothing
    RETURNS:
        maybe nothing, maybe the a singleton list containing the concatenated
        arrays
    """
    concat = np.concatenate(numpy_list, axis=0)
    np.save(output_file, concat)
    if return_concat:
        return [concat]



def sizeof_fmt(num, suffix='B'):
    """ https://stackoverflow.com/a/1094933
        answer by Sridhar Ratnakumar """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def clip_0_1(tensorlike):
    # Clips tensorlike object into [0., 1.0] range
    return torch.clamp(tensorlike, 0.0, 1.0)

def random_linf_pertubation(examples_like, l_inf):
    """ Returns an object of the same type/shape as examples_like that holds
        a uniformly random pertubation in the l_infinity box of l_inf.
        NOTE THAT THIS DOES NOT ADD TO examples_like!
    """

    is_var = isinstance(examples_like, Variable)

    random_tensor = (torch.rand(*examples_like.shape) * l_inf * 2 -
                     torch.ones(*examples_like.shape) * l_inf)

    random_tensor.type(type(examples_like))

    if is_var:
        return Variable(random_tensor)
    else:
        return random_tensor


def batchwise_norm(examples, lp, dim=0):
    """ Returns the per-example norm of the examples, keeping along the
        specified dimension.
        e.g. if examples is NxCxHxW, applying this fxn with dim=0 will return a
             N-length tensor with the lp norm of each example
    ARGS:
        examples : tensor or Variable -  needs more than one dimension
        lp : string or int - either 'inf' or an int for which lp norm we use
        dim : int - which dimension to keep
    RETURNS:
        1D object of same type as examples, but with shape examples.shape[dim]
    """

    assert isinstance(lp, int) or lp == 'inf'
    examples = torch.abs(examples)
    example_dim = examples.dim()
    if dim != 0:
        examples = examples.transpose(dim, 0)

    if lp == 'inf':
        for reduction in xrange(1, example_dim):
            examples, _ = examples.max(1)
        return examples

    else:
        examples = torch.pow(examples + 1e-10, lp)
        for reduction in xrange(1, example_dim):
            examples = examples.sum(1)
        return torch.pow(examples, 1.0 / lp)





###############################################################################
#                                                                             #
#                               CUDA RELATED THINGS                           #
#                                                                             #
###############################################################################

# fxn taken from https://discuss.pytorch.org/t/memory-leaks-in-trans-conv/12492
def get_gpu_memory_map():
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
    except:
        result = "<CAN'T GET GPU MEM>"
    try:
        return float(result)
    except:
        return result


def rough_gpu_estimate():
    """ Roughly estimates the size of the cuda tensors stored on GPUs.
        If multiple gpus, returns a dict of {GPU_id: total num elements }
        otherwise just returns the total number of elements
    """
    cuda_count = {}
    listprod = lambda l: reduce(lambda x,y: x * y, l)
    for el in gc.get_objects():
        if isinstance(el, (torch.tensor._TensorBase, Variable)) and el.is_cuda:
            device = el.get_device()
            cuda_count[device] = (cuda_count.get(device, 0) +
                                  listprod(el.size()))

    if len(cuda_count.keys()) == 0:
        return 0
    elif len(cuda_count.keys()) == 1:
        return sizeof_fmt(cuda_count.values()[0])
    else:
        return {k: sizeof_fmt(v) for k, v in cuda_count.items()}



##############################################################################
#                                                                            #
#                               CLASSIFICATION HELPERS                       #
#                                                                            #
##############################################################################
# aka little utils that are useful for classification

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



###############################################################################
#                                                                             #
#                                   NORMALIZERS                               #
#                                                                             #
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



