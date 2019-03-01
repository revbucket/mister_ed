""" Utilities for general pytorch helpfulness """
from __future__ import print_function
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.cuda as cuda
import gc
import random
import os
import warnings
from torch.autograd import Variable, Function
import subprocess


###############################################################################
#                                                                             #
#                                     SAFETY DANCE                            #
#                                                                             #
###############################################################################
# aka things for safer pytorch usage

def cudafy(use_gpu, seq, device=None):
    """ If use_gpu is True, returns cuda version of everything in tuple seq"""
    if use_gpu is False:
        return tuple(_.cpu() for _ in seq)
    else:
        if device != None:
            return tuple(_.to(device) for _ in seq)
        else:
            return tuple(_.cuda() for _ in seq)


def use_gpu():
    """ The shortcut to retrieve the environment variable 'MISTER_ED_GPU'"""
    try:
        str_val = os.environ['MISTER_ED_GPU']
    except:
        set_global_gpu()
        str_val = os.environ['MISTER_ED_GPU']
    assert str_val in ['True', 'False']
    return str_val == 'True'


def set_global_gpu(manual=None):
    """ Sets the environment variable 'MISTER_ED_GPU'. Defaults to using gpu
        if cuda is available
    ARGS:
        manual : bool - we set the 'MISTER_ED_GPU' environment var to the string
                 of whatever this is
    RETURNS
        None
    """
    if manual is None:
        val = cuda.is_available()
    else:
        val = manual
    os.environ['MISTER_ED_GPU'] = str(val)


def unset_global_gpu():
    """ Removes the environment variable 'MISTER_ED_GPU'
    # NOTE: this relies on unsetenv, which works on 'most flavors of Unix'
      according to the docs
    """
    try:
        os.unsetenv('MISTER_ED_GPU')
    except:
        raise Warning("os.unsetenv(.) isn't working properly")


def cuda_assert(use_cuda):
    assert not (use_cuda and not cuda.is_available())


def safe_var(entity, **kwargs):
    """ Returns a variable of an entity, which may or may not already be a
        variable
    """
    warnings.warn("As of >=pytorch0.4.0 this is no longer necessary",
                  DeprecationWarning)
    if isinstance(entity, Variable):
        return entity
    elif isinstance(entity, torch._C._TensorBase):
        return Variable(entity, **kwargs)
    else:
        raise Exception("Can't cast %s to a Variable" %
                        entity.__class__.__name__)


def safe_tensor(entity):
    """ Returns a tensor of an entity, which may or may not already be a
        tensor
    """
    warnings.warn("As of >=pytorch0.4.0 this is no longer necessary",
                  DeprecationWarning)
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

def clamp_0_1_delta(x, y):
    """ Returns the delta that'd have to be added to (x + y) such that
        (x + y) + delta is in the range [0.0, 1.0]
    """
    return torch.clamp(x + y, 0.0, 1.0) - (x + y)



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
        for reduction in range(1, example_dim):
            examples, _ = examples.max(1)
        return examples

    else:
        examples = torch.pow(examples + 1e-10, lp)
        for reduction in range(1, example_dim):
            examples = examples.sum(1)
        return torch.pow(examples, 1.0 / lp)


def batchwise_lp_project(x, lp, lp_bound, dim=0):
    """ Projects x (a N-by-(...) TENSOR) to be a N-by-(...) TENSOR into the
        provided lp ball. Note that this is the identity operation if the
        provided tensor is already inside the LP ball.
    ARGS:
        x : Tensor (N-by-(...)) - arbitrary style
        lp : 'inf' or int - which style of lp we use
        lp_bound : float - size of lp ball we project into
        dim : int - if not 0 is the dimension we keep and project onto
    RETURNS:
        None
    """
    assert isinstance(lp, int) or lp == 'inf'

    if lp == 'inf':
        return torch.clamp(x, -lp_bound, lp_bound)

    needs_squeeze = False
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
        needs_squeeze = True

    output = torch.renorm(x, lp, dim, lp_bound)

    if needs_squeeze:
        return output.squeeze()
    return output


def summed_lp_norm(examples, lp):
    """ Returns the sum of the lp norm of each example in examples
    ARGS:
        examples : tensor or Variable, with first dimension having size N
        lp : string or int - either 'inf' or an int for which lp norm we use
    RETURNS:
        sum of each of the lp norm of each of the N elements in examples
    """
    return torch.sum(batchwise_norm(examples, lp, dim=0))


def random_from_lp_ball(tensorlike, lp, lp_bound, dim=0):
    """ Returns a new object of the same type/shape as tensorlike that is
        randomly samples from the unit ball.



    ARGS:
        tensorlike : Tensor - reference object for which we generate
                     a new object of same shape/memory_location
        lp : int or 'inf' - which style of lp we use
        lp_bound : float - size of the L
        dim : int - which dimension is the 'minibatch' dimension
    RETURNS:
        new tensorlike where each slice across dim is uniform across the
        lp ball of size lp_bound
    """
    assert isinstance(lp, int) or lp == 'inf'

    if lp == 'inf':
        rand_direction = torch.rand_like(tensorlike)
        return rand_direction * (2 * lp_bound) - lp_bound

    elif lp == 2:
        # http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        # Transpose so 0 is the dimension we fixate on
        tensorlike = tensorlike.transpose(dim, 0)
        num_examples = tensorlike.shape[0]
        elements_per_example = tensorlike.numel() / num_examples

        # First sample from the n-sphere (gaussian and then normalize)
        rand = torch.normal(torch.zeros_like(tensorlike),
                            torch.ones_like(tensorlike))
        rand = rand.view(num_examples, -1)
        rand = rand.div(rand.norm(2, dim=1).unsqueeze(1)) *  lp_bound


        # And then scale by U[0,1]^(1/n)
        scale = torch.rand_like(rand) ** (1.0 / elements_per_example)
        rand = rand * scale

        # Transpose the answer back to input order
        return rand.view(*tensorlike.shape).transpose(dim, 0)

    else:
        #NOTE THIS IS NOT A UNIFORM SAMPLING METHOD!
        #(that's hard to implement, https://mathoverflow.net/a/9192/123034)
        rand_direction = torch.rand_like(tensorlike)

        rand_direction = rand_direction - 0.5 # allow for sign swapping
        # first magnify such that each element is above the ball
        min_norm = torch.min(batchwise_norm(rand_direction.abs(), lp, dim=dim))
        rand_direction = rand_direction / (min_norm + 1e-6)
        rand_magnitudes = torch.rand(tensorlike.shape[dim]).type(
                                                            tensorlike.type())

        # compute expand shape
        expand_shape = list(rand_direction.shape)
        for i in range(len(expand_shape)):
            if i != dim:
                expand_shape[i] = 1

        rand_magnitudes = rand_magnitudes.view(*expand_shape)\
                                         .expand(*rand_direction.shape)


        return torch.renorm(rand_direction, lp, dim, lp_bound) * rand_magnitudes


def tanh_transform(tensorlike, forward=True):
    """ Takes in Tensor or Variable and converts it between [0, 1] range and
        (-inf, +inf) range by performing an invertible tanh transformation.
    ARGS:
        tensorlike : Tensor or Variable (arbitrary shape) - object to be
                     modified into or out of tanh space
        forward : bool - if True we convert from [0, 1] space to (-inf, +inf)
                         space
                         if False we convert from (-inf, +inf) space to [0, 1]
                         space
    RETURNS:
        object of the same shape/type as tensorlike, but with the appropriate
        transformation
    """
    if forward:
        assert torch.min(tensorlike) >= 0.0
        assert torch.max(tensorlike) <= 1.0
        # first convert to [-1, +1] space
        temp = (tensorlike * 2 - 1) * (1 - 1e-6)
        return torch.log((1 + temp) / (1 - temp)) / 2.0

    else:
        return (torch.tanh(tensorlike) + 1) / 2.0


def fold_mask(x, y, mask):
    """ Creates a new tensor that's the result of masking between x and y
    ARGS:
        x : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 1
        y : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 0
        mask: ByteTensor (N) - masked values. Is only one dimensional: we expand
              it in the creation of this
    RETURNS:
        new object of the same shape/type as x and y
    """
    assert x.shape == y.shape
    assert mask.shape == (x.shape[0],)
    assert type(x) == type(y)
    is_var = isinstance(x, Variable)
    if is_var:
        assert isinstance(mask, Variable)


    per_example_shape = x.shape[1:]
    make_broadcastable = lambda m: m.view(-1, *tuple([1] * (x.dim() - 1)))

    broadcast_mask = make_broadcastable(mask)
    broadcast_not_mask = make_broadcastable(1 - safe_tensor(mask))
    if is_var:
        broadcast_not_mask = Variable(broadcast_not_mask)

    output = torch.zeros_like(x)
    output.add_(x * (broadcast_mask.type(x.type())))
    output.add_(y * (broadcast_not_mask.type(y.type())))

    return output

def scale_tensor_list(tensor_list, scale_factor):
    """ Takes in a list of tensor-like objects and returns the scaled version
        of each. Does not mutate the list!
    ARGS:
        tensor_list : Tensor[] - list of tensors to be scaled
        scale_factor : float - factor to multiply each tensor by
    RETURNS:
        new tensor list that resides in different memory location (same device!)
    """
    assert isinstance(tensor_list, list)
    output = []
    for el in tensor_list:
        output.append(el * scale_factor)
    return output

def add_tensor_list(tensor_list_1, tensor_list_2):
    """ Takes two lists of tensors each with identical shape and adds them,
        returns their sum
    ARGS:
        tensor_list_1 : Tensor[] - one component of the summand
        tensor_list_2 : Tensor[] - other component of the summad
    RETURNS:
        sum of the two tensor lists
    """

    output = []
    for tensor_a, tensor_b in zip(tensor_list_1, tensor_list_2):
        assert tensor_a.shape == tensor_b.shape
        output.append(tensor_a + tensor_b)
    return output


def pow_tensor_list(tensor_list, power):
    """ Takes elementwise powers of a list of tensor-like objects.
        Does not mutate the list!
    ARGS:
        tensor_list : Tensor[] - list of tensors to be exponentiated
        power: float - power to raise each element to
    RETURNS:
        list of elementwise powers of tensors
    """
    output = []
    for el in tensor_list:
        output.append(el ** power)
    return output

def tensor_list_op(tensor_list_1, tensor_list_2, op):
    """ Performs a specified operation on a zip of two lists of tensors
    ARGS:
        tensor_list_1 : Tensor[] - first argument to op
        tensor_list_2 : Tensor[] - second argument to op
        op : Tensor, Tensor -> Tensor - operation on two tensors
                e.g. for a sum, lambda a, b: a + b
    RETURNS:
        list of operations
    """
    return [op(tensor_a, tensor_b) for tensor_a, tensor_b in
                                    zip(tensor_list_1, tensor_list_2)]

def scatter_expand(originals, scatter_size, mask, identity_el=None):
    """ Takes original tensor into larger size
    ARGS:
        originals : tensor of size NxCxHxW
        scatter_size : int - the number of examples the scattering maps into
        mask: int[] - list of indices that each index of self maps into.
                      Should be sorted and unique
    RETURNS:
        tensor of shape scatter_size x CxHxW
    """

    original_shape = originals.shape
    num_examples = original_shape[0]
    assert scatter_size > num_examples
    assert len(mask) == num_examples
    assert all(isinstance(_, int) for _ in mask)
    assert sorted(set(mask)) == mask # sorted + unique
    mask_set = set(mask)
    # TODO: probably a way faster way to do this with some matmul stuff
    if identity_el is None:
        identity_el = torch.zeros(*original_shape[1:])
        if originals.is_cuda:
            identity_el = identity_el.cuda()

    stacked = []
    next_original_idx = 0
    for i in range(scatter_size):
        if i in mask_set:
            stacked.append(originals[next_original_idx])
            next_original_idx += 1
        else:
            stacked.append(identity_el)
    return torch.stack(stacked)


def filter_examples(model, examples, labels, normalizer=None):
    """ Filters only the examples that are correctly classified.
    ARGS:
        model : nn.Module instance - classifier that takes in examples
        examples : Tensor (NxCxHxW) - tensor of examples to be classified
        labels : Tensor (N) - long tensor of labels corresponding to examples
        normalizer : Normalizer - transform for the examples, could be None
    RETURS:
        correct_examples (N'xCxHxW), correct_examples (N')
        Returns only the elements of examples and their corresponding labels if
        the classifier correctly classifies them (top1)
    """
    if normalizer is None:
        normalizer = IdentityNormalize()
    output = model(normalizer(examples))
    top_1 = output.max(1)[1]
    correct_bytes = (top_1 == labels)
    if all(correct_bytes):
        return examples, labels
    return examples[correct_bytes], labels[correct_bytes]


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

def copy_numerical(numerical):
    """ Takes in a float, int, or Tensor and returns a deep copy of it """

    if isinstance(numerical, (float, int)):
        return numerical
    else:
        assert isinstance(numerical, torch.Tensor)
        return numerical.clone().detach()



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


def accuracy(output, target, topk=(1,), return_correct_idxs=False):
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

    if return_correct_idxs:
        correct_bytes = correct.sum(dim=0)
        correct_idxs = [i for i in range(len(correct_bytes))
                        if correct_bytes[i] > 0]

        return res, correct_idxs
    else:
        return res



###############################################################################
#                                                                             #
#                                   NORMALIZERS                               #
#                                                                             #
###############################################################################


class IdentityNormalize(Function):
    def __init__(self):
        pass

    def forward(self, var):
        return var

    def differentiable_call(self):
        pass

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
#                       TRAINING UTILITIES                                   #
#                                                                            #
##############################################################################


class TrainingLogger(object):

    def __init__(self):
        """ Unified object to keep track of training data at a specified logging
            level. Namely this tracks ground accuracy, loss and attack accuracy
            for each attack incorporated into adversarial training.
            Will ultimately contain plotting techniques too (TODO!)
        """
        self.series = {}

    def data_count(self):
        """ Returns the number of data points in this logger instance """
        return sum(len(_) for _ in self.series.values())


    def add_series(self, name):
        """ Adds the name of a 'data series' where each data series is a list
            of data-entries, where each data-entry is of the form
            ((epoch, minibatch), data-value ) [and data-value is a float]
        """
        if name not in self.series:
            self.series[name] = []


    def sort_series(self, name, return_keys=False):
        """ Simply returns the series of specified name sorted by epoch and then
            minibatch.
        ARGS:
            name: string - name of exsiting series in self.series
            return_keys: bool - if True, the output list is like
                         [((epoch, minibatch), val), ...]
                         and if False, it's just like [val, ... val...]
        RETURNS:
            sorted list of outputs, the exact form of which is determined by
            the value of return_keys
        """
        data_series = self.series[name]

        sorted_series = sorted(data_series, key=lambda p: p[0])

        if return_keys is False:
            return [_[1] for _ in sorted_series]
        else:
            return sorted_series

    def get_series(self, name):
        """ simple getter method for the given named data series """
        return self.series[name]


    def log_datapoint(self, name, data_tuple):
        """ Logs the full data point
        ARGS:
            name: string - name of existing series in self.series
            data_tuple : tuple of form ((epoch, minibatch), value)
        RETURNS:
            None
        """
        self.series[name].append(data_tuple)

    def log(self, name, epoch, minibatch, value):
        """ Logs the data point by specifying each of epoch, minibatch, value
        ARGS:
            name : string - name of existing series in self.series
            epoch: int - which epoch of training we're logging
            minibatch : int - which minibatch of training we're logging
            value : <unspecified, but preferably float> - value we're logging
        """
        self.log_datapoint(name, ((epoch, minibatch), value))


def split_training_data(data_loader, test_percentage):
    """ Takes a data loader and randomly partitions into a training and
        test set. Used for cross-validation in training
    ARGS:
        data_loader: torch.utils.data.DataLoader object
        test_percentage: float - value between [0.0, 1.0] for how big the
                         the test set should be (e.g., for 10% of the data
                         reserved for the test set, this should be 0.10)
                         [there might be a little bit of mismatch due to
                          minibatch sizes]
    RETURNS:
        [train_dataloader, test_dataloader]: list of two dataloaders with the
        train data loader first
    """
    assert 0 <= test_percentage <= 1.0
    dataset = data_loader.dataset
    dataset_size = len(dataset) # in number of examples
    test_size = int(test_percentage * dataset_size)
    train_size = dataset_size - test_size

    subsets = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Hacky way to carry over constructor args for datasets
    # copied from DataLoader source:
    # https://pytorch.org/docs/0.4.1/_modules/torch/utils/data/dataloader.html#DataLoader
    basic_attrs = ['batch_size',
                   'num_workers',
                   'collate_fn',
                   'pin_memory',
                   'drop_last',
                   'timeout',
                   'worker_init_fn']

    kwargs = {k: getattr(data_loader, k) for k in basic_attrs}

    # infer shuffle kwarg (though it's usually True)
    kwargs['shuffle'] = isinstance(data_loader.sampler,
                                   torch.utils.data.sampler.RandomSampler)

    return [torch.utils.data.DataLoader(subset, **kwargs) for subset in subsets]



