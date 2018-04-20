""" Specific utilities for image classification
    (i.e. RGB images i.e. tensors of the form NxCxHxW )
"""
import utils.pytorch_utils as utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random

def nhwc255_xform(img_np_array):
    """ Takes in a numpy array and transposes it so that the channel is the last
        axis. Also multiplies all values by 255.0
    ARGS:
        img_np_array : np.ndarray - array of shape (NxHxWxC) or (NxCxHxW)
                       [assumes that we're in NCHW by default,
                        but if not ambiguous will handle NHWC too ]
    RETURNS:
        array of form NHWC
    """
    assert isinstance(img_np_array, np.ndarray)
    shape = img_np_array.shape
    assert len(shape) == 4

    # determine which configuration we're in
    ambiguous = (shape[1] == shape[3] == 3)
    nhwc = (shape[1] == 3)

    # transpose unless we're unambiguously in nhwc case
    if nhwc and not ambiguous:
        return img_np_array * 255.0
    else:
        return np.transpose(img_np_array, (0, 2, 3, 1)) * 255.0


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
        img_tensor = utils.UnnormalizeImg(normalize)(img_tensor)

    # convert to numpy
    np_tensor = np.dstack(img_tensor.cpu().numpy())
    if ipython:
        plt.imshow(np_tensor.transpose(1, 2, 0))
    else:
        transforms.ToPILImage()(torch.from_numpy(np_tensor)).show()


def display_adversarial_2row(classifier_net, normalizer, original_images,
                        adversarial_images, num_to_show=4, which='incorrect',
                        ipython=False, margin_width=2):
    """ Displays adversarial images side-by-side with their unperturbed
        counterparts. Opens a window displaying two rows: top row is original
        images, bottom row is perturbed
    ARGS:
        classifier_net : nn - with a .forward method that takes normalized
                              variables and outputs logits
        normalizer : object w/ .forward method - should probably be an instance
                    of utils.DifferentiableNormalize or utils.IdentityNormalize
        original_images: Variable or Tensor (NxCxHxW) - original images to
                         display. Images in [0., 1.] range
        adversarial_images: Variable or Tensor (NxCxHxW) - perturbed images to
                            display. Should be same shape as original_images
        num_to_show : int - number of images to show
        which : string in ['incorrect', 'random', 'correct'] - which images to
                show.
                -- 'incorrect' means successfully attacked images,
                -- 'random' means some random selection of images
                -- 'correct' means unsuccessfully attacked images
        ipython: bool - if True, we use in an ipython notebook so slightly
                        different way to show Images
        margin_width - int : height in pixels of the red margin separating top
                             and bottom rows. Set to 0 for no margin
    RETURNS:
        None, but displays images
    """
    assert which in ['incorrect', 'random', 'correct']


    # If not 'random' selection, prune to only the valid things
    to_sample_idxs = []
    if which != 'random':
        classifier_net.eval() # can never be too safe =)

        # classify the originals with top1
        original_norm_var = normalizer.forward(utils.safe_var(original_images))
        original_out_logits = classifier_net.forward(original_norm_var)
        _, original_out_classes = original_out_logits.max(1)

        # classify the adversarials with top1
        adv_norm_var = normalizer.forward(utils.safe_var(adversarial_images))
        adv_out_logits = classifier_net.forward(adv_norm_var)
        _, adv_out_classes = adv_out_logits.max(1)


        # collect indices of matching
        selector = lambda var: (which == 'correct') == bool(float(var))
        for idx, var_el in enumerate(original_out_classes == adv_out_classes):
            if selector(var_el):
                to_sample_idxs.append(idx)
    else:
        to_sample_idxs = range(original_images.shape[0])

    # Now select some indices to show
    if to_sample_idxs == []:
        print "Couldn't show anything. Try changing the 'which' argument here"
        return

    to_show_idxs = random.sample(to_sample_idxs, min([num_to_show,
                                                      len(to_sample_idxs)]))

    # Now start building up the images : first horizontally, then vertically
    top_row = torch.cat([original_images[idx] for idx in to_show_idxs], dim=2)
    bottom_row = torch.cat([adversarial_images[idx] for idx in to_show_idxs],
                           dim=2)

    if margin_width > 0:
        margin = torch.zeros(3, margin_width, top_row.shape[-1])
        margin[0] = 1.0 # make it red
        margin = margin.type(type(top_row))
        stack = [top_row, margin, bottom_row]
    else:
        stack = [top_row, bottom_row]

    plt.imshow(torch.cat(stack, dim=1).cpu().numpy().transpose(1, 2, 0))
    plt.show()


def display_adversarial_notebook():
    pass

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
