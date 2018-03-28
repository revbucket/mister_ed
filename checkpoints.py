""" Code for saving/loading pytorch models

CHECKPOINT NAMING CONVENTIONS:
    <unique_experiment_name>.<architecture_abbreviation>.<6 digits of epoch number>path.tar
e.g.
    fgsm_def.resnet32.20180301.120000.path.tar

All checkpoints are stored in CHECKPOINT_DIR

Checkpoints are state dicts only!!!

"""

import torch
import os
import re
import glob

import config

hostname = os.uname()[1]
CHECKPOINT_DIR = os.path.expanduser(config.unexpanded_model_path)


def clear_experiment(experiment_name, architecture):
    """ Deletes all saved state dicts for an experiment/architecture pair """

    for filename in params_to_filename(experiment_name, architecture):
        full_path = os.path.join(*[CHECKPOINT_DIR, filename])
        os.remove(full_path) if os.path.exists(full_path) else None



def params_to_filename(experiment_name, architecture, epoch_val=None):
    """ Outputs string name of file.
    ARGS:
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int/(intLo, intHi)/None -
                    - if int we return this int exactly
                    - if (intLo, intHi) we return all existing filenames with
                      highest epoch in range (intLo, intHi), in sorted order
                    - if None, we return all existing filenames with params
                      in ascending epoch-sorted order

    RETURNS:
        filenames: string or (possibly empty) string[] of just the base name
        of saved models
    """

    if isinstance(epoch_val, int):
        return '.'.join([experiment_name, architecture, '%06d' % epoch_val,
                         'path', 'tar'])



    glob_prefix = os.path.join(*[CHECKPOINT_DIR,
                                 '%s.%s.*' % (experiment_name, architecture)])
    re_prefix = '%s\.%s\.' % (experiment_name, architecture)
    re_suffix = r'\.path\.tar'

    valid_name = lambda f: bool(re.match(re_prefix + r'\d{6}' + re_suffix,f))
    select_epoch = lambda f: int(re.sub(re_prefix, '',
                                        re.sub(re_suffix, '', f)))
    valid_epoch = lambda e: (e >= (epoch_val or (0, 0))[0] and
                             e <= (epoch_val or (0, float('inf')))[1])

    filename_epoch_pairs  = []
    for full_path in glob.glob(glob_prefix):
        filename = os.path.basename(full_path)
        if not valid_name(filename):
            continue

        epoch = select_epoch(filename)
        if valid_epoch(epoch):
            filename_epoch_pairs.append((filename, epoch))


    return [_[0] for _ in sorted(filename_epoch_pairs, key=lambda el: el[1])]



def save_state_dict(experiment_name, architecture, epoch_val, model,
                    k_highest=10):
    """ Saves the state dict of a model with the given parameters.
    ARGS:
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int - which epoch we're saving
        model : model - object we're saving the state dict of
        k_higest : int - if not None, we make sure to not include more than
                         k state_dicts for (experiment_name, architecture) pair,
                         keeping the k-most recent if we overflow
    RETURNS:
        The model we saved
    """

    # First resolve THIS filename
    this_filename = params_to_filename(experiment_name, architecture, epoch_val)

    # Next clear up memory if too many state dicts
    current_filenames = params_to_filename(experiment_name, architecture)
    delete_els = []
    if k_highest is not None:
        num_to_delete = len(current_filenames) - k_highest + 1
        if num_to_delete > 0:
            delete_els = sorted(current_filenames)[:num_to_delete]

    for delete_el in delete_els:

        full_path = os.path.join(*[CHECKPOINT_DIR, delete_el])
        os.remove(full_path) if os.path.exists(full_path) else None

    # Finally save the state dict
    torch.save(model.state_dict(), os.path.join(*[CHECKPOINT_DIR,
                                                  this_filename]))

    return model


def load_state_dict_from_filename(filename, model):
    """ Skips the whole parameter argument thing and just loads the whole
        state dict from a filename.
    ARGS:
        filename : string - filename without directories
        model : nn.Module - has 'load_state_dict' method
    RETURNS:
        the model loaded with the weights contained in the file
    """
    assert len(glob.glob(os.path.join(*[CHECKPOINT_DIR, filename]))) == 1

    # LOAD FILENAME
    model.load_state_dict(torch.load(os.path.join(*[CHECKPOINT_DIR, filename])))
    return model


def load_state_dict(experiment_name, architecture, epoch, model):
    """ Loads a checkpoint that was previously saved
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int - which epoch we're loading
    """

    filename = params_to_filename(experiment_name, architecture, epoch)
    return load_state_dict_from_filename(filename, model)




