""" Loads the config.json into python readable format """

import json
import os

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           'config.json'))
config_dir = os.path.dirname(config_path)
config_dict = json.loads(open(config_path, 'rb').read())

def path_resolver(path):
    if path.startswith('~/'):
        return os.path.expanduser(path)

    if path.startswith('./'):
        return os.path.join(*[config_dir] + path.split('/')[1:])


unexpanded_dataset_dir = config_dict['dataset_path']
unexpanded_model_path = config_dict['model_path']
unexpanded_output_image_path = config_dict['output_image_path']

DEFAULT_DATASETS_DIR = path_resolver(unexpanded_dataset_dir)
MODEL_PATH = path_resolver(unexpanded_model_path)
OUTPUT_IMAGE_PATH = path_resolver(unexpanded_output_image_path)


DEFAULT_BATCH_SIZE = config_dict['batch_size']
DEFAULT_WORKERS = config_dict['default_workers']
CIFAR10_MEANS = config_dict['cifar10_means']
CIFAR10_STDS = config_dict['cifar10_stds']

WIDE_CIFAR10_MEANS = config_dict['wide_cifar10_means']
WIDE_CIFAR10_STDS = config_dict['wide_cifar10_stds']


IMAGENET_MEANS = config_dict['imagenet_means']
IMAGENET_STDS = config_dict['imagenet_stds']
