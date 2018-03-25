""" Loads the config.json into python readable format """

import json
import os

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config_dict = json.loads(open(config_path, 'rb').read())

unexpanded_dataset_dir = config_dict['dataset_path']
unexpanded_model_path = config_dict['model_path']

DEFAULT_DATASETS_DIR = os.path.expanduser(unexpanded_dataset_dir)
RESNET_WEIGHT_PATH = os.path.expanduser(unexpanded_model_path)


DEFAULT_BATCH_SIZE = config_dict['batch_size']
DEFAULT_WORKERS = config_dict['default_workers']
CIFAR10_MEANS = config_dict['cifar10_means']
CIFAR10_STDS = config_dict['cifar10_stds']
