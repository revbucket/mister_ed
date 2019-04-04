""" Code to build a mnist data loader """


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.pytorch_utils as utils
import config

###############################################################################
#                           PARSE CONFIGS                                     #
###############################################################################

DEFAULT_DATASETS_DIR = config.DEFAULT_DATASETS_DIR
DEFAULT_BATCH_SIZE   = config.DEFAULT_BATCH_SIZE
DEFAULT_WORKERS      = config.DEFAULT_WORKERS

###############################################################################
#                          END PARSE CONFIGS                                  #
###############################################################################


##############################################################################
#                                                                            #
#                               DATA LOADER                                  #
#                                                                            #
##############################################################################

def load_mnist_data(train_or_val, extra_args=None, dataset_dir=None,
                    batch_size=None, manual_gpu=None,
                    shuffle=True):
    """ Builds a MNIST data loader for either training or evaluation of
        MNIST data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}
    constructor_kwargs.update(extra_args or {})
    transform_chain = transforms.Compose([transforms.ToTensor()])




    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
            datasets.MNIST(root=dataset_dir, train=train_or_val=='train',
                           transform=transform_chain, download=True),
            **constructor_kwargs)



###############################################################################
#                                                                             #
#                    SUBSET SELECTOR (e.g. for binary MNIST)                  #
#                                                                             #
###############################################################################



def load_single_digits(train_or_val, digit_list, extra_args=None,
                       dataset_dir=None, batch_size=None, manual_gpu=None,
                       shuffle=True):

    ##########################################################################
    #   First load the full MNIST dataset and define the label mapping       #
    ##########################################################################
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    dataset = load_mnist_data(train_or_val, extra_args=extra_args,
                              dataset_dir=dataset_dir, manual_gpu=manual_gpu,
                              batch_size=batch_size, shuffle=shuffle)

    digit_label_dict = {digit : i for i, digit in enumerate(digit_list)}
    def label_map(label_tensor, digit_label_dict=digit_label_dict):
        # Takes in a long tensor and maps every element to the label map

        # Just do a naive loop... w/e
        for i in range(len(label_tensor)):
            label_tensor[i] = digit_label_dict[label_tensor[i].item()]
        return label_tensor

    ##########################################################################
    #   Next select out only the relevant data points                        #
    ##########################################################################


    selected_data, selected_labels = [], []
    for data, labels in dataset:
        mask = (labels == -1)
        for digit in digit_list:
            mask += (labels == digit)
        masked_data = data.masked_select(mask.view(-1, 1, 1, 1)\
                          .expand(data.shape)).view(-1, 1, 28, 28)
        masked_labels = labels.masked_select(mask)
        selected_data.append(masked_data)
        selected_labels.append(masked_labels)

    ##########################################################################
    #   Finally concatenate into minibatches of the right size               #
    ##########################################################################

    running_count = 0
    running_mb, running_labels = [], []
    full_dataset = [] # we output this at the end
    len_selected = len(selected_data)
    zip_list = list(zip(selected_data, selected_labels))
    iter_no = 0
    data, labels = zip_list[0]
    while True:
        finished_zip_el = data.shape[0] + running_count <= batch_size
        if finished_zip_el:
            # Case when the full iter element goes into the running mb
            running_mb.append(data)
            running_labels.append(labels)
            running_count += data.shape[0]
        else:
            # Case when not all of the iter element needs to go in
            to_select = batch_size - running_count
            running_mb.append(data[:to_select])
            running_labels.append(labels[:to_select])
            running_count += to_select
            data = data[to_select:]
            labels = labels[to_select:]

        # Concatenate running elements and output to final dataset
        if running_count == batch_size or\
           (finished_zip_el and iter_no + 1 == len_selected):
            new_mb = torch.cat(running_mb)
            new_mb_label = label_map(torch.cat(running_labels).squeeze().long())
            full_dataset.append((new_mb, new_mb_label))
            running_count = 0
            running_mb, running_labels = [], []

        # Terminate if end of the line, otherwise get new batch
        if finished_zip_el:
            iter_no += 1
            if iter_no == len_selected:
                return full_dataset
            data, labels = zip_list[iter_no]

