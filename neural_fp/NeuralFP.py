""" Reimplementation of ICLR 2019 Submission
    Detecting Adversarial Examples via Neural Fingerprinting
    https://arxiv.org/abs/1803.03870

    Most code comes from the author's original Implementation
    see https://github.com/StephanZheng/neural-fingerprinting
"""

from __future__ import print_function

import os
import pickle
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from cifar10 import cifar_loader as cl
from loss_functions import NFLoss
import adversarial_perturbations as ap
import adversarial_attacks as aa
import utils.checkpoints as checkpoints
import utils.pytorch_utils as utils
import time
import logging
from loss_functions import RegularizedLoss
from loss_functions import PartialXentropy


class NeuralFP(object):
    """ Main class to do the training and detection """

    def __init__(self, classifier_net, num_dx,
                 num_class, dataset_name, log_dir,
                 eps=0.1, manual_gpu=None):
        """ Stores params training using fingerprints
        ARGS:
            classifier_net: Model used for training and evaluation
            num_dx: N number of delta x fingerprints
            num_class: Number of class for the dataset
            dataset_name: "cifar" or "mnist", use it to determine specific fingerprints and loss
            log_dir: Place to store generated fingerprints
            eps: Size of perturbation. default: 0.1
            manual_gpu: Use gpu or not. If not specified, will use gpu if available.
        FingerPrint:
            fp_target: num_classes x num_perturb x num_class Tensor: Every element
                       of the first axis has the same value
            fp_dx: num_dx * 1 * Channel * Width * Height numpy array: Contains
                   fingerprints for different perturbed directions

        """
        self.classifier_net = classifier_net
        self.logger = utils.TrainingLogger()
        self.dataset_name = dataset_name

        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

        self.num_dx = num_dx
        self.num_class = num_class

        # build fingerprints
        if self.dataset_name == "mnist":
            fp_dx = [np.random.rand(1, 1, 28, 28) * eps for _ in range(num_dx)]

            fp_target = -0.2357 * np.ones((num_class, num_dx, num_class))
            for j in range(num_dx):
                for i in range(num_class):
                    fp_target[i, j, i] = 0.7

            # save the fingerprints
            """ TODO: Clean args.log_dir"""
            pickle.dump(fp_dx, open(os.path.join(log_dir, "fp_inputs_dx.pkl"), "wb"))
            pickle.dump(fp_target, open(os.path.join(log_dir, "fp_outputs.pkl"), "wb"))

            self.fp_target = utils.np2var(fp_target, self.use_gpu)

        else:  # CIFAR 10
            fp_dx = ([(np.random.rand(1, 3, 32, 32) - 0.5) * 2 * eps for _ in range(num_dx)])

            # num_target_classes x num_perturb x num_class
            fp_target = -0.254 * np.ones((num_class, num_dx, num_class))

            for j in range(num_dx):
                for i in range(num_class):
                    fp_target[i, j, i] = 0.6
            fp_target = 1.5 * fp_target

            # save the fingerprints
            # pickle.dump(fp_dx, open(os.path.join(log_dir, "fp_inputs_dx.pkl"), "wb"))
            # pickle.dump(fp_target, open(os.path.join(log_dir, "fp_outputs.pkl"), "wb"))

            fp_target = utils.np2var(fp_target, self.use_gpu)

        self.fp_dx = fp_dx  # numpy array
        self.fp_target = fp_target  # torch variable

    def train(self, train_loader, test_loader, normalizer, num_epochs,
              train_loss, verbosity_epoch=1, optimizer=None, logger=None,
              regularize_adv_criterion=None, regularize_adv_scale=None):
        """ Build some Neural fingerprints given the perturbed directions. Train
            the network with new regularization to minimize the distance between
            model output and fingerprint delta y
            ARGS:
                train_loader: torch DataLoader can be Cifar10 or MNIST
                test_loader: data loader for validation
                normalizer: user defined normalizer
                num_epochs: number of training epochs
                train_loss: loss function for vanilla classification training
                optimizer: default: Adam Optimizer
                regularize_adv_criterion: NeuralFP regularization function
                regularize_adv_scale: scale between vanilla training and regularization
                logger : if not None, is a utils.TrainingLogger instance. Otherwise
                         we use this instance's self.logger object to log
                verbosity_epoch: TBD

        """

        ######################################################################
        #   Setup/ input validations                                         #
        ######################################################################
        self.classifier_net.train()  # in training mode
        assert isinstance(num_epochs, int)

        assert not (self.use_gpu and not cuda.is_available())
        if self.use_gpu:
            self.classifier_net.cuda()

        # restore fingerprints
        fp_dx = self.fp_dx
        fp_target = self.fp_target

        # setup logger
        if logger is None:
            logger = self.logger
        if logger.data_count() > 0:
            print("WARNING: LOGGER IS NOT EMPTY! BE CAREFUL!")
        logger.add_series('training_loss')

        # setup loss fxn, optimizer

        optimizer = optimizer or optim.SGD(self.classifier_net.parameters(),
                                           lr=0.01, weight_decay=1e-6, momentum=0.5)

        # setup regularize adv object
        regularize_adv_criterion = regularize_adv_criterion or nn.MSELoss()

        ######################################################################
        #   Training loop                                                    #
        ######################################################################
        start = time.time()
        for epoch in range(num_epochs + 1):
            for idx, train_data in enumerate(train_loader, 0):
                inputs, labels = train_data
                if self.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                real_bs = labels.size(0)  # real batch size
                num_class = self.num_class
                # Build Perturbed Images

                # Batch * num_dx * num_class
                fp_target_var = torch.index_select(fp_target, 0, labels)

                # inputs_net contains inputs and perturbed images
                inputs_net = inputs
                for i in range(self.num_dx):
                    dx = fp_dx[i]
                    dx = utils.np2var(dx, self.use_gpu)  # now dx becomes torch var
                    inputs_net = torch.cat(
                        (inputs_net,
                         inputs_net + dx))  # append to the end. x_net now contains x and dx of all directions

                # Now proceed with standard training
                normalizer.differentiable_call()
                self.classifier_net.train()
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                # forward step
                logits_net = self.classifier_net.forward(normalizer(inputs_net))
                output_net = F.log_softmax(logits_net, dim=1)  # softmax of the class axis

                outputs = output_net[0:real_bs]
                logits = logits_net[0:real_bs]
                logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, num_class)
                loss = train_loss.forward(outputs, labels)  # vanilla classification loss

                reg_adv_loss = 0
                # compute fingerprint loss
                for i in range(self.num_dx):
                    fp_target_var_i = fp_target_var[:, i, :]
                    logits_p = logits_net[(i + 1) * real_bs:(i + 2) * real_bs]  # logits of x+dx in one direction
                    logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs,
                                                                                                            num_class)

                    diff_logits_p = logits_p_norm - logits_norm + 0.00001
                    reg_adv_loss += regularize_adv_criterion(diff_logits_p, fp_target_var_i)

                # Print Loss
                if idx % 100 == 0:
                    print("Time is ", time.time() - start)
                    start = time.time()
                    print("reg_adv_loss:", float(reg_adv_loss))
                    print("vanilla loss:", float(loss.data))

                # set relative importance between vanilla and FP loss
                if regularize_adv_scale is not None:
                    loss = loss + regularize_adv_scale * reg_adv_loss
                elif self.dataset_name == "cifar":
                    loss = loss + (1.0 + 50.0 / self.num_dx) * reg_adv_loss
                else:
                    loss = loss + 1.0 * reg_adv_loss

                # backward step
                loss.backward()
                optimizer.step()

                # test accuracy
                if idx % 100 == 0:
                    self.classifier_net.eval()

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for test_data in test_loader:
                            val_images, val_labels = test_data
                            if self.use_gpu:
                                val_images = val_images.cuda()
                                val_labels = val_labels.cuda()

                            outputs = self.classifier_net(val_images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += val_labels.size(0)
                            correct += (predicted == val_labels).sum().item()
                    print("The Accuracy is ", 100 * correct / total, "%")

            # end_of_epoch
            if epoch % verbosity_epoch == 0:
                print("Finish Epoch:", epoch, "Vanilla loss:", float(loss))
                checkpoints.save_state_dict("NFPTraining",
                                            "ResNet50",
                                            epoch, self.classifier_net,
                                            k_highest=3)

        print('Finished Training')

        return logger


def train(batch_size=48):
    # set random seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # use resnet32
    classifier_net = cl.load_pretrained_cifar_resnet(flavor=32,
                                                     return_normalizer=False,
                                                     manual_gpu=None)
    # load cifar data
    cifar_train = cl.load_cifar_data('train', batch_size=batch_size)
    cifar_test = cl.load_cifar_data('train', batch_size=256)

    nfp = NeuralFP(classifier_net=classifier_net, num_dx=5, num_class=10, dataset_name="cifar",
                   log_dir="~/Documents/deep_learning/AE/submit/mister_ed/pretrained_model")

    num_epochs = 30
    verbosity_epoch = 5

    train_loss = nn.CrossEntropyLoss()

    logger = nfp.train(cifar_train, cifar_test, normalizer, num_epochs, train_loss,
                       verbosity_epoch)

    return logger


def test():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    # TODO: Check Normalizer's effect
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # get the model
    classifier_net = CW2_Net()
    print("Eval using model", classifier_net)

    # load the weight
    PATH = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/ckpt" \
           "/state_dict-ep_80.pth"
    classifier_net.load_state_dict(torch.load(PATH))
    classifier_net.cuda()
    classifier_net.eval()
    print("Loading checkpoint")

    # Original Repo uses pin memory here
    cifar_test = cl.load_cifar_data('val', shuffle=False, batch_size=64)
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

    # restore fingerprints
    fingerprint_dir = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/"

    # fixed_dxs = pandas.read_pickle(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"))
    # fixed_dys = pandas.read_pickle(os.path.join(fingerprint_dir, "fp_outputs.pkl"))

    fixed_dxs = np.load(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), encoding='bytes')
    fixed_dys = np.load(os.path.join(fingerprint_dir, "fp_outputs.pkl"), encoding='bytes')

    # print(fixed_dxs)
    # print(fixed_dys)

    fixed_dxs = utils.np2var(np.concatenate(fixed_dxs, axis=0), cuda=True)
    fixed_dys = utils.np2var(fixed_dys, cuda=True)

    # print(fixed_dxs.shape)
    # print(fixed_dys.shape)

    reject_thresholds = \
        [0. + 0.001 * i for i in range(0, 2000)]

    print("Dataset CIFAR")

    loss = NFLoss(classifier_net, num_dx=30, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys, normalizer=normalizer)

    logger = logging.getLogger('sanity')
    hdlr = logging.FileHandler('/home/tianweiy/Documents/deep_learning/AE/NeuralFP/log/pgd_2000_16_5_testV3.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)

    # sanity check all clean examples are valid
    print("USE PGD 2000 STEP")
    for weight in [1., 10., 100., 1000., 10000.]:
        logger.exception("Use Weight " + str(weight))

        dis_adv = []
        dis_real = []

        for idx, test_data in enumerate(cifar_test, 0):
            inputs, labels = test_data

            inputs = inputs[0].unsqueeze(0)
            labels = labels[0].unsqueeze(0)

            inputs = inputs.cuda()  # comment this if using CPU
            labels = labels.cuda()

            # build adversarial example
            delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                             'lp_bound': 16.0 / 255})

            vanilla_loss = PartialXentropy(classifier_net, normalizer)
            losses = {'vanilla': vanilla_loss, 'fingerprint': loss}
            scalars = {'vanilla': 1., 'fingerprint': -1 * weight}

            attack_loss = RegularizedLoss(losses=losses, scalars=scalars)

            pgd_attack_object = aa.PGD(classifier_net, normalizer, delta_threat, attack_loss)
            perturbation_out = pgd_attack_object.attack(inputs, labels, num_iterations=2000, verbose=False)
            adv_examples = perturbation_out.adversarial_tensors()

            assert adv_examples.size(0) is 1

            # compute adversarial loss
            l_adv = loss.forward(adv_examples, labels)
            loss.zero_grad()

            # compute real image loss
            l_real = loss.forward(inputs, labels)
            loss.zero_grad()

            dis_adv.append(l_adv)
            dis_real.append(l_real)
            # if idx % 1000 == 0:
            #    print("FINISH", idx, "EXAMPLES")
            #    print("Adversarial Percent is ", adversarial / total * 100, "%")
            #    print("False Positive is ", fpositive / total * 100, "%")

        total = len(dis_adv)

        for tau in reject_thresholds:
            true_positive = 0
            false_positive = 0

            for adv in dis_adv:
                if adv > tau:
                    true_positive += 1

            for real in dis_real:
                if real > tau:
                    false_positive += 1

            logger.exception("The Threshold is "+str(tau))
            logger.exception("True Positive is " + str(true_positive / total * 100) + '%')
            logger.exception("False Positive is " + str(false_positive / total * 100) + '%')
            # print("The Accuracy on Clean Example is ", correct / total * 100, "%")






if __name__ == '__main__':
    test()

