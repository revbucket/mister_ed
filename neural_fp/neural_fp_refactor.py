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
import time
import torch.optim as optim

import utils.checkpoints as checkpoints
import utils.pytorch_utils as utils

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
        assert self.dataset_name in ['mnist', 'cifar']

        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

        self.num_dx = num_dx
        self.num_class = num_class

        fp_dx, fp_target = self._build_fingerprints(num_dx, num_class, eps,
                                                    log_dir)
        self.fp_dx = fp_dx
        self.fp_target = fp_target

    def _build_fingerprints(self, num_dx, num_class, eps, log_dir):
        """ Builds the fingeprints, saves them, and returns them """

        ######################################################################
        #   Build Fingerprints based on dataset                              #
        ######################################################################

        if self.dataset_name == "mnist":

            # WHY DON'T WE SUBTRACT HALF HERE?
            fp_dx = [np.random.rand(1, 1, 28, 28) * eps for _ in range(num_dx)]

            fp_target = -0.2357 * np.ones((num_class, num_dx, num_class))
            for j in range(num_dx):
                for i in range(num_class):
                    fp_target[i, j, i] = 0.7

        elif self.dataset_name == "cifar":
            fp_dx = ([(np.random.rand(1, 3, 32, 32) - 0.5) * 2 * eps
                       for _ in range(num_dx)])

            # num_target_classes x num_perturb x num_class
            fp_target = -0.254 * np.ones((num_class, num_dx, num_class))

            for j in range(num_dx):
                for i in range(num_class):
                    fp_target[i, j, i] = 0.6
            fp_target = 1.5 * fp_target
            fp_target = utils.np2var(fp_target, self.use_gpu)

        else:
            raise Exception("Unknown dataset %s" % self.dataset_name)


        ######################################################################
        #   Save and return fingerprints                                     #
        ######################################################################

        fp_dx_save = os.path.join(log_dir,
                                  'fp_%s_inputs_dx.pkl' % self.dataset_name)
        fp_targ_save = os.path.join(log_dir,
                                    'fp_%s_outputs.pkl' % self.dataset_name)

        pickle.dump(fp_dx.cpu().numpy(), open(fp_dx_save, 'wb'))
        pickle.dump(fp_target.cpu().numpy(), open(fp_targ_save, 'wb'))

        self.fp_dx = fp_dx  # numpy array
        self.fp_target = fp_target  # torch variable


    def _compute_fingerprint_loss(self, inputs, original_outputs,
                                  normalizer, fp_dx, fp_target):

        # First get the original outputs into the right form
        og_output_norm = utils.batchwise_norm(original_outputs, 2, dim=0)
        og_direction = original_outputs / (og_output_norm + 1e-10)

        # Next thing: compute the input + dx for each dx in fp_dx
        fingerprint_xs = [inputs + fingerprint for fingerprint in fp_dx]

        # Then compute capital F's for each fingerprint
        capital_fs = []
        for x_plus_dx in fingerprint_xs:
            fp_out = self.classifier_net(normalizer(x_plus_dx))
            fp_norms = utils.batchwise_norm(fp_out, 2, dim=0)
            fp_direction = fp_out / (fp_norms + 1e-10)
            capital_fs.append(fp_direction - og_direction)

        # Finally compare to the fingeprint targets and take summed l2 norm
        total_loss = None
        for i, capital_f in enumerate(capital_fs):
            fp_target_i = fp_target[:, i, :]
            this_loss = (capital_f - fp_target_i).pow(2)
            for _ in range(1, this_loss.dim()):
                this_loss =  this_loss.sum(1)
            if total_loss is None:
                total_loss = this_loss
            else:
                total_loss += this_loss
        return total_loss


    def train(self, train_loader, test_loader, normalizer, num_epochs,
              train_loss, verbosity_epoch=1, optimizer=None, logger=None,
              fp_scale=1.0):
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

        # restore fingerprints into tensors on right device
        try:
            fp_dx = [torch.from_numpy(_) for _ in self.fp_dx]
            fp_target = [torch.from_numpy(_) for _ in self.fp_target]
        except:
            pass

        if self.use_gpu:
            fp_dx = [_.cuda() for _ in fp_dx]
            fp_target = [_.cuda() for _ in fp_target]

        # setup logger
        if logger is None:
            logger = self.logger
        if logger.data_count() > 0:
            print("WARNING: LOGGER IS NOT EMPTY! BE CAREFUL!")
        logger.add_series('training_loss')

        # setup loss fxn, optimizer

        optimizer = optimizer or optim.SGD(self.classifier_net.parameters(),
                                           lr=0.01, weight_decay=1e-6,
                                           momentum=0.5)

        ######################################################################
        #   Training loop                                                    #
        ######################################################################
        start = time.time()
        for epoch in range(num_epochs + 1):
            for idx, train_data in enumerate(train_loader, 0):

                # Batch setup
                inputs, labels = train_data
                if self.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()


                # First compute the misclassification loss
                original_outputs = self.classifier_net(normalizer(inputs))
                misclass_loss = train_loss.forward(original_outputs, labels)


                # Next do the fingerprint loss
                fp_loss = self._compute_fingerprint_loss(inputs,
                                                         original_outputs,
                                                         normalizer, fp_dx,
                                                         fp_target)

                loss = misclass_loss + fp_scale * fp_loss


                # Print Loss
                if idx % 100 == 0:
                    print("Time is ", time.time() - start)
                    start = time.time()
                    print("fingerprint_loss:", float(fp_loss))
                    print("vanilla loss:", float(misclass_loss))

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





