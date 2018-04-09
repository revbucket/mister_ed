# mister_ed 

This repository is intended to be a well-groomed pytorch mirror of [cleverhans](https://github.com/tensorflow/cleverhans). There's a rich literature on exploiting vulnerabilities in the robustness of neural nets, though the following resources make for a good introduction to the subject:
- [Towards Deep Learning Models Resistant to Adversarial Attacks - Madry et. al](https://arxiv.org/abs/1706.06083)
- [Tricking Neural Networks: Create your own Adversarial Examples - ML@Berkeley Blog](https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/)
- [Intriguing properties of neural networks - Szegedy et. al](https://arxiv.org/abs/1312.6199)
- [Adversarial Examples: Attacks and Defenses for Deep Learning- Yuan et. al (late 2017 survey paper)](https://arxiv.org/pdf/1712.07107.pdf)

The remainder of this README will focus on how to set up `mister_ed` to build your own adversarial examples/defenses and offer a brief tour of the contents of the repository. The hope is that by leveraging `mister_ed`, you'll be able to sidestep much of the machinery used in adversarial examples and get straight to building new attacks and defenses. 

## Setting up `mister_ed`

### Dependencies
This library uses [Pytorch](http://pytorch.org/) to perform differentiation and other common computation required by machine learning models. Follow the link above to perform installation. It is also recommended that you leverage GPU's to speed up computation, though generating adversarial attacks (particularly on MNIST/CIFAR10) are typically less expensive than classical neural net training.

### Installation 
The best way to set up `mister_ed` right now is to clone this git repository. 
```
git clone https://github.com/revbucket/mister_ed
```
Feel free to fork and play with the code. 

### Config + CIFAR10 setup 
To get started immediately, you'll need to ensure that you have access to a pretrained network, and a dataset. We'll use CIFAR10 as an example dataset. Configuration parameters are stored in `mister_ed/config.json`. The important parameters to set up right now are `dataset_path` and `model_path`. 
- `dataset_path`: if you already have datasets on your machine, simply set this to the directory where they live. Datasets can be downloaded using standard [`pytorch.torchvision` methods](http://pytorch.org/docs/master/torchvision/datasets.html).
- `model_path`: if you already have pretrained pytorch models on your machine, simply set this to the directory where they live. Pretrained models are saved as files ending in `.th`, using the standard `torch.save(model.state_dict(), ...)` method.

To get you going as quickly as possible, run the `mister_ed/scripts/setup_cifar.py` script to do the following:
1. Ensure all dependencies are installed correctly 
2. Ensure that CIFAR data can be accessed locally 
3. Ensure that a functional classifier for CIFAR has been loaded. By default, [pretrained CIFAR10 resnets](https://github.com/akamaster/pytorch_resnet_cifar10) from Yerlan Idelbayev are used.

## Brief Overview of Library Contents
Leveraging python's OOP nature, attacks, defenses, loss functions, and evaluations are wrapped up objects, which are all defined in the top level directory of `mister_ed`. Helper functions, and application specific files are stored in lower level directories (e.g. utilities for image processing and general pytorch helpers are contained in the `mister_ed/utils` directory, and CIFAR10-specific architectures and dataloaders are contained in the `mister_ed/cifar10` directory). 

The important classes to understand are `DifferentiableNormalize`, `RegularizedLoss`, `AdversarialAttack`, `AdversarialAttackParameters`, `AdversarialEvaluation`.
- `DifferentiableNormalize` (`mister_ed/utils/pytorch_utils.py`): Simple functional object to scale tensors to/from zero-mean unit-variance versions of themselves. Keeping track of scaling is important for image manipulation and images are handled in the 0.0 to 1.0 range as much as possible.
- `RegularizedLoss` (`mister_ed/loss_functions.py`): Loss functional object that is designed to have regularization terms built in. Each individual term of a loss function can be modeled by a `PartialLoss` subclass. See `mister_ed/prebuilt_loss_functions.py` as an example of how to build `RegularizedLoss` objects. The hope is that great strides can be made in adversarial attacks by clever choices of loss functions, so being able to concisely generate new loss functions is a key aim for `mister_ed`.
- `AdversarialAttack` (`mister_ed/adversarial_attacks.py`): Main wrapper for performing adversarial attacks. By default, subclasses of this just need to be instantiated with knowledge of a classifier net and how to normalize images to zero-mean unit-variance. The parent class contains various helper methods, while each subclass contains the code to perform each type of attack. The `attack()` method simply takes attack-specific parameters and a single minibatch.
- `AdversarialAttackParameters` (`mister_ed/adversarial_training.py`): While the `AdversarialAttack` subclasses are meant to be kept general with respect to attack-specific parameters, instances of this class have these parameters passed on instantiation, such that a minibatch of attacks can be generated by only passing the minibatch data. This is useful in both training and evaluation 
- `AdversarialTraining` (`mister_ed/adversarial_training.py`): This is a wrapper to train a model using adversarial examples. Attacked examples are generated on-the-fly as specified by an `AdversarialAttackParameters` object provided on training start. 
- `AdversarialEvaluation` (`mister_ed/adversarial_evaluation.py`): This is a wrapper to evaluate the efficacy of adversarial attacks on a specified network. The two main methods here are `evaluate_ensemble`, which outputs accuracy of a provided ensemble of attacks against a network; and `full_attack`, which builds attacked examples (against the provided classifier net) and outputs these examples to a `.npy` file.


## Tutorial: Creating your first adversarial attacks

## Tutorial: Creating your first adversarially trained network 

## Tutorial: Evaluating
