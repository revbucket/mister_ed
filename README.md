# mister_ed 

This repository is intended to be a well-groomed pytorch mirror of [cleverhans](https://github.com/tensorflow/cleverhans). There's a rich literature on exploiting vulnerabilities in the robustness of neural nets, though the following resources make for a good introduction to the subject:
- [Towards Deep Learning Models Resistant to Adversarial Attacks - Madry et. al](https://arxiv.org/abs/1706.06083)
- [Tricking Neural Networks: Create your own Adversarial Examples - ML@Berkeley Blog](https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/)
- [Intriguing properties of neural networks - Szegedy et. al](https://arxiv.org/abs/1312.6199)
- [Adversarial Examples: Attacks and Defenses for Deep Learning- Yuan et. al (late 2017 survey paper)](https://arxiv.org/pdf/1712.07107.pdf)

The remainder of this README will focus on how to set up `mister_ed` to build your own adversarial examples/defenses and offer a brief tour of the contents of the repository. The hope is that by leveraging `mister_ed`, you'll be able to sidestep much of the machinery used in adversarial examples and get straight to building new attacks and defenses. 

## Setting up `mister_ed`

### Dependencies
This library uses [Pytorch](http://pytorch.org/) >=0.4 on python 2,3 to perform differentiation and other common computation required by machine learning models. Follow the link above to perform installation. It is also recommended that you leverage GPU's to speed up computation, though generating adversarial attacks (particularly on MNIST/CIFAR10) are typically less expensive than classical neural net training.

### Installation 
The best way to set up `mister_ed` right now is to clone this git repository. 
```
git clone https://github.com/revbucket/mister_ed
```
and if you manage python packages with pip, the requirements can be installed via
``` pip install -r requirements.txt ```

Feel free to fork and play with the code. 

### Config + CIFAR10 setup 
To get started immediately, you'll need to ensure that you have access to a pretrained network, and a dataset. We'll use CIFAR10 as an example dataset. Configuration parameters are stored in `mister_ed/config.json`. The important parameters to set up right now are `dataset_path` and `model_path`. 
- `dataset_path`: if you already have datasets on your machine, simply set this to the directory where they live. Datasets can be downloaded using standard [`pytorch.torchvision` methods](http://pytorch.org/docs/master/torchvision/datasets.html).
- `model_path`: if you already have pretrained pytorch models on your machine, simply set this to the directory where they live. Pretrained models are saved as files ending in `.th`, using the standard `torch.save(model.state_dict(), ...)` method.

To get you going as quickly as possible, run the 

```python scripts/setup_cifar.py``` 
script to do the following:
1. Ensure all dependencies are installed correctly 
2. Ensure that CIFAR data can be accessed locally 
3. Ensure that a functional classifier for CIFAR has been loaded. By default, [pretrained CIFAR10 resnets](https://github.com/akamaster/pytorch_resnet_cifar10) from Yerlan Idelbayev are used.

Then there are tutorials_{1,2,3}.ipynb located in `notebooks/` that contain an overview of this repository's contents and how to get started!

