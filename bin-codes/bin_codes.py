""" Holds activation layers   """

from __future__ import print_function
import torch.nn as nn



class SequentialSelector(nn.Module):
    """ NN that keeps track of ReluActivation statuses """
    def __init__(self, original_net, selector_class=nn.modules.activation.ReLU):
        """ Takes in an input neural net and will keep track (on forward) of
            the status of all the ReLUs.
            Needs to only be composed of linear layers and ReLus
        """
        super(SequentialSelector, self).__init__()
        self._safety_dance(original_net, selector_class)
        self.original_net = original_net
        self.selector_class = selector_class
        self.new_seq = self._flatten(original_net)

    def _safety_dance(self, original_net, selector_class):
        """ Asserts that all layers are linear or ReLU's """


        assert selector_class == nn.modules.activation.ReLU
        assert isinstance(original_net, nn.Sequential)

        for el in original_net:
            if isinstance(el, nn.Sequential):
                self._safety_dance(el, selector_class)
            else:
                assert isinstance(el, (nn.Linear, nn.ReLU))

    def _flatten(self, original_net):
        def _inner_loop(net, output_list):
            for el in net:
                if isinstance(el, nn.Sequential):
                    output_list = _inner_loop(el, output_list)
                else:
                    output_list.append(el)
            return output_list

        flat_list = _inner_loop(original_net, [])
        new_seq = nn.Sequential()
        for i, el in enumerate(flat_list):
            new_seq.add_module(str(i), el)
        return new_seq

    def forward_activations(self, x):
        intermed = x
        bincodes = []
        for layer in self.new_seq:
            intermed = layer(intermed)
            if isinstance(layer, self.selector_class):
                bincodes.append(intermed.clone())
        return bincodes, intermed

    def forward(self, x):
        return self.forward_activations(x)[1]

    def bincodes(self, x, binary=True):



        activations = torch.stack([_.view(-1) for _ in
                                   self.forward_activations(x)[0]])

        if binary:
            return activations > 0
        else:
            return activations


