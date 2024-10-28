




import torch
from torch import nn

from .module_impl import Module



class CNNDropout(Module):
    """



    """

    def __init__(
        self,
        net,
        dtype,
    ):
        super().__init__()
        initializer_W = self.initializer(net.initializer)
        initializer_b = self.initializer("zeros")
        self.layers = nn.ModuleList()
        self.activation_chain = []
        self.activation_parameters = self.populate_activation(
            activation_chain = self.activation_chain,
            net_activation=net.activation,
            net_layers=net.layers,
            dtype=dtype,
        )


    def forward(self, inputs):
        x = inputs
        # todo insert cnn layer
        act_chain = self.activation_chain
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activate(
                inputs=x,
                activation=act_chain[i],
                params=self.activation_parameters[i:i+1,:] if self.activation_parameters is not None else None,
            )
        x = self.layers[-1](x)
        return x

