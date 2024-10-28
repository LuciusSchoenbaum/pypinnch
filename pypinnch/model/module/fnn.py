


from torch import nn
from torch import exp
from .module_impl import Module



class FNN(Module):
    """
    Fully-connected neural network.

    Parameters:

        net:
            Instance of FNN network.
        dtype:
            data type specified by driver.

    """

    def __init__(
            self,
            net,
            dtype,
    ):
        super().__init__(
            net=net,
            dtype=dtype,
        )
        initializer_W = self.initializer(net.initializer)
        initializer_b = self.initializer("zeros")
        self.layers = nn.ModuleList()
        for i in range(1, len(net.layers)):
            self.layers.append(
                nn.Linear(
                    net.layers[i - 1],
                    net.layers[i],
                    dtype=dtype
                )
            )
            initializer_W(self.layers[-1].weight)
            initializer_b(self.layers[-1].bias)
        # todo not implemented: regularizer
        self.regularizer = net.regularizer
        self.activation_chain = []
        self.activation_parameters = self.populate_activation(
            activation_chain = self.activation_chain,
            net_activation=net.activation,
            net_layers=net.layers,
            dtype=dtype,
        )
        self.exp_final = net.exp_final


    def forward(self, inputs):
        x = self.encoding_stage(inputs)
        act_chain = self.activation_chain
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activate(
                inputs=x,
                activation=act_chain[i],
                params=self.activation_parameters[i:i+1,:] if self.activation_parameters is not None else None,
            )
        x = self.layers[-1](x)
        if self.exp_final:
            x = exp(-x)
        return x

