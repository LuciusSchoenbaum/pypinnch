







from torch import nn
from torch import exp

from .module_impl import Module


class WTPNN(Module):
    """

    Neural network architecture with transformer sidecar networks,
    a PINN architecture proposed in [1].
    Based on ML literature on attention mechanisms.
    Projects inputs to a higher dimensional feature space
    and transforming all layers of the main layered architecture.
    Argued in [1] that this mitigates stiffness of gradients,
    and experiments show significant PINN accuracy gains.

    [1] S. Wang, Y. Teng, & P. Perdikaris. Understanding and
    Mitigating Gradient Flow Pathologies in Physics-Informed
    Neural Networks. SIAM J. Sci. Comput., 43(5) A3055-A3081, 2021.

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
        # transformer networks
        self.t_layers = nn.ModuleList()
        # main layer chain
        self.layers = nn.ModuleList()
        # todo instead of this kludge involving "pseudo layers", implement a new populate_activation method for a new kind of activation chain, a "sidecar activation chain"
        for i in range(1, len(net.transformer_layers)-1):
            self.t_layers.append(
                nn.Linear(
                    net.layers[0],
                    net.transformer_layers[i],
                    dtype=dtype,
                )
            )
            initializer_W(self.t_layers[-1].weight)
            initializer_b(self.t_layers[-1].bias)
        for i in range(1, len(net.layers)):
            self.layers.append(
                nn.Linear(
                    net.layers[i - 1],
                    net.layers[i],
                    dtype=dtype,
                )
            )
            initializer_W(self.layers[-1].weight)
            initializer_b(self.layers[-1].bias)
        # todo not implemented
        self.regularizer = net.regularizer
        self.activation_chain = []
        self.t_activation_chain = []
        self.activation_parameters = self.populate_activation(
            activation_chain=self.activation_chain,
            net_activation=net.activation,
            net_layers=net.layers,
            dtype=dtype
        )
        self.t_activation_parameters = self.populate_activation(
            activation_chain=self.t_activation_chain,
            net_activation=net.transformer_activation,
            net_layers=net.transformer_layers,
            dtype=dtype
        )
        self.exp_final = net.exp_final


    def forward(self, inputs):
        x = self.encoding_stage(inputs)
        Ts = []
        for i, layer in enumerate(self.t_layers):
            T = layer(x)
            T = self.activate(
                inputs=T,
                activation=self.t_activation_chain[i],
                params=self.t_activation_parameters[i:i+1,:] if self.t_activation_parameters is not None else None,
            )
            Ts.append(T)
        U = Ts[0]
        V = Ts[1]
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activate(
                inputs=x,
                activation=self.activation_chain[i],
                params=self.activation_parameters[i:i+1,:] if self.activation_parameters is not None else None,
            )
            # x is now paper's Z.
            # Recall * can be used for elementwise multiplication.
            x = (1.0 - x) * U + x * V
        x = self.layers[-1](x)
        if self.exp_final:
            x = exp(-x)
        return x


