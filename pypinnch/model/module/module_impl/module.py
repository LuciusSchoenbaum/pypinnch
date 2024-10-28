

from torch import nn, tensor
from torch import full, hstack
from .activation import Activation
import torch.nn.init as I


initializer_label2f = {
    "Xavier normal": I.xavier_normal_,
    "Xavier uniform": I.xavier_uniform_,
    "Kaiming normal": I.kaiming_normal_,
    "Kaiming uniform": I.kaiming_uniform_,
    "zeros": I.zeros_,
    "ones": I.ones_,
}

from .encoding_modules import (
    BasicEncodingModule,
    PositionalEncodingModule,
    GaussianEncodingModule,
)


class Module(nn.Module):
    """
    Wrapper for a PyTorch nn.Module
    that initializes adaptive activation functions.

    .. note::

        Keras vs. Torch nomenclature::

            Keras   |  Torch
            --------------------
            Glorot  |   Xavier
               He   |  Kaiming


    """
    # todo more documentation for setting the activation function

    def __init__(
            self,
            net,
            dtype,
    ):
        super().__init__()
        self.with_t = net.with_t
        self.indim = net.indim
        if net.encoding is None:
            self.encoding_module = None
        else:
            encoding = net.encoding.__class__.__name__
            if encoding == 'BasicEncoding':
                self.encoding_module = BasicEncodingModule()
            elif encoding == 'PositionalEncoding':
                self.encoding_module = PositionalEncodingModule(
                    sigma = net.encoding.sigma,
                    m = net.encoding.m,
                )
            elif encoding == 'GaussianEncoding':
                self.encoding_module = GaussianEncodingModule(
                    indim = net.indim,
                    sigma = net.encoding.sigma,
                    m = net.encoding.m,
                    b = net.encoding.b,
                    dtype = dtype,
                )
            else:
                raise ValueError(f"[Module] Unrecognized encoding type {encoding}")


    def initializer(self, label):
        if isinstance(label, str):
            return initializer_label2f[label]
        else:
            raise ValueError(f"Could not interpret initializer {label}")


    def encoding_stage(self, x):
        """
        Encode an input tensor using the encoding layer.

        Arguments:

            x (Tensor):
                input on the forward path

        Returns:

            Tensor

        """
        if self.encoding_module is None:
            out = x
        else:
            if self.with_t:
                out = hstack((self.encoding_module.forward(x[:,:self.indim]),x[:,self.indim:]))
            else:
                out = self.encoding_module.forward(x)
        return out


    def populate_activation(
            self,
            activation_chain,
            net_activation,
            net_layers,
            dtype,
    ):
        sz = 0
        if not isinstance(activation_chain, list) or len(activation_chain) > 0:
            raise ValueError(f"Caller must initialize activation chain as empty list.")
        # nactivation = len(net_layers)-1
        for i in range(len(net_layers)-2):
            label = net_activation[i] if isinstance(net_activation, list) else net_activation
            activation_chain.append(Activation(label=label, ninputs=net_layers[i+1]))
            if activation_chain[i].adaptive:
                sz0 = len(activation_chain[i].initial_values)
                sz = max(sz, sz0)
        if sz > 0:
            # > initialize the (useful) activation parameters.
            # Implementation: You can see that (in general)
            # there are unused model parameters registered.
            # This might raise an issue for scalability, perhaps.
            # However, layer size, layer width,
            # and # of adaptive parameters per activation function
            # are typically small, say between 1 and 100, or no bigger than 1k.
            # Nevertheless, this code should be scrutinized after more experience.
            initial_ap = []
            for act in activation_chain:
                initial_ap.append(act.initial_values)
            # > register adaptive activation parameters in the model.
            # After this step, they are exposed to the optimizer.
            activation_parameters = nn.Parameter(
                tensor(
                    initial_ap,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            activation_parameters = None
        return activation_parameters


    def evaluate_on_input(self, x):
        """
        Unceremoniously evaluate the model on a set of
        inputs, without modifying the model
        or affecting the backprop facilities in any way.

        This method is provided to make scripts cleaner and
        more readable. The steps to
        "evaluate model M on inputs X" aren't so hard but still
        error prone. This also provides a level of indirection for
        input reformatting/encoding.

        Arguments:

            x:
                inputs, compatible in shape with the module.
                Alternatively, a pair (X, t); t will be extended and stacked with X.
                The latter is the same calling signature used by a Solution method.

        Returns:

            y: the output generated by the model.

        """
        # todo deprecate? cf. driver.evaluate
        if isinstance(x, tuple):
            X = x[0]
            t = x[1]
            x_ = hstack((X, full((X.shape[0], 1), t, dtype=X.dtype, device=X.device)))
        else:
            x_ = x
        # Switch to eval mode, this switches off
        # gradient computation and switches off
        # dropout (if any), and ... (?)
        self.eval()
        # I need to detach from the graph when
        # the module is in eval mode, oh well.
        y = self.forward(x_).detach().requires_grad_(False)
        # Switch back to train mode
        self.train()
        return y


    def activate(self, inputs, activation, params):
        """
        Execute an activation function (typ. after a layer),
        accounting for the possibility of adaptive activation.

        Arguments:

            inputs: inputs to activation function.
            activation: activation function.
            params: parameters of the activation function.

        Returns:

            x
        """
        x = inputs
        act = activation
        x = act(x, params) if act.adaptive else act(x)
        return x


    def num_parameters(self):
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


