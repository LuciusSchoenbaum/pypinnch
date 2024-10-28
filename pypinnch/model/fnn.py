

from .model_impl import Model
from .module import FNN as Module_FNN



class FNN(Model):
    """
    Fully-connected neural network specification,
    via human-readable strings, hosting human-readable labels
    for PINN inputs and outputs.

    Parameters:

        hidden_layer_sizes (list of integer):
            Layer sizes, in list, from entry point to exit point,
            including input and output sizes. You can use Python,
            for example::

                hidden_layer_sizes = [256, 128, 32]
                hidden_layer_sizes = 6*[512] # [512, 512, 512, 512, 512, 512]
                hidden_layer_sizes = 3*[512] + 2*[64] # [512, 512, 512, 64, 64]

        activation (string or list):
            Either a list of activations to be applied in
            order to the NN layers, or a single activation
            to be applied at all the NN layers.
        initializer (string):
            initializer for model parameters.
        regularizer:
            regularizer, e.g. None, l1, l2
        exp_final (boolean):
            If true, apply x=exp(-x) as the final step in the forward path.
            This can be applied in case an output should be always positive,
            to avoid nonphysical values.
            Default: False
        labels (string):
            Labels can be passed in here or during config stage.
            If there is one model whose labels are the problem
            labels, this will be set automatically.

    """

    def __init__(
            self,
            hidden_layer_sizes,
            activation,
            initializer,
            regularizer=None,
            exp_final=False,
            labels=None,
            encoding=None,
    ):
        super().__init__(
            labels=labels,
            encoding=encoding,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.activation = activation
        self.initializer = initializer
        self.regularizer = regularizer
        self.exp_final = exp_final
        self._Module = Module_FNN


    def init(self):
        super().init()
        layers_indim = self.get_layers_indim()
        outdim = len(self.lbl) - self.indim
        self.layers = [layers_indim] + self.hidden_layer_sizes + [outdim]
        if isinstance(self.activation, list) and not (len(self.layers) - 1) == len(self.activation):
            raise ValueError(f"Invalid list of activations.")


    def __str__(self):
        out = ""
        out += f"{self.__class__.__name__}\n"
        out += f"encoding: {self.encoding}\n"
        out += f"layers: {self.layers}\n"
        out += "signature: "
        m = self.indim
        if self.indim == 0:
            out += "t --> "
        else:
            for i in range(m-1):
                out += self.lbl[i] + ", "
            out += self.lbl[m-1]
            if self.with_t:
                out += ", t"
            out += " --> "
        n = self.layers[-1]
        for i in range(n-1):
            out += self.lbl[m+i] + ", "
        out += self.lbl[m+n-1] + "\n"
        out += f"activation: {self.activation}\n"
        out += f"initializer: {self.initializer}\n"
        out += f"regularizer: {self.regularizer}\n"
        return out

