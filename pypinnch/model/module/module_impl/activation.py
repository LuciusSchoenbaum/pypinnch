


import torch.nn.functional as F
from torch import (
    sin as torch_sin,
)


activation_label2sigma = {
    "relu": F.relu,
    "ReLU": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "sin": torch_sin,
    "selu": F.selu,
    "elu": F.elu,
    "silu": F.silu,
    "hardswish": F.hardswish,
}




class Activation:
    """
    Activation function with field for parameters, used in case
    of an adaptive/trainable activation. Supports general
    techniques/ideas/experimentation with adaptive (i.e., trainable)
    activation functions.

    Example: a tahn, sigmoid, relu, etc. activation is defined via the line::

        activation = "tanh"
        activation = "sigmoid"
        activation = "relu" # or ReLU

    Example: a L-LAAF ReLU (see below) with the scaling factor n = 10
    is defined with the argument (to the network)::

        activation = "L-LAAF-10 relu"

    Locally adaptive activation functions, L-LAAF and N-LAAF:
    In [1] L-LAAF is introduced along with neuron-wise version, N-LAAF.
    From the experiments reported in [1], it is not clear that valuable
    additional benefit is gained from the more computationally expensive
    N-LAAF. It is noteworthy, however, that from the software engineering
    perspective, however, both N-LAAF and L-LAAF are remarkably
    inexpensive when any modern ML framework is used. (It's a tiresome
    old saw of mine that we should never lose sight of how remarkable this
    sort of thing is.)

    [1] A. D. Jagtap, K. Kawaguchi, & G. E. Karniadakis. Locally adaptive activation
    functions with slope recovery for deep and physics-informed neural networks.
    Proceedings of the Royal Society A, 476(2239), 20200334, 2020.

    """

    def __init__(
            self,
            label,
            ninputs = None,
    ):
        self.initial_values = []
        self.adaptive = False
        # > define field sigma, vectorized function: x, ap --> y
        if label is None or label == "None":
            self.sigma = lambda x, ap: x
        elif isinstance(label, str):
            # simple parsing logic
            if label.find("LAAF") >= 0:
                self.adaptive = True
                # Expected: [char]-LAAF or [char]-LAAF-[digits]
                split0 = label.split()
                if len(split0) <= 1:
                    raise ValueError(f"Parsing error")
                else:
                    split1 = split0[0].split("-")
                    if len(split1) <= 1:
                        raise ValueError(f"Parsing error")
                    LAAF_type = split1[0]
                    n = float(split1[2]) if len(split1) == 3 else 1.0
                    label0 = split0[-1]
                    if LAAF_type == "N":
                        if ninputs is None:
                            raise ValueError(f"Need number of inputs to define N-LAAF")
                        a0s = ninputs*[1./n]
                        self.initial_values += a0s
                        sigma0 = activation_label2sigma[label0]
                        self.sigma = lambda x, ap: sigma0(x*ap[0:1,:]*n)
                    elif LAAF_type == "L":
                        a0 = 1./n
                        self.initial_values.append(a0)
                        sigma0 = activation_label2sigma[label0]
                        self.sigma = lambda x, ap: sigma0(x*ap[0:1,0:1]*n)
                    else:
                        raise ValueError(f"Unrecognized LAAF type {LAAF_type} with activation {label}")
            # elif label.find("something else") >= 0:
                # a similar activation function construction can be added here.
                # self.adaptive = True
                # self.sigma = foo, bar
            else:
                # vanilla torch function
                sigma0 = activation_label2sigma[label]
                self.sigma = lambda x, ap: sigma0(x)
        else:
            raise TypeError(f"Could not interpret activation {label}")



    def __call__(
            self,
            x,
            activation_parameters = None,
    ):
        ap = activation_parameters
        return self.sigma(x, ap)




