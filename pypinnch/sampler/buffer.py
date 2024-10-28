# File buffer.py Created by Lucius Schoenbaum April 10, 2023



# todo fw
from torch import Tensor


from copy import deepcopy


class Buffer:
    """
    Houses a buffer field for a Base class.
    Create from a Base class instance, and use to
    populate a Base class instance.

    Parameters:

        base (:any:`Base`):
            Base sampler instance
    """
    # todo Buffer will be removed.

    # todo re-impl Base and Buffer as either
    #  XNoF or XFormat inheritors

    def __init__(self, base = None):
        if base is not None:
            self.X = deepcopy(base.X)
            self.t = deepcopy(base.t)
        else:
            self.X = None
            self.t = None


    def __call__(
            self,
            base,
    ):
        """
        Arguments:

            base (:any:`Base`):
                Base sampler instance
        """
        if isinstance(base.X, Tensor):
            self.X = base.X.clone().detach()
            self.t = deepcopy(base.t)
        else:
            self.X = deepcopy(base.X)
            self.t = deepcopy(base.t)

