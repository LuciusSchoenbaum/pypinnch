





from .strategy_impl.strategy import Strategy

import torch

from copy import deepcopy



lbfgs_high = {
    "lr": 1,
    "max_iter": 120,
    # maximal number of function evaluations per optimization step
    # DeepXDE uses 15000 via tensorflow
    # default: max_iter*1.25
    "max_eval": 2*100*100,
    # small value used by DeepXDE
    "tolerance_grad": 1e-8,
    # patient value used by DeepXDE
    "tolerance_change": 0,
    # If OoM error, decrease this.
    # algorithm requires additional param_bytes * (history_size + 1) bytes (!)
    "history_size": 150,
    # perform line searches (at extra performance cost)
    "line_search_fn": "strong_wolfe",
    # pytorch does not expose a max_ls value but is set to 25.
    # DeepXDE uses 50.
}

lbfgs_low = {
    "lr": 1,
    # falls between deepxde and pytorch default
    "max_iter": 60,
    # maximal number of function evaluations per optimization step
    # DeepXDE uses 15000 via tensorflow
    # default: max_iter*1.25
    "max_eval": 100*100,
    # small value used by DeepXDE
    "tolerance_grad": 1e-8,
    # patient value used by DeepXDE
    "tolerance_change": 0,
    # If OoM error, decrease this.
    # algorithm requires additional param_bytes * (history_size + 1) bytes (!)
    # the default value 100
    "history_size": 110,
    # caveat emptor, but this size seems to cause issues todo what issues?
    # history_size=200,
    # perform line searches (at extra performance cost)
    "line_search_fn": "strong_wolfe",
    # pytorch does not expose a max_ls value but is set to 25.
    # DeepXDE uses 50.
}

lbfgs_pytorch_default = {
    "lr": 1,
    "max_iter": 20,
    "max_eval": 20*1.25, # max_iter*1.25
    "tolerance_grad": 1e-7,
    "tolerance_change": 1e-9,
    "history_size": 100,
    "line_search_fn": None,
}

adamw_default = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False, # todo try?
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None
}

adamw_pytorch_default = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None
}




def get_Adam(module, kit):
    """
    Train using Adam, applied whenever level != 0.
    Adam is a high quality first-order stochastic optimization
    method with strong hardware+software support.

    """
    # todo - review pytorch args and create adam dict (?)
    out = torch.optim.Adam(
        params=module.parameters(),
        lr=kit.learning_rate,
    )
    return out



def get_LBFGS(module, kit):
    """
    L-BFGS is a second order optimization method.
    It is much more memory intensive than Adam, for example,
    and typically requires orders of magnitudes fewer
    training iterations.

    """
    lbfgs_params = lbfgs_high
    kwargs = {"params": module.parameters()}
    for key in lbfgs_params:
        kwargs[key] = kit.lbfgs[key] if key in kit.lbfgs else lbfgs_params[key]
    out = torch.optim.LBFGS(**kwargs)
    return out



def get_AdamW(module, kit, amsgrad = False):
    """
    todo documentation
    """
    kwargs = {"params": module.parameters()}
    for key in adamw_default:
        kwargs[key] = kit.adamw[key] if key in kit.adamw else adamw_default[key]
    if amsgrad:
        # todo I don't know whether other
        #  parameters should be modified
        #  when using amsgrad.
        kwargs["amsgrad"] = True
    out = torch.optim.AdamW(**kwargs)
    return out



class Optimizer(Strategy):
    """
    Optimizer chooses the optimizer, depending on the level.

    Parameters:

        label (string): optimizer type,

            - Adam
            - LBFGS
            - AdamW
            - AMSGrad

        kit (:any:`Kit`):
            kit of optimizer parameters

    """

    def __init__(
            self,
            label = "None",
            kit = None,
    ):
        super().__init__(name='optimizer')
        # todo very old code, fix
        if label == "None" or label == "Adam":
            self.id = 0
            self.label = "Adam"
        elif label == "LBFGS" or label == "L-BFGS":
            self.id = 1
        elif label == "AdamW":
            self.id = 2
        elif label == "AMSGrad":
            self.id = 3
        elif label == "version1":
            self.id = 4
        else:
            raise ValueError(f"Unrecognized training strategy")
        # A copy of the received kit for reinitializing
        self.init_kit = kit
        self.kit = None
        self._reset_kit()


    def init(self, phase):
        pass


    def _reset_kit(self):
        self.kit = deepcopy(self.init_kit)


    def using(self):
        return True


    def get(self, level, module):
        """
        Must pass in kit here,
        in case Grading strategy is being applied.

        Arguments:
            level: int
            module: nn.Module instance
        Returns:
            optimizer instance
        """
        self._reset_kit()
        if self.id == 0:
            out = get_Adam(module, self.kit)
        elif self.id == 1:
            out = get_LBFGS(module, self.kit)
        elif self.id == 2:
            out = get_AdamW(module, self.kit)
        elif self.id == 3:
            out = get_AdamW(module, self.kit, amsgrad = True)
        else:
            if level == 0:
                out = get_Adam(module, self.kit)
            else:
                out = get_LBFGS(module, self.kit)
        return out



    def __str__(self):
        out = super().__str__()
        out += str(self.kit)
        return out


