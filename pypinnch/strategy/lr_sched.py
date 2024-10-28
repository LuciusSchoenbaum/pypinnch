








from .strategy_impl.strategy import Strategy

import torch



# helper
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



class LRSched(Strategy):
    """
    LR Scheduler (learning rate scheduler).

    .. note::
        LR curve (wrt iteration) is plotted by :any:`LossCurves`.

    Parameters:

        label (string):

            - None or ConstantLR:
                train using constant learning rate,
                initialized at start of training
            - exp or ExponentialLR:
                train using torch's ExponentialLR(gamma):
                on step, ``lr *= gamma``.

    """

    def __init__(
            self,
            label = "None",
            niter = None,
    ):
        super().__init__(name='lr_sched')
        if label == "None" or label == "ConstantLR":
            self.id = 0
            self.label = "Constant"
        elif label == "exp" or label == "Exponential" or label == "ExponentialLR":
            self.id = 1
            self.label = "Exponential"
        else:
            raise ValueError(f"Unrecognized training strategy")
        self.kit = None
        self._lrsched = None
        self.niter = niter


    def init(self, phase):
        self.kit = phase.strategies.optimizer.kit


    def get(self, level, optimizer, kit):
        """
        Get a new lr scheduler, given an optimizer
        and a kit, and a graded level.

        Arguments:

            level: graded level
            optimizer: optimizer
            kit:

        """
        # todo change name from "get", which is misleading.
        #  Same for optimizer.get().
        # todo level is not used ITCINOOD.
        self.kit = kit
        if self.id == 0:
            pass
        else: # self.id == 1:
            # Formula is:
            # lr *= gamma
            # on a step.
            # ...You must initialize the param_group object,
            # there may be only one of these.
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = self.kit.learning_rate
                param_group['lr'] = self.kit.learning_rate
            self._lrsched = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=self.kit.gamma,
                # last_epoch=-1, # lr0 is acquired from optimizer
                last_epoch=1,
                verbose=False,
            )


    def step(self, optimizer, phase, iteration = None):
        # todo review, it should not write into phase, phase can do that
        if self.id == 0:
            pass
        else: # self.id == 1:
            if self.niter is None:
                # > step on end of epoch
                if iteration is None:
                    self._lrsched.step()
                    # update the kit that everything reads/sees
                    phase.strategies.optimizer.kit.learning_rate = get_lr(optimizer=optimizer)
                else:
                    pass
            else:
                if iteration is None:
                    pass
                else:
                    if iteration % self.niter == 0:
                        # update the kit that everything reads/sees
                        phase.strategies.optimizer.kit.learning_rate = get_lr(optimizer=optimizer)



