




from .strategy_impl.strategy import Strategy




def no_strategy(
        losses_ic,
        Lic,
        losses_c,
        Lc,
        epoch,
):
    """
    All lambdas are 1.
    """
    lambdas_ic = len(losses_ic)*[1.0]
    lambdas_c = len(losses_c)*[1.0]
    return lambdas_ic, lambdas_c



class LAWeighting(Strategy):
    """
    Loss-aware weighting strategies are aware of the
    loss on all constraints and IC constraints when the
    weight is determined. Like other types of weighting,
    they also have the freedom to change behavior on the
    basis of iteration/epoch counters.

    The loss-aware weight is combined with an ordinary
    weight to generate the final weight on a constraint::

        L0 = ... # compute true loss
        loss = lambda*L0 # weighted loss

    No weight ``lambda`` should be greater than one.

    A strategy callable takes the form of this example::

        def strategy(
            losses_ic,
            Lic,
            losses_c,
            Lc,
            epoch,
        ):
            lambdas_ic = len(losses_ic)*[0.0]
            lambdas_c = len(losses_c)*[0.0]
            # calculate ...
            return lambdas_ic, lambdas_c

    Parameters:

        strategy (optional callable):

    """

    def __init__(
            self,
            strategy = None,
    ):
        super().__init__(name='laweighting')
        if callable(strategy):
            self.strategy = strategy
        elif strategy is None:
            self.strategy = no_strategy


    def init(self, phase):
        pass


    def get(self, losses_ic, Lic, losses_c, Lc, epoch):
        """
        Called by a :any:`Phase` during training,
        to retrieve the weights on losses set by the loss-aware
        computation defined by the user script.

        Arguments:

            losses_ic (list of scalar): losses broken out, for each IC constraint
            Lic (scalar): sum of all losses in losses_ic
            losses_c (list of scalar): losses broken out, for each constraint
            Lc (scalar): sum of all losses in losses_c
            epoch (integer): present epoch

        Returns:

            pair of list of scalar

        """
        lambdas_ic, lambdas_c = self.strategy(
            losses_ic=losses_ic,
            Lic=Lic,
            losses_c=losses_c,
            Lc=Lc,
            epoch=epoch,
        )
        return lambdas_ic, lambdas_c

