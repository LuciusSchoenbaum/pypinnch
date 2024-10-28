




class ExponentialWeight:
    """
    Exponential Weight Scheduling is a strategy
    whereby a weight is allowed to decrease exponentially
    with the number of epochs, from all data sources.

    For example, you could coordinate a weight with the evolution of the
    learning rate, so that the weight decreased once it is "safe" to do so
    because the learning rate is not large enough to throw the solver
    out of its current local maximum. Then the intention is to shift the
    attention of the solver.

    The weighting formula's parameters are W0 > 1 and zeta > 0.
    The formula for the weight is
        W(e) = 1 - (W0-1)*zeta^e
    This decreases the weight exponentially by an factor
    zeta, from a specified initial value, allowing it to decrease
    smoothly to one.

    Parameters:

        W0 (scalar): initial weight
        zeta (scalar): decay rate from W0 to 1
        niter (optional integer):
            If None, weight is updated after each epoch of
            training data. Otherwise, the weight is updated
            after each iteration.

    """

    # todo pretty print in docs

    def __init__(
            self,
            W0,
            zeta,
            niter=None,
     ):
        self.W0 = W0
        self.zeta = zeta
        if self.zeta >= 1.0 or self.zeta <= 0.0:
            raise ValueError(f"Invalid zeta factor.")
        if self.W0 < 0.0:
            raise ValueError(f"Initial weight must be >= 0.")
        self.niter = niter

    def __call__(self, epoch, iteration):
        if self.niter is None:
            return 1.0 + (self.W0 - 1.0)*(self.zeta**epoch)
        else:
            return 1.0 + (self.W0 - 1.0)*(self.zeta**(iteration//self.niter))





