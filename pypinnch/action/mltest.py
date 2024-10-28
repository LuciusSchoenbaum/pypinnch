


from .action_impl import Action


class MLTest(Action):
    """
    ML testing.
    This refers to computation of loss
    on a "test" set. This is standard practice
    in data-driven ML and is
    also used by some PINN solvers.


    .. note::
        For PINNs, loss has a mathematically precise meaning:
        the average per-sample-point loss
        is an approximation of the L2 loss of the residual
        of a constraint. Therefore, for a PINN,
        there's a Simple Interpretation of Loss (SIOL):
        "less loss is better."
        The SIOL is valid when dimensionality of
        data and outputs is not high.
        This is the case for PINNs as long as
        problem dimensionality is small enough,
        and we believe this is the case up to input dimension 3 or 4,
        including time.
        So although we reserve judgment about higher dimensions,
        for many kinds of PINN problems we believe that
        a test sample set is not necessary in order
        to observe loss convergence.

    """

    def __init__(self):
        super().__init__()
        self.ml_test_data = 123



    def after_train(self, B):
        # do ML testing.
        # todo
        pass




