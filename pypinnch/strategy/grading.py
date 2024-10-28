



from .strategy_impl.strategy import Strategy

from .._impl.types import rightzeros
from .._impl.impl2.numpy import wgen
from .._impl.kit import Kit


from math import log2


class Grading(Strategy):
    """
    Strategy to apply for graded training.

    Parameters:

        label (string):

                - None: No graded training strategy is applied.
                - full: Full or eager graded training strategy is applied.
                    Use the maximal strategy of increasing the level
                    to the maximum possible at the given timestep.
                - logarithmic: Logarithmic (less eager than full) graded training
                    strategy is applied.

        steps_per_stride (integer):
            Power of 2: 4, 8, 16, etc. [timesteps].
            In training, `steps_per_stride` is the number of
            steps that are taken during each "stride" of graded training.
            After each stride, the neural network is taken down and
            a new one takes its place.
            The "depth" of drill training is log2(steps_per_stride).
        kits:
            List of instances of kits.
            None if no grading is used.
            Otherwise as many kits as depth -
            i.e., the log2 of steps per stride.
        order:
            The order of the weighting applied to the graded target.
            Default: linear (1).
            Other possible values: 2, 3.

    """

    # todo review

    def __init__(
            self,
            label = "None",
            steps_per_stride = None,
            kits = None,
            order = None,
    ):
        super().__init__(name='grading')
        self.steps_per_stride = steps_per_stride
        self.w = None
        self.order = order
        self.kits = kits
        if label == "None":
            self.id = 0
        elif label == "full":
            self.id = 1
        elif label == "logarithmic":
            self.id = 2
        else:
            raise ValueError(f"Unrecognized training strategy")
        if self.id > 0 and kits is None:
            raise ValueError(f"Require to set kits (max_iterations, tolerance, ...) in grading case.")
        if self.order is None:
            self.order = 1


    def init(self, phase):
        """
        Called at the initialization of train().
        """
        level = phase.L
        depth = int(log2(phase.th.Nstep()))
        if self.id > 0:
            if self.kits is None:
                # use defaults
                self.kits = depth*[Kit()]
            else:
                if len(self.kits) != depth:
                    raise ValueError(f"Set kits (max_iterations, tolerance, ...) for all levels in graded case.")
        if level == 0:
            # Use constant weighting
            self.w = wgen(order=0)
        elif level > 0:
            # Use weighting pointed to the graded target
            self.w = wgen(
                order=self.order,
                t0 = phase.th.tinit,
                # 2^L + 1 to add a little weight to the far-future values (no deeper reason).
                dt = phase.th.stepsize*(2**level + 1)
            )


    def using(self):
        return self.id > 0


    def nexpand(self, step):
        """
        The number of levels to expand during graded training.

        Arguments:
            step (integer):

        """
        n = 0
        if self.id == 1:
            # expand to the maximum amount possible
            n = self.steps_per_stride - step
        elif self.id == 2:
            # expand logarithmically
            n = rightzeros(step, self.steps_per_stride)
        return n


    def __str__(self):
        out = super().__str__()
        if self.id != 0:
            out += f"order: {self.order}\n"
            if self.kits is not None:
                for i, kit in enumerate(self.kits):
                    out += f"Level {i}:\n{str(kit)}"
        return out



