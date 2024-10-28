



from .strategy_impl.strategy import Strategy

import torch
import numpy as np
from .._impl.impl2.numpy import wgen, maxgen, onesgen, zerosgen

from math import exp



class TAWeighting(Strategy):
    """
    TAWeighting (time-adaptive weighting) implements a handful of
    variants of the time-adaptive strategy originally proposed by
    Wight and Zhao [arXiv:2004.04542]. It can be viewed as a type of
    warmup stage, or it can be viewed as a coarse (loss-unaware)
    implementation of a causal training strategy.

    Parameters:

        label (string):

            - None: No weighting
            - Wight-Zhao (W-Z or WZ):
                Wight-Zhao weighting strategy.
                Optionally set TA, otherwise TA is evenly set
            - linearA:
                linearA weighting strategy.
                Optionally set TA, otherwise TA is evenly set
            - linear:
                Linear weighting is applied throughout.
            - parabolic:
                Parabolic weighting is applied throughout.
            - cubic:
                Cubic weighting is applied throughout.

        nstages (integer):
            A number of steps to use. Weighting proceeds
            through the number of stages progressively,
            gradually increasing the weights
            on future and far-future sample points.
            Example: nstages is 4.
            Then weighting is applied through stages
            labeled 0, 1, 2 in which weights are gradually lifted.
            Finally, a stage labeled 3 is performed in which
            there is no weighting applied, and the neural network
            is expected to fully satisfy a tolerance throughout the
            target time domain.
        beta (optional scalar):
            Parameter for "Poisson" or "even" TA function.
            Pass a positive float for Poisson, or None for even.
        TA (optional callable):
            For linearA, return the area below the weighting curve
            for the ith stage, TA(int i) --> float.
        epochs_per_stage (optional integer):
            If set, how many epochs pass before each stage advances to
            the next one. Otherwise, stage advance occurs due to
            a tolerance condition taken from the kit's tolerance field,
            which is the default behavior.
            (Default: None)
        gradual_niter (optional integer):
            Setting this parameter puts TAW in "gradual mode"
            The values (if set) of nstages and epochs_per_stage
            will be overriden. Instead of using these,
            TAW will use gradual_niter stages and advance
            the stage once per iteration. This will have the
            effect of a gradual change of the weights
            rather than a sudden change.
            (Default: None)

    """

    def __init__(
            self,
            label = "None",
            nstages = None,
            beta = None,
            TA = None,
            epochs_per_stage = None,
            gradual_niter = None,
    ):
        super().__init__(name='taweighting')
        if label == "None":
            self.id = 0
        elif label == "Wight-Zhao" or label == "W-Z" or label == "WZ":
            self.id = 10
        elif label == "linearA":
            self.id = 20
        elif label == "linear":
            self.id = 1
        elif label == "parabolic":
            self.id = 2
        elif label == "cubic":
            self.id = 3
        else:
            raise ValueError(f"Unrecognized training strategy")
        self.TA = TA
        self.beta = beta
        if gradual_niter is None:
            self.gradual_niter = None
            self.nstages = nstages
            self.epochs_per_stage = epochs_per_stage
            if self.using() and (not isinstance(self.nstages, int) or self.nstages <= 1):
                raise ValueError(f"Require nstages is an integer > 1.")
        else:
            self.gradual_niter = gradual_niter
            self.nstages = gradual_niter
            self.epochs_per_stage = None
        # initial time (constant)
        self.t0 = None
        # weighting step (constant)
        self.dt = None
        # target time (constant)
        self.target = None

        self.stage = 0
        self.w = None

        # linearA static field
        self.case2 = False

        # flag read/unset only by taweighting clinic
        self.advanced = True

        # epoch counter
        self.nepoch = 0

        # This ensures finished() returns true if not using().
        # Do not change this.
        if not self.using():
            self.nstages = 0

        # modified during training to allow TAW to see loss.
        self.loss = 0.0

        # tolerance
        self.tolerance = None


    def init(self, phase):
        """
        Set up the weighting. Called by train() method.

        .. note::

            This uses the level, which is a grading concept.
            However, this allows grading+taweighting strategy (combined strategy),
            and it works fine in the case that a grading strategy is not used, ITCINOOD.

        .. note::

            Passing None for phase is possible but should be done via the reset() method.

        """
        if phase is not None:
            # set the tolerance
            self.tolerance = phase.strategies.optimizer.kit.tolerance
        # reset the stage
        self.stage = 0
        # reset case2 for linearA
        self.case2 = False
        if self.id == 0:
            # set weighting function to constant w(t) = 1
            self.w = wgen(order=0)
        else: # self.id > 0:
            # todo review
            # initial weights are always zero. (train on ic's only)
            self.w = wgen(order=-1)

            # general purpose quantities
            if phase is not None:
                self.t0 = phase.samplesets.icbase.t
                self.target = self.t0 + phase.th.stepsize()*(2**phase.L)
                self.dt = (self.target - self.t0)/self.nstages

            # TA(i), definition of ith (stage) total area in W-Z or linearA
            if self.id >= 10:
                if self.TA is None:
                    if self.beta is None:
                        # "even" TA is requested.
                        # same for id == 10, id == 20
                        self.TA = lambda i: self.dt*i
                    else:
                        # "Poisson" TA is requested.
                        if not self.beta > 0:
                            raise ValueError(f"Invalid beta")
                        self.TA = lambda i: (self.target - self.t0) \
                                            * exp(-(self.nstages - i)/self.beta)
                else:
                    # custom TA
                    pass
            self._advance()
            # Invariant: this is only used/unset by the taweighting clinic.
            self.advanced = True
            self.nepoch = 0


    def using(self):
        """
        Whether TAweighting is being applied.

        Returns:
            boolean
        """
        return self.id > 0


    def reset(self):
        """
        Reset the TAWeighting logic.
        """
        if self.using():
            self.init(phase=None)


    def on_end_of_epoch(self):
        self.nepoch += 1


    def end_of_stage(self):
        out = False
        if self.using() and not self.finished():
            if self.gradual_niter is None:
                # Advance via epoch counting/tolerance threshold
                if self.epochs_per_stage is not None:
                    if self.nepoch % self.epochs_per_stage == 0:
                        out = True
                else:
                    if self.loss < self.tolerance:
                        out = True
            else:
                # gradual mode.
                # > always advance
                out = True
        return out


    def set_loss(self, L):
        self.loss = L


    def step(self):
        """
        Step the TAWeighting.
        """
        self._advance()
        self.advanced = True


    def finished(self):
        """
        Whether weighting has progressed to the finished state.
        Require: this call is made at the _end_ of the present stage.

        Returns:

            boolean
        """
        return self.stage == self.nstages


    def gradual_mode(self):
        """
        Whether gradual mode is being used.

        Returns:
            boolean
        """
        return self.gradual_niter is not None


    ########################################################


    def _advance(self):
        """
        Advance the weighting to the next stage.
        """
        # Invariant: self.id != 0.
        self.stage += 1
        if self.stage == self.nstages:
            # final weights are always one.
            self.w = wgen(order=0)
        else:
            if self.id >= 10:
                if self.id == 10:
                    # W-Z
                    self.w = self.wgen_WZ()
                else: # self.id == 20:
                    # linearA
                    self.w = self.wgen_linearA()
            else:
                dt = self.dt*self.stage
                if self.id == 1:
                    self.w = wgen(order = 1, t0=self.t0, dt=dt)
                elif self.id == 2:
                    self.w = wgen(order = 2, t0=self.t0, dt=dt)
                else: # self.id == 3:
                    self.w = wgen(order = 3, t0=self.t0, dt=dt)


    def wgen_WZ(self):
        def w(t):
            ones = onesgen(t)
            zeros = zerosgen(t)
            if isinstance(t, np.ndarray):
                return np.where(t - self.t0 <= self.TA(self.stage), ones, zeros)
            elif isinstance(t, torch.Tensor):
                return torch.where(t - self.t0 <= self.TA(self.stage), ones, zeros)
        return w


    def wgen_linearA(self):
        T = self.target - self.t0
        if self.case2:
            x = self.target
            y = 2.0/T * self.TA(self.stage) - 1
        else:
            x = 2*self.TA(self.stage) - self.t0
            y = 0.0
            if 2*self.TA(self.stage + 1) > T:
                self.case2 = True
        def w(t):
            s = (y - 1.0)/(x - self.t0)
            return maxgen(s*(t - self.t0) + 1.0)
        return w


    def __str__(self):
        out = super().__str__()
        out += "  "
        if self.id >= 10:
            if self.beta is not None:
                out += f"TA: Poisson(beta = {self.beta})\n"
            else:
                if self.TA is None:
                    out += f"TA: even\n"
                else:
                    out += f"TA: custom\n"
        if self.id != 0:
            out += f"  nstages: {self.nstages}\n"
        else:
            pass
        return out


