





from .. import Probe

import numpy as np

from ..._impl.impl2.numpy import get_taw_curve

from mv1fw import sortdown


class TAWeightingClinic(Probe):
    """
    A clinic for TAWeighting strategy.
    Generate visualizations of the weights being applied.

    Print a representation of the weights
    for each stage of the taweighting.
    Optionally, also print the actual
    weights being generated as a function
    of the actual time values in the batch.
    This can generate a large number of plots.

    Parameters:

        residual (boolean):
            Whether to perform clinic operations after the
            residual loss is calculated.
        every_residual: (integer):
            How often to generate a plot in residual functionality.
        stopafter_residual (integer):
            Stop generating plots after this many calls
            in residual functionality.

    """

    def __init__(self, residual = False, every_residual = 1, stopafter_residual = 100):
        super().__init__()
        self.residual = residual
        self.every = every_residual
        self.stopafter = stopafter_residual
        self.counter = 0
        self.gradual_pct_delta = 10.0
        self.gradual_pct_counter = 0.0

    def on_iter(self, B, BB):
        """
        Check strats.taweighting to see if a new stage is reached.
        If so, generate a visualization of the stage's weights,
        tagged with the stage's integer counter value.

        """
        taw = B.phase.strategies.taweighting
        if taw.stage == 1:
            # reset the local pct counter
            self.gradual_pct_counter = 0.0
        if taw.advanced:
            # unset the flag
            taw.advanced = False
            make_plot = False
            if taw.gradual_mode():
                pct = taw.stage/taw.nstages*100
                if taw.stage == taw.nstages:
                    self.log(f"ti {B.ti} tj {B.tj} tr {B.tr} iter {BB.iteration}: taw is 100% completed.")
                    make_plot = True
                elif pct - self.gradual_pct_counter >= self.gradual_pct_delta:
                    self.gradual_pct_counter += self.gradual_pct_delta
                    self.log(f"ti {B.ti} tj {B.tj} tr {B.tr} iter {BB.iteration}: taw is {self.gradual_pct_counter}% completed.")
                    make_plot = True
            else:
                self.log(f"ti {B.ti} tj {B.tj} tr {B.tr} iter {BB.iteration}: taw has advanced to stage {taw.stage}.")
                make_plot = True
            if make_plot:
                # plot the weighting used
                X = get_taw_curve(B)
                filename = self.cog.filename(
                    action=self,
                    handle="w",
                    driveri=B.driveri,
                    phasei=B.phasei,
                    stem="fig",
                    ending="png",
                    ti=B.ti,
                    tj=B.tj,
                    tr=B.tr,
                    it=BB.iteration,
                    level=B.L,
                )
                title = self.cog.title(
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=B.ti,
                    tj=B.tj,
                    tr=B.tr,
                    it=BB.iteration,
                    level=B.L,
                )
                t = B.phase.samplesets.icbase.t
                self.fig.series(
                    filename=filename,
                    X=X,
                    inlabel="t",
                    inidx=0,
                    outlabels=["w"],
                    outidxs=[1],
                    title=title + "w(t)",
                    xlim=(B.phase.th.tinit, B.phase.th.tfinal),
                    ylim=(-0.05, 1.05),
                    vlines=(t,t + B.phase.th.stepsize()),
                    vlineslim=(0.0,1.0),
                )




    def after_residual(self, B, BB):
        """
        Plot the weights, taking the
        values directly as-is from within the training routine.

        """
        if self.residual:
            if self.counter % self.every == 0 and self.counter < self.stopafter:
                if BB.T is not None:
                    BB.T = BB.T.detach().numpy()
                    BB.W = BB.W.detach().numpy()
                    # plot the weighting used
                    filename = self.cog.filename(
                        action=self,
                        handle="w",
                        phasei=B.phasei,
                        stem="fig",
                        ending="png",
                        ti=B.ti,
                        tj=B.tj,
                        tag=str(self.counter),
                        level=B.L,
                    )
                    self.fig.series(
                        filename=filename,
                        X=sortdown(np.hstack((BB.T, BB.W))),
                        inlabel="t",
                        inidx=0,
                        outlabels=["w"],
                        outidxs=[1],
                        title="",
                        ylim=(-0.05, 1.05),
                        vlines=(B.problem.td.t,),
                        vlineslim=(0.0,1.0),
                    )
            self.counter += 1



