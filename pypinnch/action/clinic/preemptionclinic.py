




from .. import Probe

import numpy as np





class PreemptionClinic(Probe):
    """
    A clinic for SingleTasking (ST) CAWeighting strategy.
    Observe the epochs when the solver is preempted by the
    SingleTasking to work on IC's.

    """

    def __init__(self):
        super().__init__()
        self.X = None
        self.counter = 0

    def on_train(self, B):
        # > set up an array of values
        maxit = B.phase.strategies.optimizer.kit.max_iterations
        self.X = np.full([maxit, 3], -1.0)
        self.counter = 0

    def on_iter(self, B, BB):
        """
        Check strats.taweighting to see if a new stage is reached.
        If so, generate a visualization of the stage's weights,
        tagged with the stage's integer counter value.
        """
        caw = B.phase.strategies.caweighting
        # these are defined in all cases
        state = caw.preemption_state
        change = caw.preemption_change
        taw = B.phase.strategies.taweighting
        state1 = 1.0 if state else 0.0
        state2 = 0.0 if state else taw.stage/taw.nstages
        self.X[BB.iteration:BB.iteration+1,:3] = np.array([BB.iteration, state1, state2])
        if state:
            self.counter += 1
        if change:
            if state:
                self.log(f"ti {B.ti} tj {B.tj} tr {B.tr} iter {BB.iteration}: preemption {caw.pre_attempt_counter} triggered.")
            else:
                self.log(f"ti {B.ti} tj {B.tj} tr {B.tr} iter {BB.iteration}: preemption {caw.pre_attempt_counter} ends after {self.counter} iterations.")
                self.counter = 0


    def after_iterloop(self, B, BB):
        # plot the weighting used
        filename = self.cog.filename(
            action=self,
            handle="preemption",
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
        end = 0
        maxit = B.phase.strategies.optimizer.kit.max_iterations
        while end < maxit and self.X[end,0] >= 0:
            end += 1
        self.fig.series(
            filename=filename,
            X=self.X[:end,:],
            inlabel="iter",
            inidx=0,
            outlabels=["pre", "stage"],
            outidxs=[1,2],
            title=title,
            xlim=(0, maxit),
            ylim=(-0.05, 1.05),
        )
        filename = self.cog.filename(
            action=self,
            handle="preemption",
            driveri=B.driveri,
            phasei=B.phasei,
            stem="dat",
            ending="txt",
            ti=B.ti,
            tj=B.tj,
            tr=B.tr,
            it=BB.iteration,
            level=B.L,
        )
        np.savetxt(filename, X=self.X[:end,:], header="iter,pre,stage")

