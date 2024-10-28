











import numpy as np

from .. import Probe



class WeightMonitor(Probe):
    """

    Emit plots showing the weights applied to losses
    with respect to "train time" (not the same as physical time,
    counted by timesteps and strides, but rather what is counted
    by the iterations and by training cycles).

    Can be useful for debugging and also for experimenting with
    novel weighting strategies.

    """

    # todo finish implementing

    # FUNGIBLE NOTES:
    # If it's on, it has a simple job and it should simply print *everything*,
    # because it can easily be disabled.
    # So a first draft of the probe is easy:
    # > on training session, init ledgers.
    # > on iter, at the time when you can get them,
    #     get the weights on all the losses, ledger these.
    # > after training session, store dat and store plot ayd.
    #
    # Any refinement of this procedure can be done later.


    def __init__(self):
        super().__init__()
        self.ledgers = None


    def on_time_to_do_it(self, B, BB):
        # > how big are the ledgers?
        size = 123
        # > make the ledgers
        self.ledgers = 123


    def on_we_have_weights(self, B, BB):
        # > get the weights...
        weights = 123
        # > put them in a ledger...
        self.ledgers = 456


    def on_all_done_for_now(self, B, BB):
        for ledger in self.ledgers:
            # > plot the ledger
            filename = self.cog.filename(
                action=self,
                handle="weights",
                stem="",
                ending="dat",
                driveri=B.driveri,
                phasei=B.phasei,
            )
            with open(filename, 'w') as f:
                # todo ledger has string method that gives a dat
                f.write(str(ledger))
            filename = self.cog.filename(
                action=self,
                handle="weights",
                stem="fig",
                ending="png",
                driveri=B.driveri,
                phasei=B.phasei,
            )
            title = self.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                tr=B.tr,
                # tag=f"weights t {t}"
            )
            self.fig.series(
                filename=filename,
                title=title,
                # todo
                # Xs=Xs,
                # in1=0,
                # in2=1,
                # in3=2,
                # lbl=lbl[0:2] + ["t"],
                # show=False,
            )




