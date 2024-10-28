





import numpy as np

from .. import Action



class PerformanceMonitor(Action):
    """
    Create a digest about the execution time of the run.

    """

    # todo this class should use action callbacks to
    #  set timers for subroutines.
    #  An older idea was to use timingstore to do this
    #  (and always do it; rather than wrap it as an action,
    #  which in hindsight is better).
    #  should PerformanceMonitor use the timingstore?
    #  do it "the old fashioned way"? ... ?

    def __init__(self):
        super().__init__()


    def on_end(self, B):
        # todo
        info = ""
        info += f"batchsize: {B.phase.batchsize}\n"
        filename = self.cog.filename(
            action=self,
            handle="perf",
            stem="txt",
            driveri=B.driveri,
            phasei=B.phasei,
        )
        with open(filename, 'w') as f:
            f.write(info)



