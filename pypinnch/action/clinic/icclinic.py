




from .. import Probe






class ICClinic(Probe):
    """
    A clinic to peek at the ICs
    prior to each slice (step, including initial step).

    """

    def __init__(self):
        super().__init__()
        self.X = None
        self.counter = 0

    # def on_step(self, B):
    #     self.log(f"t = {B.problem.base.t}")
    #     self.log(B.problem.base.X)



    def gate_iterloop(self, B, BB):
        self.log(f"phase sample set:")
        self.log(f"t = {B.phase.samplesets.base.t}")
        self.log(B.phase.samplesets.base.X)
        # for bc in B.phase.samplesets.cs:
        #     # sorry for this
        #     cyl = bc.cyl
        #     self.log("basesample:")
        #     self.log(cyl.basesample)
        #     self.log("timesample:")
        #     self.log(cyl.timesample)



    def after_batch(self, B, BB):
        self.log(f"training points for ICs: ")
        self.log(BB.XX)
        self.log(f"reference points for ICs: ")
        self.log(BB.QQref)
        self.log(f"END")
