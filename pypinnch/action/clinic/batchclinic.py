







import torch
from numpy import random

from .. import Probe


class BatchClinic(Probe):
    """
    Probe whose purpose is to review and observe batches as they
    actually appear to the training system.
    When used, the training will stop early.

    This clinic is not so much a fixed utility,
    as it is a template for doing whatever it is that needs
    to be done.

    """

    def __init__(self, nsamples = 5):
        super().__init__()
        self.nsamples = nsamples
        self.counter = 0


    def after_batch(self, B, BB):
        """
        Called after a batch is drawn.
        Can be modified as needed, as you may require
        more/less/different kinds of information while debugging.
        """
        n = random.rand()
        # 0 ≤ n < 1.
        if n >= 0.5 and self.counter < self.nsamples:
            self.counter += 1
            self.log(f"batch sample #{self.counter}/{self.nsamples} [[")
            XX = BB.XX
            QQref = BB.QQref
            XXs = BB.XXs
            self.log(f"XX = {XX}")
            self.log(f"QQref = {QQref}")

            self.log(f"XX min = {torch.min(XX)}")
            self.log(f"...({XX.shape[0]} values)...")
            self.log(f"XX max = {torch.max(XX)}")
            for i, XXi in enumerate(XXs):
                self.log(f"[{i}] XX = {XXi}")
                self.log(f"[{i}] XX min = {torch.min(XXi)}")
                self.log(f"...({XXi.shape[0]} values)...")
                self.log(f"[{i}] XX max = {torch.max(XXi)}")
            self.log(f"]] batch sample #{self.counter}/{self.nsamples}")
        if self.counter == self.nsamples:
            self.log(f"Finished sampling batches.")
            B.action_triggered_break = True
            # todo If there are multiple steps/strides this only stops the training, it does not stop the simulation,
            #  the solver goes on to more steps unless you drive early_nstep to 1.
            #  BatchClinic should do that, because
            #  otherwise batchclinic will just trivially shut down every subsequent call to Phase. (adding pointless cycles)



