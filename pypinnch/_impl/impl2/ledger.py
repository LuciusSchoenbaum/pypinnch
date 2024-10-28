




import numpy as np


class Ledger:
    """
    Helper for managing a running list of
    data, hiding some arithmetic/counters.
    Handles data conveniently even in the case that the
    ledger is not entirely filled, for some reason.

    every is an integer. Negative value of every is
    interpreted as 1/every, this indicates a substep.

    Parameters:

        nstep:
        every:
        size:
        labels:

    """

    # Numpy is used because Ledger is used
    # in :any:`Result` for plotting/data export.

    def __init__(
            self,
            nstep,
            every,
            size,
            labels,
    ):
        # the extra 2 is paranoia, I think -
        # I didn't document when I originally wrote it.
        # cf. retrieve()
        if every < 0:
            n_per_step = -every
            nreserve = int(nstep*n_per_step) + 2
        else:
            nreserve = int(nstep/every) + 2
        self.beg = 0
        self.end = 0
        self.data = np.zeros((nreserve, size))
        self.labels = labels


    def add(self, item):
        self.data[self.end,:] = item
        self.end += 1


    def retrieve(self):
        return self.data[self.beg:self.end,:]


