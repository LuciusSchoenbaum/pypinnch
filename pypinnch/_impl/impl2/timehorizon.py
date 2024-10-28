






class TimeHorizon:
    """
    Class that manages canonical time parameters
    for a time "horizon" for a given class, at a given time.
    This determines precisely how far ahead in time
    the class needs to see.
    (no strides, no leaps).
    In particular, it

        - manages no counters
        - never changes after initialization.

    The problem's time horizon is managed by :any:`Problem`
    and from there it passes to the top engine
    during initialization.

    Parameters:

        tinit (scalar): initial time
        stepsize (optional scalar): stepsize
        textent (optional scalar): extent in time, length of time interval

    """

    def __init__(self, tinit, stepsize = None, textent = None):
        # canonical timesteps
        self.tinit = tinit
        self.tfinal = tinit if textent is None else tinit + textent
        self._stepsize = stepsize
        self._nstep = 0


    def init(self, textent):
        """
        Initializes by calculating tfinal if it is not known.

        Arguments:

            textent (scalar):
                The scalar value tfinal - tinit.

        """
        tmp = self.tinit
        self._nstep = 0
        # todo branch is deprecated
        # if isinstance(self._stepsize, NStep):
        #     self._nstep = self._stepsize()
        #     # > set stepsize
        #     self._stepsize = textent/float(self._nstep)
        #     tmp += textent
        if isinstance(self._stepsize, float):
            # stepsize is float
            # > set nstep
            while tmp <= self.tinit + textent - 1e-14:
                tmp += self._stepsize
                self._nstep += 1
        elif self._stepsize is None:
            self._nstep = None
            tmp = None
        else:
            raise ValueError
        self.tfinal = tmp
        log = self._check(tmp)
        return log


    def init_via_stepsize(self, stepsize):
        """
        Assumes: tinit and tfinal are already set.

        Arguments:

            stepsize (scalar):
                The distance in time taken during a single step.

        """
        self._stepsize = stepsize
        self._nstep = 0
        tmp = self.tinit
        while tmp < self.tfinal:
            tmp += stepsize
            self._nstep += 1
        log = self._check(tmp)
        return log


    def init_via_nstep(self, nstep):
        """
        Assumes: tinit and tfinal are already set.

        Arguments:

            nstep (integer):
                The desired number of steps dividing up textent = tfinal - tinit.

        """
        self._nstep = nstep
        self._stepsize = (self.tfinal - self.tinit)/nstep
        # calculate just to check arithmetic
        tfinal = self.tinit + self._stepsize*nstep
        log = self._check(tfinal)
        return log


    def shift(self, shamt):
        """
        Shift (via linear translation) the time horizon by a constant amount.
        """
        self.tinit += shamt
        self.tfinal += shamt


    def Nstep(self):
        """

        Returns:

            integer:
                The total number of steps traversed during the time horizon's
                full time interval.

        """
        return self._nstep


    def extent(self):
        return self.tfinal - self.tinit


    def stepsize(self):
        return self._stepsize


    def range(self):
        return self.tinit, self.tfinal


    def _check(self, tfinal):
        if tfinal is None:
            return ""
        log = f"Time Horizon:\n"
        log += f"tfinal = {tfinal:.4f} = {self.tinit:.4f} + {self._nstep} x stepsize {self._stepsize:.2f}.\n"
        log += f"(requested: {self.tfinal:.4f})\n"
        return log


    def __str__(self):
        out = ""
        out += f"tinit: {self.tinit}\n"
        out += f"tfinal: {self.tfinal}\n"
        out += f"stepsize: {self._stepsize}\n"
        out += f"nstep: {self._nstep}\n"
        return out



