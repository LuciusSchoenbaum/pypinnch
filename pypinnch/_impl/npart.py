





class NPart:
    """
    Substitute for ``SPD``, see :any:`TopLine`.
    Expresses the
    desire to subdivide the interval into N steps,
    where N is requested.

    It is allowed to multiply/divide by integers::

        ns = NPart(4)
        stepsize1 = ns # 4 steps
        stepsize2 = ns*2 # 8 steps
        stepsize3 = ns/2 # 2 steps

    Arguments:

        npart (integer):
            Number of requested subdivisions

    """

    def __init__(self, npart):
        if npart < 1:
            raise ValueError(f"npart cannot be {npart}, it must be ≥ 0.")
        self.n = npart

    def __call__(self):
        return self.n

    def __mul__(self, other):
        if not isinstance(other, int):
            raise ValueError
        return NPart(int(self.n*other))

    def __truediv__(self, other):
        if not isinstance(other, int):
            raise ValueError
        return NPart(int(self.n/other))

    def __str__(self):
        return f"NPart: {self.n}"

