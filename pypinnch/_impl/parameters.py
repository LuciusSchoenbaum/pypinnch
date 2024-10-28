



class Parameters:
    """
    Class that houses parameters used
    in the framework/setup of the :any:`Problem`.

    It is very common that problems arising
    have free coefficients and parameters,
    and these can be set up here.
    It is convenient to do this, because
    this allows them to be edited (changed)
    and then the solver can be rerun
    with no further setup being necessary.

    The simplest Parameters class only implements
    the ``ranges`` member. This member
    is a dict where scalar ranges are defined, for example::

            labels = "x, y; u"

                class Parameters(pypinnch.Parameters):

                    def __init__(self):
                        super().__init__()

                        # set up an LxL 2d box
                        # with a output ranging from -1 to 1
                        L = 1.0
                        self.ranges = {
                            'x': (0, L),
                            'y': (0, L),
                            'u': (-1.0, 1.0),
                        }

    Ranges of output values are not used by the solver,
    which relies on the :any:`Constraint` s created.
    However, the output range can be used by actions, for example, plots.
    Otherwise, the max and min values
    produced during the run are computed and used.
    It is also possible to set up "alternate" ranges for plotting,
    see :any:`Result`.

    .. note::
        When setting the ranges dict, the
        value of outputs can be set to None
        if the range is not known.

    """

    # todo test ranges dict when values are set to None ---- do this in _T

    def __init__(self):
        self.ranges = {}
        # dict: output (or solution) label to a valid pyplot colormap,
        # e.g. 'viridis', 'plasma', 'magma', there are dozens:
        # from matplotlib import colormaps
        # list(colormaps)
        self.colormaps = {}

    def tinit(self):
        return self.ranges['t'][0]

    def tfinal(self):
        return self.ranges['t'][1]

    def min(self, label):
        return self.ranges[label][0]

    def max(self, label):
        return self.ranges[label][1]

    def extent(self, label):
        m, M = self.ranges[label]
        return M - m

    def range(self, label):
        return self.ranges[label]

    def set_max(self, label, x):
        self.ranges[label] = (self.ranges[label][0], x)

    def set_min(self, label, x):
        self.ranges[label] = (x, self.ranges[label][1])

    def set_tfinal(self, x):
        label='t'
        self.ranges[label] = (self.ranges[label][0], x)

    def set_tinit(self, x):
        label='t'
        self.ranges[label] = (x, self.ranges[label][1])

    def __str__(self):
        out = ""
        # This works well most of the time
        out += str(self.__dict__)
        return out

