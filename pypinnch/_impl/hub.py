







class Hub:
    """
    Hub for a training procedure
    exposing access to lower-level objects.
    No data is managed, a Hub instance is just a "pack" of
    pointers and options.

    Parameters:

        modules:
            list of modules
        lbls:
            list of lists of labels
        indims:
            list of indims
        begs:
            list of beginnings, i.e. index in problem output labels where
            model output labels start. See driver.format() method.
        fw_type:
            framework data type, or ``fw_type`` used by the framework.

    """

    def __init__(
            self,
            modules,
            lbls,
            indims,
            begs,
            # device = None,
            fw_type = None,
    ):
        self.modules = modules
        self.lbls = lbls
        self.indims = indims
        self.begs = begs
        # self.device = device
        self.fw_type = fw_type

        # ic batch and references for ic's
        self.XX = None
        self.QQref = None
        # lists of batches, one per constraint
        self.XXs = None
        self.QQrefs = None

        # pointers for problem.get()
        # that are set in phase.
        # During loss computation, _x and _u
        # change depending on whether the loss
        # targeted is the ic loss or a loss from a non-ic constraint.
        # todo review this policy.
        self._x = None
        self._u = None

        # the timestep - passed to sample set advance() methods
        # via the hub, so that a future update might support an
        # adaptive timstep.
        self.dt = None

        # the training iteration and maximum number of iterations,
        # a simple coordinate system for the receiver of the hub
        self.iteration = None
        self.max_iterations = None




