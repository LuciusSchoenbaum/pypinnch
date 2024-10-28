


class Bundle:
    """
    Either an :any:`ActionBundle` or :any:`ProbeBundle`.

    :meta private:
    """



class ActionBundle(Bundle):
    """
    A :any:`Bundle` of things that any :any:`Action` can see.

    To do: add more documentation!

    These counters are exposed:

    - ti:
        The counter for total timesteps (in the steps defined in the :any:`Problem` instance).
        Counts from 0: ``ti = 0`` is the first timestep.

    - tj:
        The counter for steps (in the steps defined in the current phase)
        within the current stride.
        Counts from 0: ``tj = 0`` is the first step within the stride, during the phase.

    - sj:
        The counter for timeslices during steps.
        The relationship between sj and tj is diagrammed as follows, in ASCII art::

            tj:     0      1      2      3       etc.
                 .------.------.------.------.
            sj:  0      1      2      3      4   etc.


    :meta private:
    """

    # todo
    #   The bundle manages both a ``tj`` and an ``sj`` counter,
    #   this should be fixed

    def __init__(
            self,
            engine,
    ):
        # What %ages of maxiter to execute a checkpoint.
        self.checkpoints = None
        # General tags for output artifacts at checkpoints.
        self.checkpoint_tags = None

        # > never changing among drivers or phases:

        self.engine = engine
        self.problem = None
        self.models = None
        self.out = None

        # > action triggered break: any action can set this to
        # true in order to quit at end of current trainloop
        self.action_triggered_break = False

        # > counters

        # current level
        self.L = None
        # current driver counter
        self.driveri = None
        # number of drivers
        self.ndriver = None
        # current phase counter
        # todo change name to phi
        self.phasei = None
        # total timesteps counter, updated after each stride
        # total timesteps: self.ti+self.tj
        self.ti = None
        # stride counter
        self.stride = None

        # > driver-dependent

        self.driver = None
        # reference to the modules todo access via driver.hub.modules to avoid confusion about who "has" the modules
        self.modules = None
        self.passed = None

        # > phase-dependent

        self.phase = None
        # whether the phase is the final phase
        # todo review if this is still being used
        self.final = False
        # step (within stride) counter
        self.tj = None
        # slice (within stride) counter
        self.sj = None
        # training counter
        # This counter keeps artifacts in order
        # in case they are produced once per call to train().
        # It is reset during each phase.
        self.tr = None


    def init(self):
        self.problem = self.engine.problem
        self.models = self.engine.models
        # todo driver dependent? review
        self.checkpoints = self.engine.checkpoints
        self.checkpoint_tags = [f"C{x}" for x in self.checkpoints]
        self.driveri = 0
        self.ndriver = len(self.engine.drivers)
        self.stride = 0
        self.ti = 0


    def init_strideloop(
            self,
            driver,
    ):
        # todo review - can we simplify this by just setting the self.driver
        #  in init()? if not, document why not. Clearly, the idea was
        #  to have the manager (engine) call init() and each driver
        #  calls init_strideloop(), but is this still true?
        #  NB idea is that bundles are distinct for each driver -
        #  but this is not tested yet.
        # todo UPDATE: there is confusion here but don't revise
        #  anything until we are ready to test >= 2 drivers.
        self.driver = driver
        self.driveri += 1
        # Note: this assumes that
        # the driver pattern is always what
        # it is in Strides and StridesIC,
        # if that assumption is not valid,
        # these labels may be invalid
        if self.driveri == self.ndriver + 1:
            self.driveri = 0
        # expose driver's output manager to bundles,
        # e.g. so that actions can use the logger.
        self.out = driver.out


    def init_stride(
            self,
            driver,
    ):
        # bundles expose modules to actions.
        # modules may change between strides in testing contexts.
        # todo awk, remove and remove driver argument
        self.modules = driver.hub.modules
        self.phasei = 0
        self.stride += 1
        self.final = False

    # todo deinit to move .out back to the engine .out

    def init_phase(
            self,
            phase,
    ):
        self.phase = phase
        self.phasei += 1
        if self.phasei == len(self.driver.phases):
            self.final = True
        self.tj = 0
        self.sj = 0
        self.L = 0
        self.tr = 0


    def step(self):
        return self.ti + self.tj




class ProbeBundle(Bundle):
    """
    A :any:`Bundle` of things that any :any:`Probe` can see.
    A probe can see a superset of things that an action can see.

    .. note::

        A possible source of confusion: a :any:`Probe` inherits
        from an :any:`Action`. But a :any:`ProbeBundle` does
        not inherit from a :any:`ActionBundle`. Rather, a :any:`Probe`
        receives both an action bundle, and a probe bundle,
        as separate ''bundles'' as this seems more natural in practice
        under Python's duck typing.

    """

    # todo documentation

    # todo should ProbeBundle inherit from ActionBundle, or would this be a nuisance or be error-prone?
    #  ...it's tempting to call ActionBundle "Bundle" and let ProbeBundle inherit from it.

    def __init__(self):
        # current iteration counter
        self.iteration = None
        # current kit
        self.kit = None

        # todo why not just attach hub to probe bundle? (a legacy issue)
        # batch for ic constraints
        self.XX = None
        # reference for ic constraints
        self.QQref = None
        # batches for constraints
        self.XXs = None
        # references for constraints
        self.QQrefs = None

        # residual at points in batch
        self.R = None
        # weights computed for R
        self.W = None
        # T vector used to generate weights
        # ...convenient to have.
        self.T = None

        # reduced loss for ics
        self.ic_losses = None
        # reduced loss for each constraint
        self.losses = None

        # problem.get clinic/investigation
        self.valuesin = None
        self.varlist = None
        self.values = None

