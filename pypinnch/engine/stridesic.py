









from .engine_impl import Engine

from copy import deepcopy

from .._impl.types import timed



class StridesIC(Engine):
    """
    A drop-in replacement
    for :any:`Strides` that does not use
    wrong-IC training. It could be said to
    apply "correct-IC training" instead of the
    "wrong-IC training" of :any:`Strides`.
    It exists mainly for the purpose of testing
    the effects of wrong-IC training,
    with or without considering
    benefits of hardware parallelism.
    It's logic is similar, but simpler and
    more straightforward than that of :any:`Strides`.

    """

    def __init__(
            self,
            phases = None,
            drivers = None,
            handle = None,
            topline = None,
            background = None,
            problem = None,
            models = None,
            strategies = None,
            actions = None,
            checkpoints = None,
            file = None,
    ):
        super().__init__(
            phases=phases,
            drivers=drivers,
            handle=handle,
            topline=topline,
            background=background,
            problem=problem,
            models=models,
            strategies=strategies,
            actions=actions,
            checkpoints=checkpoints,
            file=file,
        )
        # todo experiment, removing this field
        # set during init()
        # self.stride = None
        # number of drivers that are retired
        self.nretired = 0


    @timed("init")
    def init(self):
        """

        Called after start() and after config stage.

        """
        # > init topline, background, models
        super().init()
        # > init neural networks
        # and propagate information down to drivers
        first = True
        for driver in self.drivers:
            driver.init(
                background=self.background,
                problem=self.problem,
                models=self.models,
                manager=self.out,
                first=first,
            )
            first = False
        # > init IC's of non-zeroth driver
        for i in range(1, len(self.drivers)):
            self.drivers[i].icbase = deepcopy(self.drivers[0].icbase)
        # > init time horizons
        stride = self.topline.stride
        stride_extent = self.problem.th.extent()/stride
        # > set all driver's time horizons (time allocations).
        th = deepcopy(self.problem.th)
        # it is probably not necessary to log this.
        thlog = th.init(textent=stride_extent)
        for i, driver in enumerate(self.drivers):
            th.shift(shamt=i*stride_extent)
            driver.th = deepcopy(th)


    @timed("deinit")
    def deinit(self):
        """

        Called at end of start().

        """
        super().deinit()
        for driver in self.drivers:
            driver.deinit()
        self.nretired = 0
        self.topline.deinit(self)


    @timed("critical_section")
    def critical_section(self):
        """

        Only train the driver with the correct ICs,
        no wrong-IC training on other drivers.

        .. note::
            When the drivers list wraps around,
            there will still be an effect similar to
            wrong-IC training, only the wrong-IC's
            will be further into the past of the
            simulation than they would be in Strides
            training.
            A way around this (in a test) is to make the
            list of drivers long enough
            that there is no wrap-around effect
            during the test.
        """
        self.drivers[self.nretired].critical_section()


    @timed("communication")
    def communication(self):
        """
        Propagate the correct ICs to the next driver.
        """
        if not self.terminus_check():
            self.drivers[self.nretired+1].icbase(
                buffer=self.drivers[self.nretired].fcbuffer,
            )


    @timed("increment")
    def increment(self, ti):
        # > update ti due to completion
        progress = self.drivers[self.nretired].th.Nstep()
        # progress = 1
        if not self.terminus_check():
            # > shift the 0th driver
            self._shift()
            # > reset the 0th driver's time horizon
            self._reset_th(ti)
            # > rotate drivers
            self.drivers.rotate()
        else:
            # > retire a driver
            self.nretired += 1
            if self.nretired > len(self.drivers):
                self.out.log(f"Too many retired drivers. Something wrong?")
        return ti + progress


    #####################################


    def terminus_check(self):
        return self.drivers[len(self.drivers)-1].terminus_check()


    def _shift(self):
        shamt = 0.0
        for driver in self.drivers:
            shamt += driver.th.extent()
        self.drivers[0].th.shift(shamt=shamt)


    def _reset_th(self, ti):
        driver0 = self.drivers[0]
        # todo out of date lapdative removed
        # todo reimpl with hasattr
        if self.strategies.ladaptive.using():
            raise NotImplementedError
        else:
            N = driver0.th.Nstep()
            Nrem = self.problem.th.Nstep() - ti
            Nout = min(N, Nrem)
            delta = Nout * driver0.th.stepsize()
        # todo arg delta is obsolete
        driver0.th.init(delta=delta)


    def start(
            self,
            output_absolute_directory = None,
            reference_absolute_root_directory = None,
            case = None,
            code = None,
            file=None,
    ):
        """
        The backbone routine of the engine.

        Arguments:

            output_absolute_directory (optional string):
                A path to an output directory, if not specified,
                the current working directory + "output" is used.
            reference_absolute_root_directory (optional string):
                If using references, this must be set to direct to the
                location where reference data will be located.
            case (optional :any:`Case`):
                QueueG Case instance for a case-driven run.
            code (optional string)
                code for a QueueG case-driven run.
            file (string):
                Pass `__file__` if the run script is isolated from
                the engine, otherwise `None`.

        """
        super().start(
            output_absolute_directory=output_absolute_directory,
            reference_absolute_root_directory=reference_absolute_root_directory,
            case=case,
            code=code,
            file=file,
        )

        self.init()

        self.out.after_init()

        self.out.gate_strideloop(driver=self.drivers[0])

        ti = 0
        Nstep = self.problem.th.Nstep()

        while ti < Nstep:

            self.out.on_stride(driver=self.drivers[0])

            self.critical_section()

            self.out.after_critical_section()

            self.communication()

            self.out.after_communication()

            # update ti, move keeper
            ti = self.increment(ti=ti)

            self.out.after_stride(ti=ti)

        #} // stride

        self.out.after_strideloop()

        self.out.on_end()

        self.deinit()

    #} // start




