









from .engine_impl import Engine

from .._impl.types import timed




class TimeIndependent(Engine):
    """
    A time-independent engine
    that supports domain decomposition parallelism.

    """
    # Domain decomposition is planned but not fully
    # implemented yet ITCINOOD.

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
        names = ['causalweighting', 'taweighting', 'grading']
        for driver in self.drivers:
            for plb in driver.phases:
                phase = driver.phases[plb]
                for name in names:
                    if phase.strategies.using(name):
                        raise ValueError(f"Cannot train {self.__class__.__name__} engine with {name} strategy.")


    @timed("init")
    def init(self):
        """

        Called after start() and after config stage.

        """
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


    @timed("deinit")
    def deinit(self):
        """

        Called at end of start().

        """
        super().deinit()
        for driver in self.drivers:
            driver.deinit()
        self.topline.deinit(self)


    @timed("critical_section")
    def critical_section(self):
        """

        """
        # todo for domain decomposition, this loop is parallelized
        for driver in self.drivers:
            driver.critical_section()


    @timed("communication")
    def communication(self):
        """
        Propagate the BCs to the neighboring drivers.
        """
        # todo for domain decomposition, asynchronously trained drivers communicate.
        #  either a lot of work or a lot of assumptions are required
        pass


    #####################################


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

        # todo strides now correspond to breaks for IPC (communication).
        #  whenever there is one driver, there is no need to have more than one stride.
        if len(self.drivers) > 1:
            raise ValueError(f"Multiple drivers not supported yet.")
        else:
            # todo get nstrides from Topline
            nstrides = 1

        for ti in range(nstrides):

            self.out.on_stride(driver=self.drivers[0])

            self.critical_section()

            self.out.after_critical_section()

            self.communication()

            self.out.after_communication()

            self.out.after_stride(ti=ti)

        #} // stride

        self.out.after_strideloop()

        self.out.on_end()

        self.deinit()

    #} // start




