



from ..._impl.impl2.actionmanager import ActionManager
from ...action.action_impl.action import separate_actions_probes
from ...action.info import Info
from ...strategy.strategy_impl.strategies import Strategies
from ..._impl.impl2 import RotateDict
from ..._impl.driver import Driver


class Engine:
    """
    Base type of :any:`Engine` objects.

    Arguments:

        phases (optional dict: string -> :any:`Phase`):
            a list of phases for single-process training.
        drivers (:any:`Driver` or dict: string -> :any:`Driver`):
            drivers for the engine, each driver manages one process/device,
            for multi-process (parallelized) training or for a single-process
            training using a user-defined driver.
        handle (string):
            Name of engine, or None.
            If None, the name of the engine class will be used.
        topline (:any:`TopLine`):
        background (:any:`Background`):
        problem (:any:`Problem`):
        models (:any:`Models` or list of models):
            Instance of a Model class, or list of models.
        strategies (list of :any:`Strategy`):
            strategies that apply to the engine
        actions (list of :any:`Action`):
            actions that apply to the engine
        checkpoints (list of integer):
            Set the checkpoints (affecting logging,
            loss data storing, and model checkpointing
            if Checkpoint action is set).
            Default: 10, 30, 60, 90.
            This means checkpointing occurs after 10%,
            30%, 60%, and 90% of training is finished.

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
        if phases is not None:
            if drivers is not None:
                raise ValueError(f"[Engine] phases cannot be defined if drivers are defined.")
            self.drivers = RotateDict({'D1': Driver(phases=phases)})
        else:
            # Note: drivers is a RotateDict, so it is effectively a list.
            # The ability to rotate may be used by engines for pipelining.
            if isinstance(drivers, Driver):
                # reduce syntax if there is only one driver
                self.drivers = RotateDict({'D1': drivers})
            elif drivers is None:
                raise ValueError(f"[Engine] one of phases and drivers must be defined.")
            else:
                self.drivers = RotateDict(drivers)
        # handle, describes the engine
        if handle == "" or handle is None:
            self.handle = self.__class__.__name__
        else:
            self.handle = handle
        self.topline = topline
        self.background = background
        self.problem = problem
        self.models = models
        # >> Invariant: background must be set either here or
        # via call to set_background prior to the config stage.
        self.out = None if self.background is None else ActionManager(engine=self)
        # > set strategies
        self.strategies = Strategies(strategies)
        # > set actions/probes
        self.actions = []
        self.probes = [Info()]
        if actions is not None:
            separate_actions_probes(
                actions,
                self.actions,
                self.probes,
            )
        # > set checkpoints
        self.checkpoints = [10, 30, 60, 90] if checkpoints is None else checkpoints
        self.checkpoint_loadpath = None
        # location of engine, if in a separate file
        self.file = file
        # case runs
        self.case_str = None
        self.code_str = None
        self.code = None


    def init(self):
        """
        New engine init() methods should call this via super().init(), at the top.
        """
        # > init background
        self.background.init(self.out)
        # > init topline
        self.topline.init(self)
        # > init problem
        self.problem.init(
            background=self.background,
            out=self.out,
        )
        # > init models
        if len(self.models) == 1:
            self.models[0].set_labels(self.problem.labels())
        else:
            for model in self.models:
                if model.labels is None:
                    raise ValueError(f"Model labels have not been set.")
        self.models.init()
        # > compare problem labels vs. model labels
        self._labelcheck()


    def deinit(self):
        self.problem.deinit(engine=self)


    ###################################################
    # Main Module Methods
    # (can be called in any order)


    def set_background(self, background):
        """
        Main module method. (Called by user.)
        Set the background instance, prior to config stage.
        Use this unless a background already exists prior to
        creating engine.

        Arguments:
            background (:any:`Background`):
        """
        if self.background is not None:
            raise ValueError(f"The background should only be set once.")
        self.background = background
        self.out = ActionManager(engine=self)


    def set_topline(self, topline):
        """
        Main module method. (Called by user.)
        Set the topline instance, prior to config stage.
        Use this unless a topline already exists prior to
        creating engine.

        Arguments:
            topline (:any:`TopLine`):
        """
        self.topline = topline


    ###################################################
    # Config Methods
    # (can be called in any order)


    def set_output_absolute_directory(
            self,
            abs_path,
    ):
        """
        Config method. (Called by user.)
        Set the engine's output directory to
        an absolute path.

        Arguments:

            abs_path (string):
                absolute path that exists at time of execution.

        """
        # passthrough method
        self.out.set_output_absolute_directory(
            abs_path=abs_path,
        )


    def set_output_relative_directory(
            self,
            rel_path,
    ):
        # passthrough method
        self.out.set_output_relative_directory(
            rel_path=rel_path,
        )


    def set_verbosity(self, level):
        """
        Config method. (Called by user.)
        Set the engine to a level of verbosity.
        This only affects messages to stdout ITCINOOD.
        Typical use is to make the solver run quietly,
        as it normally generates a lot of stdout messages,
        which are also recorded in the log.

        Arguments:

            level (string): a level
                Currently level can only be "quiet" ITCINOOD.
        """
        # passthrough method
        self.out.log.set_verbosity(level)


    def set_problem(self, problem):
        """
        Config method. (Called by user.)
        Set a problem during config stage.

        Arguments:
            problem (:any:`Problem`):
        """
        self.problem = problem


    def set_models(self, models):
        """
        Config method. (Called by user.)
        Set models during config stage.

        Arguments:
            models (:any:`Models`):
        """
        self.models = models


    def set_checkpoints(self, checkpoints):
        """
        Set checkpoints to execute during training.
        Default: [10, 30, 60, 90], or if this method is run
        with None as argument, no checkpoints will be executed.

        Arguments:
            checkpoints (list of integers): list of integers i, 0 ≤ i ≤ 100,
                representing percentages of training at which to checkpoint.
        """
        self.checkpoints = checkpoints if checkpoints is not None else []


    def set_case(self, case, code):
        """
        Config method. (Called by user.)

        Set a case using a QueueG Case instance.

        Arguments:

            case (optional :any:`Case`): Case instance, or None for no effect.
            code (string): the code for the case,
                typically "a1", or something similar.

        """
        if case is not None:
            self.case_str = str(case)
            self.code = code
            self.code_str = case.str_code(code)
            case.configure(engine=self, code=code)


    def set_strategies(self, strategies):
        """
        Config method. (Called by user.)
        Set/reset engine strategies.

        Arguments:

            strategies (list of :any:`Strategy`):

        """
        self.strategies = Strategies(strategies)


    def set_checkpoint_loadpath(self, filename):
        """
        Load training state from storage.
        Inverse of Checkpoint::save().

        """
        self.checkpoint_loadpath = filename

        # todo move to init routines
        # _, fname = split_path(filename)
        # self.log(f"\nRestoring model from {fname}...\n")
        # checkpoint = torch.load(filename)
        # module.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



    ###################################################


    def _labelcheck(self):
        """
        Review output labels, called once during initialization.
        Check the user's choices
        for any issues; not intended as bulletproof,
        but enforces the conditions:

        .. note::

            - Invariant: if there are multiple models, then the
                input lists of models are identical and given in identical order
                to the problem input labels.
            - Invariant: problem output labels are in 1-1 correspondence
                with set of all model output labels.
            - Invariant: each model output has a unique label.
            - Invariant: problem output labels are all unique and distinguishable.
            - Invariant: as a set, model output labels are all unique and distinguishable.

        """
        # todo these checks evolved over time...can probably be done more efficiently
        # - todo Invariant: if there are multiple models,
        #    model output labels are contiguous in the problem output labels.
        problem_outlabels = self.problem.outlabels()
        if len(problem_outlabels) != len(set(problem_outlabels)):
            raise ValueError(f"Problem output labels are not unique: {problem_outlabels}")
        model_outlabels = {}
        for model in self.models:
            lbl = model.outlabels()
            for lb in lbl:
                if lb in model_outlabels:
                    raise ValueError(f"Multiple networks cannot generate outputs with identical labels.")
                else:
                    model_outlabels[lb] = True
        for lb in self.problem.outlabels():
            if model_outlabels[lb] is None:
                raise ValueError(f"Problem output label {lb} has no associated network.")
            else:
                model_outlabels[lb] = False
        for lb in model_outlabels:
            if model_outlabels[lb]:
                raise ValueError(f"Network output label {lb} is not a problem output label.")
        # uniqueness check.
        lbls = []
        for model in self.models:
            lbls += model.lbl
        if len(set(lbls)) != len(lbls):
            raise ValueError(f"Invalid network labels {lbls}. Each network output must have a unique label.")
        if len(self.models) > 1:
            prob_inputs = set(self.problem.lbl[:self.problem.indim])
            for model in self.models:
                model_inputs = set(model.lbl[model.indim:])
                if prob_inputs != model_inputs:
                    raise ValueError(f"Model inputs must be identical to the problem inputs, but labels {model.lbl} do not match.")


    def start(
            self,
            output_absolute_directory = None,
            reference_absolute_root_directory = None,
            case = None,
            code = None,
            file = None,
    ):
        """
        Start the engine (run the solver).
        Each Engine class should call this method
        at the beginning of its overriding start() routine.

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
            file (optional string):
                Pass __file__ here if the run script is isolated from
                the engine, otherwise None.

        """
        self.out.set_output_absolute_directory(
            abs_path=output_absolute_directory,
        )
        self.background.reference_absolute_root_directory = reference_absolute_root_directory
        self.set_case(case, code)
        if file is not None:
            self.file = file
        self.out.on_start()


    def savetxt(self, filename, X):
        """
        Save text (a string) in an ad-hoc
        way in the output directory, i.e.,
        outside of a designated :any:`Action`.

        Arguments:

            filename (string):
                filename (relative or full path),
                e.g., ``foo.txt``, will be overwritten
            X (string):
                written to target file.

        """
        fname = self.out.cog.filename(
            handle=filename,
        )
        try:
            with open(fname, 'w') as f:
                f.write(X)
        except FileNotFoundError:
            self.out.log(f"[Engine:savetxt] file path not found: {filename}")
        except IOError:
            self.out.log(f"[Engine:savetxt] unable to write to {filename}")
        except Exception as e:
            raise ValueError(f"[Engine:savetxt] An unexpected error occured: {e}")


    def __str__(self):
        div = "\n-\n\n"
        out = ""
        if self.handle == self.__class__.__name__:
            out += f"engine: {self.handle}\n"
        else:
            out += f"engine: {self.handle} : {self.__class__.__name__}\n"
        out += str(self.topline)
        out += str(self.background)
        # list actions
        out += 'actions: [' + ', '.join(self.actions) + ']\n'
        # list strategies todo fix
        out += f"strategies: {str(self.strategies)}\n"
        out += div
        out += "~PROBLEM~\n"
        out += str(self.problem)
        for modeli, model in enumerate(self.models):
            out += div
            out += f"|-|-| Model {modeli+1} |-|-|\n"
            out += str(model)
        # pass control to drivers
        for driveri, driver in enumerate(self.drivers):
            out += div
            out += f"__ D r i v e r {driveri+1} __\n"
            out += str(driver)
        if self.case_str is not None:
            out += "This run was performed as part of a case study:\n"
            out += "\nCases/This Case:\n"
            out += self.case_str + "\n"
            out += self.code_str + "\n"
        return out



