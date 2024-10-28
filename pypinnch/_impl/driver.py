# File driver.py Created by Lucius Schoenbaum, June 24, 2023
# based on drill.py Created by Lucius Schoenbaum, April 21, 2023





from torch import (
    empty as torch_empty,
    hstack as torch_hstack,
    from_numpy as torch_from_numpy,
)
from copy import deepcopy
# import sys # sizeof
from scipy.interpolate import griddata


from .driverconfig import DriverConfig
from .types import timed, get_beg
from .hub import Hub
# from .nstep import NStep

from ..action.action_impl.action import separate_actions_probes

from ..sampler import ICBase, Buffer


from mv1fw.fw import (
    XFormat,
)
from mv1fw import (
    parse_fslabels,
)



class Driver:
    """
    Provides support for multi-phase training
    using :any:`Phase` instances.
    This class manages the step loop,
    which may be further divided into strides.
    A stride can undergo multiple phases of
    training.
    There are essentially three loops:
    the outermost loops over strides,
    then there is a loop over phases,
    then (finally) there is a loop over steps.

    Parameters:

        config (:any:`DriverConfig`):
            driver config information.
        phases (dict of string: :any:`Phase`):
            Assign one or more phases of training.
        strategies (optional list of :any:`Strategy`):
            Strategies that apply to the driver.
        actions (optional list of :any:`Action`):
            Actions that are applied by the driver.

    """

    # todo drivers in multiple threads via multiprocessing

    def __init__(
            self,
            phases,
            config = DriverConfig(),
            strategies = None,
            actions = None,
    ):
        self.strategies = strategies
        self.actions = []
        self.probes = []
        separate_actions_probes(
            actions if actions is not None else [],
            self.actions,
            self.probes,
        )
        self.config = config
        self.phases = phases
        self.phase = None
        self.L = 0
        self.hub = None
        self.out = None
        # time horizon, managed by engine
        self.th = None
        # base timeslice
        self.icbase = None
        # fc buffer for communication
        self.fcbuffer = Buffer()
        # set during phase_init and unset during phase_deinit,
        # to help manage callbacks. todo review
        self.during_phase = False
        # a copy of the problem's time horizon
        self.problem_th = None
        # a copy of the problem's fslabels
        self.problem_fslabels = None
        # unset at start of each phase
        # todo rename after this is settled
        self.the_model_is_not_ready_yet = True
        # final phase label
        self.final_plb = None


    @timed("init")
    def init(
            self,
            background,
            problem,
            models,
            manager,
            first,
    ):
        """
        (Called by :any:`Engine`)

        Initialize neural networks.

        Arguments:

            background (:any:`Background`):
            problem (:any:`Problem`):
            models ():
            manager ():
            first (boolean):

        :meta private:
        """
        self.out = manager
        self.config.init(
            background=background,
        )
        self.problem_fslabels = problem.fslabels
        self.problem_th = deepcopy(problem.th)
        # Create modules.
        # NOTE: the modules are created once per run, here and now:
        # For each driver creates a module for every model.
        # They are not recreated or reinitialized/reset between strides (or between steps).
        models_ = models if models is not None else []
        modules = []
        lbls = []
        indims = []
        begs = []
        for model in models_:
            module = model.generate_module(dtype=self.config.fw_type)
            module.to(self.config.device)
            modules.append(module)
            lbls.append(model.lbl)
            indims.append(model.indim)
            begs.append(get_beg(problem.lbl, problem.indim, model.lbl, model.indim))
        self.hub = Hub(
            modules = modules,
            lbls = lbls,
            indims = indims,
            begs = begs,
            # todo deprecated field
            # device = self.config.device,
            fw_type = self.config.fw_type,
        )
        self.the_model_is_not_ready_yet = True

        if self.phases:
            for plb in self.phases:
                self.final_plb = plb
            ###########################
            # > double check the final phase
            final_phase = self.phases[self.final_plb]
            # double-check the user, to ensure
            # final phase's stepsize is problem's stepsize,
            # otherwise results will not meet problem's specification.
            final_phase.step_multiple = 1
            # double-check the user, to ensure
            # the Result action is added to the final phase.
            # todo - it's tricky because of the options you may pass in to Result.
            ###########################

        ###########################
        # problem access to driver,
        # mainly so that user callbacks can
        # have access to modules and other artifacts:
        # single-driver case:
        problem._driver = self
        # multi-driver case:
        # Each driver needs a copy of the problem
        # print("size of problem: ", sys.getsizeof(problem))
        # self.problem = deepcopy(problem) # todo review/doc - why?
        # self.problem = problem
        # expose instance of the output manager todo review/doc - why?
        # self.problem.out = self.out
        # todo - v. 030 - review driver's need to have problem,
        #  and try to remove self.problem, and this write to problem.out.
        #
        ###########################

        # > initialize phases
        for plb in self.phases:
            phase = self.phases[plb]
            phase.init(
                problem=problem,
            )
        # > initialize icbase
        if problem.with_t:
            # initialize IC's from problem-specified sample.
            # This is done only by the zeroth driver, and copied
            # to other drivers by the engine.
            self.icbase = ICBase()
            if first:
                self.icbase.init(
                    dtype=self.config.fw_type,
                    problem=problem,
                    # Use the last phase's SPL to generate the ic_sample
                    SPL=self.phases[self.final_plb].SPL,
                )


    def deinit(self):
        """
        Called by the engine in start().

        :meta private:
        """
        # free memory
        self.hub = None


    def deinit_stride(self):
        if self.icbase is not None:
            # know: self.phase has the final phase.
            # know: final phase did not deinit.
            # copy icbase to fcbuffer
            self.fcbuffer(self.phase.samplesets.icbase)
        # free memory
        self.phase.deinit()
        self.during_phase = False


    def terminus_check(self):
        if self.icbase is not None:
            # todo test
            # todo documentation
            return self.th.tfinal == self.problem_th.tfinal


    #############################################


    @timed("init_phase")
    def init_phase(self, phase):
        """
        This is called at the beginning of each phase, "on phase".

        Arguments:
            phase (:any:`Phase`):

        """
        self.L = 0
        self.during_phase = True
        self.phase = phase
        self.phase.init_phase(
            icbase=self.icbase,
            config=self.config,
            hub=self.hub,
            manager=self.out,
            th=self.th,
        )


    @timed("expand")
    def expand(self, step):
        # the number of levels to expand
        N = self.phase.strategies.grading.nexpand(
            step = step,
            steps_per_stride = self.phase.th.Nstep(),
        )
        for _ in range(N):
            self.out.on_expand()
            # todo self.phase.samplesets.expand()
            for c in self.phase.samplesets.cs:
                c.expand()
            self.L += 1


    @timed("train_and_contract")
    def train_and_contract(self):
        while self.L > 0:
            self.out.on_train()
            self.phase.train(level=self.L)
            self.out.after_train()
            self.out.on_contract()
            # todo self.phase.samplesets.contract()
            for c in self.phase.samplesets.cs:
                c.contract()
            self.L -= 1


    @timed("advance")
    def advance(self):
        self.phase.advance()
        # now the model is ready for certain output tasks
        self.the_model_is_not_ready_yet = False


    def kludge_advance(self):
        """
        A very tricky issue required this fix to get unstuck.
        """
        self.phase.kludge_advance()


    def critical_section(self):
        """
        The main routine of the driver,
        during there is never
        communication with other drivers.

        NOTE: when the step loop ends,
        the sample sets of all phases have been
        advanced to an "off the end" state.
        Currently, ITCINOOD, this state is discarded:
        the sample sets are created anew with each
        call to start. (Only icbase and fcbuffer survive.)
        In the future, this may change.

        :meta private:
        """

        self.out.gate_phaseloop()

        for plb in self.phases:
            phase = self.phases[plb]

            self.init_phase(phase=phase)

            self.out.on_phase(plb=plb, phase=phase)

            self.out.gate_steploop()

            for step in range(phase.th.Nstep()):
                self.out.on_step()

                # graded training entry point
                if phase.strategies.using('grading'):
                    self.expand(step=step)
                    self.train_and_contract()
                # final training session
                # Invariant: L = 0
                self.out.on_train()
                passed = self.phase.train(level=self.L)
                self.out.after_train()

                # > advance the phase in time
                self.out.on_advance()
                self.advance()

                self.out.after_step(passed=passed)

                self.kludge_advance()

            #} // step

            self.out.after_phase()

            if plb != self.final_plb:
                self.phase.deinit()

        #} // phase

        self.out.after_phaseloop()

        self.deinit_stride()

    #} // critical_section


    #######################################################


    def evaluate(
            self,
            X,
            force_evaluate = False,
    ):
        """
        Evaluate the modules on the input X = [x1, x2, ..., xn, t],
        or evaluate input X = [x1, x2, ..., xn] at timestep t.

        Arguments:

            X (:any:`XFormat`):
                Data to be formatted by drawing on the model/ICs,
                containing input data for all
            force_evaluate (boolean):
                Force the algorithm to evaluate on the zeroth step,
                instead of evaluating using IC. STIUYKB. Default: False

        Returns:

            XU (:any:`XFormat`):
                tensor of inputs and outputs [x1, x2, ..., xn, u1, u2, ..., um]

        """
        p_lbl, p_indim, p_with_t = parse_fslabels(self.problem_fslabels)
        lbl, indim, with_t = parse_fslabels(X.fslabels())
        # > check indim, for sanity (xor..)
        if indim != p_indim or ((with_t and not p_with_t) or (p_with_t and not with_t)):
            raise ValueError(f"Cannot evaluate (number of inputs {indim}, expected {p_indim})")
        X0 = X.X()
        t = X.t()
        end = X0.shape[1] - X.reserve
        # todo review and improve - cf. evaluate_output() documentation,
        #  which explains the planned revision
        UU, _ = self.evaluate_output(X=X, force_evaluate=force_evaluate)
        Xout = torch_hstack((X0[:,:end], UU.to(X0.device)))
        fslabelsout = self.problem_fslabels
        return XFormat(
            X=Xout,
            t=t,
            fslabels=fslabelsout,
        )


    def evaluate_output(
            self,
            X,
            force_evaluate = False,
    ):
        """
        Evaluate the input data and return all the corresponding problem
        output label data as a raw pytorch tensor.

        .. note::

            [Concerning an update that should be carried out soon.]
            This less structured method is actually the original Driver.evaluate()
            method before XFormat was added. It is called by Solution.evaluate(),
            and by the new Driver.evaluate(),
            but nowhere else ITCINOOD. The new Driver.evaluate() is
            simply a placeholder hiding an incomplete update.
            Instead of that, it would be worthwhile
            instead of always delivering the output "UU" in bulk, like this method,
            to deliver U for just one label (because this is what Solution.evaluate() needs)
            or else all the problem output labels (because this is what Problem methods need).
            If you look at the code below, it's not *entirely* simple but
            we can probably do that, and
            then we can use XFormat.append to perform Driver.evaluate(),
            and we do not need any such "evaluate_output" at all, and Driver.evaluate()
            will be sufficient.
            This would finally provide a clean data flow
            in the Result: from input data, to Solution, to Reference,
            to plotting/validation methods,
            and the entire thing would be quite friendly and easy to maintain.
            But in order to have that we have to decide how to handle the
            tricky IC case in this code, which I don't have time to work on at the moment.

        Arguments:

            X (:any:`XFormat`):
            force_evaluate (boolean):
                Force the algorithm to evaluate on the zeroth step,
                instead of evaluating using IC. STIUYKB, this is probably
                only needed if you are computing moments to go
                in constraints. Default: False

        Returns:

            UU (Pytorch tensor), lbl (list of string): tensor and labels for the tensor.

        """
        lbl, indim, with_t = parse_fslabels(self.problem_fslabels)
        outdim = len(lbl) - indim
        X0 = X.X()
        t = X.t()
        if self.the_model_is_not_ready_yet and not force_evaluate:
            # It is a time dependent problem, and
            # we are at the IC timeslice (sj == 0).
            # Invariant: t is time at IC timeslice.
            Xbase = self.phase.samplesets.icbase.X
            if indim == 0:
                # Most likely, X.shape[0] == 1
                shape0 = X0.shape[0]
                u_ = Xbase[0:1,:]
                UU = torch_empty((shape0, Xbase.shape[1])).to(device=self.config.device)
                for i in range(shape0):
                    UU[i:i+1,:] = u_
            else:
                # > interpolate Xbase outputs to X
                # t is constant, so it can be removed
                # from the interpolation call
                points = []
                for i in range(indim):
                    points.append(Xbase[:,i])
                Xhost = X0.cpu()
                UU = griddata(
                    points=tuple(points),
                    # todo this assumes the first output ic's are the same as the outdim labels. -> XFormat
                    values=Xbase[:,indim:indim+outdim],
                    xi=Xhost[:,:indim],
                    method="linear",
                ).reshape((-1, outdim))
                UU = torch_from_numpy(UU).to(device=self.config.device)
        else:
            # > make space
            UU = torch_empty((X0.shape[0], outdim)).to(device=self.config.device)
            for ni, module in enumerate(self.hub.modules):
                model_lbl = self.hub.lbls[ni]
                model_indim = self.hub.indims[ni]
                model_beg = self.hub.begs[ni]
                # todo branch on t is None inside the call (and pass indim) for the sake of cleaner code?
                if self.icbase is None:
                    # time-independent case
                    Ui = module.evaluate_on_input(X0[:,:indim])
                else:
                    if t is None:
                        Ui = module.evaluate_on_input(X0[:,:indim+1])
                    else:
                        Ui = module.evaluate_on_input((X0[:,:indim], t))
                beg = model_beg - indim
                end = beg + len(model_lbl[model_indim:])
                UU[:,beg:end] = Ui
        return UU, lbl[indim:]


    def __str__(self):
        div = "\n-\n\n"
        out = ""
        # out = f"type: {self.__class__.__name__}\n"
        # list actions
        out += "actions: ["
        if len(self.actions) == 0:
            out += "None"
        else:
            out += str(self.actions[0])
            for i in range(1,len(self.actions)):
                out += ", " + str(self.actions[i])
        out += "]\n"
        out += f"strategies: {str(self.strategies)}\n"
        # config
        out += str(self.config)
        # phases
        for plb in self.phases:
            phase = self.phases[plb]
            out += div
            out += f"Phase {plb}\n"
            out += str(phase)
        return out




