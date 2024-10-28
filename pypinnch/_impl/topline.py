





from .impl2 import TimeHorizon
# from .nstep import NStep
from .npart import NPart

from copy import deepcopy


class TopLine:
    """
    Class to house settings that
    are commonly edited in the final run-up
    as a simulation is staged and run.

    Parameters:

        dryrun (boolean):
            Whether to run with outputs ignored,
            in order to test a few iterations of training
            for obvious bugs, device status, etc.
        step (integer):
            The number of steps to take in a single stride.
        stride (integer):
            The number of strides to take during a full training period.
        SPD: (optional scalar or :any:`NPart`):
            If set, it defines the **step partition delta**,
            the distance (which becomes regular,
            i.e. regularly spaced) between time
            values where sample points are defined.

            This value must be set if the :any:`Problem` defines
            :any:`Moment` field. Otherwise moments
            cannot be calculated and the solver does not run.
            If the :any:`Problem` does not define any :any:`Moment`
            (this would be noticeable by looking at the script
            that defines it) then SPD affects how sampling is
            performed on the time axis, but otherwise has
            no effect.

            A logic diagram might help explain the possibilities::

                if SPD is defined:
                    if there are moments, or if there are not moments:
                        # time is evenly divided and the number
                        # of time values per timestep
                        # is defined by SPD value.
                else:
                    if there are moments:
                        # error.
                    if there are not moments:
                        # the SPL value is used
                        # to define the sample space density
                        # in the time dimension,
                        # exactly like the other dimensions.

            Note: You may define ``SPD`` using :any:`NPart`::

                SPD = NPart(4), # five time values per step
                SPD = NPart(6), # seven time values per step

            There is one more time value than the NPart
            because you are specifying the number of
            "gaps" or "partition cells" and the partition
            of the timestep always includes the endpoints.
            Similarly, if you define SPD explicitly (using
            a scalar value), then there will be the minimum
            possible number of regular cells of length ``SPD``
            until the full length of the timestep is exceeded.
            For example, if the timestep is size 1.0,
            and the SPD is set to 0.4,
            and (at a given time during the simulation)
            the base timeslice is at t = 4.0, then during
            the solver's training on the next time step (targeting length
            1.0), there will be these times: 4.0, 4.4, 4.8, 5.2.
            Therefore using NPart has at least two advantages:
            it guarantees a timeslice on the target "next step",
            if there is one, and it automatically scales
            if the timestep is modified. However, this
            automatic change might not always be desirable.
            (Default: None, i.e., no SPD is defined)
        substep (optional integer):
             Global substep value.
             This value is propagated to every :any:`Solution`
             whose substep field is not set explicitly.
             It is always positive, and works like a denominator of a fraction,
             so::

                substep = 1

            will evaluate solutions once per step, and::

                substep = 4

            will evaluate solutions four times per step, etc.
             (Default: 1)
        early_nstep (optional integer):
            if not None, the simulation will stop
            after a specified number of steps have been executed.
            This will override the ``early_nstride``, forcing early_nstride == 1.
            The number of steps (early_nstep) must be ≤ the number
            of requested steps per stride. This is not a logically required
            condition, but it is required for the generated output to
            line up one-to-one with output from a "full" run without
            an early stop, so it is enforced.
        early_nstride (optional integer):
            if not None, the simulation will stop
            after a specified number of strides have been executed.

    """
    # todo deprecated API: time_divisions replaced by (step, stride)
    # time_divisions: (optional pair of integers or :any:`NStep`):
    # Takes the form ``(stepsize, stridesize)``, where:
    #
    #     - ``stepsize``:
    #     the size of a time step during the simulation,
    #     the smallest time subdivision considered.
    # - ``stridesize``:
    # the size of a stride (set of time steps).
    #
    # Both fields can be :any:`NStep`.
    # The `time_divisions` can be `None`, but only
    # for a time-independent problem.


    # todo perhaps dryrun can be:
    #  - dryrun: as already defined
    #  - siterun: coerce maxiter to 2 (this tests batches, SPL, ...)
    #  - prerun: run with early_nstep = 1
    #  - fullrun: run exactly as defined.
    #  ...in practice, there seem to be these qualitative
    #  "types" of dryruns, but I think it is still preliminary
    #  to implement this and document it.

    def __init__(
            self,
            dryrun,
            stride = None,
            step = None,
            substep = 1,
            SPD = None,
            early_nstep = None,
            early_nstride = None,
    ):
        self.dryrun = False if dryrun is None or dryrun == False else True
        if dryrun is None or dryrun == False:
            self.early_nstep = early_nstep
            self.early_nstride = early_nstride
            if self.early_nstep is not None:
                if self.early_nstep < 1:
                    raise ValueError(f"early_nstep must be integer ≥ 1.")
                # > silently override early_nstride
                self.early_nstride = None
            if self.early_nstride is not None:
                if self.early_nstride < 1:
                    raise ValueError(f"early_nstride must be integer ≥ 1.")
                #> silently override early_nstep
                self.early_nstep = None
            # Invariant: exactly one of early_nstep, early_nstride is None, or else both are None.
        else:
            # dryrun is True: ignore early integers
            self.early_nstep = None
            self.early_nstride = None
        self.stride = stride if stride is not None else 1
        self.step = step if step is not None else 1
        self.substep = substep if substep is not None else 1
        self.time_dependent = None
        # SPD
        self.SPD = SPD
        # dryrun
        self.DRYRUN_MAXITER = 2


    def set_dryrun(self, value = True):
        """
        Config method. (Called by user.)
        Toggle a dryrun.

        Arguments:

            value (boolean):

        """
        self.dryrun = value
        if self.dryrun:
            self.early_nstep = None
            self.early_nstride = None


    def init(self, engine):
        """
        (Called by :any:`Engine`)

        For time-dependent problem,
        will initialize time horizon, which defines the
        resolution of the solver.
        If debugging, print ``thlogs``.

        :meta private:
        """
        # > type checking/sanity
        if not isinstance(self.step, int):
            raise ValueError(f"After version 0.3.1 the `step` parameter must be an integer.")
        if not isinstance(self.stride, int):
            raise ValueError(f"After version 0.3.1 the `stride` parameter must be an integer.")
        if not engine.problem.with_t:
            if self.step != 1:
                print("[Warning] The step was set, but it is not defined or used in a time-independent problem.")
                self.step = 1
            if self.substep != 1:
                print("[Warning] The substep was set, but it is not defined or used in a time-independent problem.")
                self.substep = 1
            if self.stride > 1:
                raise NotImplementedError(f"Time-indendent problem does not yet support multiple strides.")
            if self.early_nstep:
                raise ValueError(f"Cannot set early_nstep in a time-independent problem.")
        # > init
        if engine.problem.with_t:
            self.time_dependent = True
            # > the problem's time-wise layout specification
            tinit, tfinal = engine.problem.p.tinit(), engine.problem.p.tfinal()
            problem_extent = tfinal - tinit
            stride_extent = problem_extent/self.stride
            step_extent = stride_extent/self.step
        else:
            self.time_dependent = False
            tinit, tfinal = 0.0, 1.0
            problem_extent = 1.0
            stride_extent = 1.0
            step_extent = 1.0
        if self.dryrun is True:
            # > coerce step and stride to 1
            self.stride = 1
            self.step = 1
            early_problem_extent = step_extent
        elif self.early_nstride is not None:
            # > coerce stride to early_nstride
            self.stride = self.early_nstride
            early_problem_extent = self.early_nstride*stride_extent
        elif self.early_nstep is not None:
            # > coerce stride to 1
            self.stride = 1
            # > coerce step to early_nstep
            self.step = self.early_nstep
            early_problem_extent = self.early_nstep*step_extent
        else:
            early_problem_extent = problem_extent
        # > modify problem
        # The data plotters etc. will take the time from the Parameters class,
        # so in "early" time-dependent cases there will be blank empty
        # space where the solver stopped early.
        engine.problem.th = TimeHorizon(tinit, textent = early_problem_extent)
        thlog = engine.problem.th.init_via_nstep(self.step*self.stride)
        # > modify drivers
        for driver in engine.drivers:
            driver.th = deepcopy(engine.problem.th)
            if self.dryrun:
                for plb in driver.phases:
                    phase = driver.phases[plb]
                    phase.step_multiple = 1
                    phase.SPL = 32
                    phase.batchsize = 16
                    if phase.strategies.using('optimizer'):
                        phase.strategies.optimizer.kit.max_iterations = self.DRYRUN_MAXITER
                        phase.strategies.optimizer.init_kit.max_iterations = self.DRYRUN_MAXITER
                        # LBFGS safety
                        phase.strategies.optimizer.init_kit.lbfgs["max_eval"] = 4
            elif self.early_nstep:
                # > check step multiples to ensure a "faithful" short run
                for plb in driver.phases:
                    phase = driver.phases[plb]
                    if self.early_nstep%phase.step_multiple != 0:
                        raise ValueError(f"early_nstep does not divide a phase's step multiple.")
        # > initialize SPD
        if self.SPD:
            if isinstance(self.SPD, NPart):
                N = self.SPD()
                self.SPD = engine.problem.th.stepsize()/float(N)
            elif not isinstance(self.SPD, float):
                raise ValueError(f"SPD must be NPart instance or float, not {type(self.SPD)}.")
            engine.problem.SPD = self.SPD
            # Invariant: either engine.problem.SPD is float (SPD was set),
            # or engine.problem.SPD is None (SPD was not set).
        # > init global substep
        for labels in engine.problem.solutions:
            sol = engine.problem.solutions[labels]
            if sol.substep is None:
                sol.substep = self.substep
        self.log_msg(engine.out.log)




    # def after_stride(self):
    #     """
    #     Check to see if an early exit is requested.
    #
    #     Returns:
    #
    #         boolean
    #
    #     :meta private:
    #     """
    #     # todo review for time independent case
    #     if self.early_nstride is not None:
    #         self.early_nstride_counter += 1
    #         return self.early_nstride_counter == self.early_nstride
    #     else:
    #         return False


    def deinit(self, engine):
        self.log_msg(engine.out.log)


    def log_msg(self, log):
        if self.dryrun is True:
            log(
                f"[Info] dryrun: True\n"
                f"[Info] All max_iteration values are coerced to value {self.DRYRUN_MAXITER}.\n"
                f"[Info] Sample set size parameter SPL coerced to value 16.\n"
                f"[Info] Batch size parameter batchsize coerced to value 16.\n"
            )
            if self.time_dependent:
                log(
                    f"[Info] The entire problem time interval is traversed in 1 step.\n"
                    f"[Info] All step multiples for all phases are coerced to value 1.\n"
                )
            log(f"[Info] Outputs have no meaning.")
        elif self.early_nstep is not None:
            n = self.early_nstep
            s = "s" if n > 1 else ""
            log(
                f"[Info] early_nstep: {n}\n"
                f"[Info] Simulation will run normally but exit after {n} step{s}."
            )
        elif self.early_nstride is not None:
            n = self.early_nstride
            s = "s" if n > 1 else ""
            log(
                f"[Info] early_nstride: {n}\n"
                f"[Info] Simulation will run normally but exit after {n} stride{s}."
            )


    def __str__(self):
        out = ""
        if self.dryrun:
            out += f"dryrun: True\n"
        elif self.early_nstep is not None:
            out += f"early_nstep: {self.early_nstep}\n"
        elif self.early_nstride is not None:
            out += f"early_nstride: {self.early_nstride}\n"
        out += f"substep: {self.substep}\n"
        out += f"step: {self.step}\n"
        out += f"stride: {self.stride}\n"
        return out


