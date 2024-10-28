





from .icbase import ICBase
from .constraintsampleset import ConstraintSampleSet

from ..source.source_impl.bounding_box import BoundingBox


from mv1fw.fw import get_dtype



class SampleSets:
    """
    Sample sets for a phase of training, i.e. a training loop.

    If there are multiple constraints (such as an interior constraint
    and a boundary constraint), or if there is an IC
    in addition to some ordinary constraint (of any kind),
    then there are multiple notions of epoch, in general.
    There is one notion for each data set (each cylinder
    or IC-type sample set i.e. "base").
    Having a simple unit measuring a "pass through the training data"
    allows us to quantify training iterations
    just like when there is a single, monolithic dataset.
    We call this the **epoch** of training: an epoch is defined
    to be traversed when all sample sets have
    traversed at least one "age", or epoch on the particular sample set.
    The :any:`SampleSets` keeps counters for
    keeping track of ages of training, so it can keep track of the epoch.

    Parameters:

        SPL (integer):
            samples per measure, in time dimension (one per constraint)
            and for space dimensions relative to the total number of dimensions,
            for example dim = 3, SPM = SPL*SPL*SPL.
            The SPL may vary by :any:`Phase`.

    """

    # Note: in the code, we use the notation
    # "icc" or "ic" for an ICConstraint, "c" for a Constraint,
    # and "css" for a ConstraintSampleSet.

    def __init__(
            self,
            problem,
            SPL,
    ):
        self.SPL = SPL
        if problem.with_t:
            self.icbase = ICBase()
        else:
            self.icbase = None
        # switch that can be changed for a performance test,
        # not important to users
        self.version = "v1"
        # > build dict of constraint sample sets
        self.csss = {}
        for lb in problem.constraints:
            constraint = problem.constraints[lb]
            self.csss[lb] = ConstraintSampleSet(
                constraint,
                self.version,
            )
        # > the list active sample sets (labels only)
        self.active_csss = None
        # the number of epochs of training.
        # An epoch of training occurs when all
        # sources have progressed through at least one age
        # since the previous epoch.
        self.epoch_counter = 0


    def _time_dependent(self):
        """
        Helper to switch on time-independence
        in a readable way.
        """
        return self.icbase is not None


    def init_phase(
        self,
        active_constraint,
        out,
        batchsize,
        SPD,
        grading,
        config,
        icbase = None,
        shelf = None,
        th = None,
    ):
        """
        (Called by :any:`Phase`)

        Generate constraint and IC constraint sample sets.
        This call is performed at the beginning of a phase,
        and during this call the time decomposition is
        set for the phase.

        Arguments:

            active_constraint (dict string: :any:`Constraint`):
                The phase's active constraint boolean map.
            out (:any:`Manager`):
                Output manager instance.
            batchsize (integer):
                the batch size.
            SPD (optional scalar):
                step division delta, the distance between time samples
                where moments are extracted. Global, set in :any:`TopLine`.
            grading (boolean):
                whether grading strategy is used.
            config (:any:`DriverConfig`):
            icbase (optional :any:`Base`):
                Base instance from the driver,
                or None if time-independent.
            shelf (optional scalar):
                shelf length, applied to all (time-dependent) constraints
                when defining their sample sets.
            th (optional :any:`TimeHorizon`):
                the (phase-dependent) time horizon.

        """
        # todo allow phases of a stride to share sample sets.
        #  ...it would be nice but it is not a bottleneck.
        time_dependent_problem = (icbase is not None)
        out.log("~phase sample sets init~\n\n")
        if time_dependent_problem:
            th_ = th
            self.icbase.init_phase(
                icbase=icbase,
                batchsize=batchsize,
                th=th,
                device=config.device,
            )
        else:
            th_ = None
        # Aside: it's assumed that the user is not attempting to
        # apply the same sample set to two or more constraints.
        # Allowing that is not a priority at present, as creating sample sets
        # is ordinarily not a bottleneck, and if it becomes one
        # (due to memory pressure, presumably) we can return to it.
        self.active_csss = []
        for lb in self.csss:
            css = self.csss[lb]
            if active_constraint[lb]:
                css.init_phase(
                    label=lb,
                    dtype = get_dtype(config.fw_type),
                    SPL = self.SPL,
                    out = out,
                    batchsize = batchsize,
                    SPD = SPD,
                    shelf=shelf,
                    grading=grading,
                    mode = "pseudo",
                    th = th_,
                )
                self.active_csss.append(lb)
        out.log("\n~end~\n")


    def deinit(self):
        """
        (Called by :any:`Phase`)

        """
        if self._time_dependent():
            self.icbase.deinit()
        for lb in self.active_csss:
            css = self.csss[lb]
            css.deinit()


    def base_measure(self):
        # todo deprecate
        return self.icbase.result_measure


    def end_of_epoch(self):
        """
        Must be called after initializing batch cylinders.
        Call this once per train loop regardless of
        whether you do anything
        at the end of an epoch, as in::

            if samplesets.end_of_epoch():
                pass # no action

        Returns:

            boolean: Whether an epoch is reached.

        """
        num_done = 0
        for lb in self.active_csss:
            css = self.csss[lb]
            if css.cyl.epoch_marker:
                num_done += 1
        end_csss = (num_done == len(self.active_csss))
        end_base = self.icbase.end_of_epoch() if self._time_dependent() else True
        if end_csss and end_base:
            # reset the markers to indicate an epoch is reached
            if self._time_dependent():
                self.icbase.epoch_marker = False
            for lb in self.active_csss:
                css = self.csss[lb]
                css.cyl.epoch_marker = False
            self.epoch_counter += 1
            return True
        return False


    def epoch(self):
        """
        The current epoch, or number of epochs
        that have been fully completed.
        """
        return self.epoch_counter


    def advance(
            self,
            hub,
            config,
            problem,
    ):
        """
        This pushes a new IC for the next timestep.
        The new IC the old result.

        Arguments:

            hub (:any:`Hub`):
                Hub instance containing
                the neural network for the solution,
                passed here to hide the evaluation necessary to
                update the result, and
                the device where models are to be evaluated.
            config (:any:`DriverConfig`):
                driver's configuration, including the device
            problem (:any:`Problem`):
                Problem instance

        """
        self.icbase.advance(
            hub=hub,
            config=config,
            problem=problem,
        )
        # advance the constraints
        for lb in self.active_csss:
            css = self.csss[lb]
            css.advance(
                hub=hub,
            )


    # todo deprecated
    def timesample(self, N, t, dt):
        """
        Helper for getting a new vector of time values.

        Arguments:

            N: sample size
            t: initial time
            dt: time extent

        Returns:

            array with shape (N,1)

        """
        # todo using this?
        time_independent = (self.icbase is None)
        no_constraints = (len(self.active_csss) == 0)
        if no_constraints or time_independent:
            raise ValueError
        else:
            sample = None
            for lb in self.active_csss:
                css = self.csss[lb]
                sample = css.cyl.sampler(N)
                break
            return dt*sample + t


    def bounding_box(self):
        """
        Generate a bounding box based
        on the list of constraints (specifically the sources
        from each constraint, which may typ. be interior,
        or boundary constraints.)

        Returns:

            :any:`BoundingBox`

        """
        # all sources have the same "ambient" dimension
        dim = None
        for lb in self.active_csss:
            css = self.csss[lb]
            dim = css.constraint.source.dim
            break
        if dim is None:
            raise ValueError("[SampleSets:bounding_box] No constraints found.")
        out = BoundingBox(dim=dim)
        for lb in self.active_csss:
            css = self.csss[lb]
            out += css.constraint.source.bounding_box()
        return out


    def __str__(self):
        out = ""
        out += "SampleSets\n"
        out += f"SPL: {self.SPL}\n"
        out += f"Constraints: (number: {len(self.active_csss)})\n"
        for lb in self.active_csss:
            css = self.csss[lb]
            out += str(css)
        return out


