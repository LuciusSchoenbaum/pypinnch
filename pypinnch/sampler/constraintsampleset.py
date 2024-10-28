



from .cylinder import Cylinder

from ..source.dataset import DataSet
from .._impl.residual import DataResidual

from math import ceil



class ConstraintSampleSet:
    """
    For each phase, a :any:`ConstraintSampleSet` is
    associated to each :any:`Constraint`.
    This provides batches for training.

    The SPL (samples per unit length)
    is used to generate the sample set.
    This value sets the density of the samples.
    Sometimes the abbreviation SPM (samples per unit measure)
    is used for the general-dimension concept.
    (L: 1-dimensional, M: n-dimensional, for any n.)

    Parameters:

        constraint (:any:`Constraint`):
        version (string):
            Version of cylinder sampler used.
            Default: "v1".

    """

    def __init__(
            self,
            constraint,
            version = "v1",
    ):
        self.constraint = constraint
        self.cyl = None
        self.version = version
        self.time_dependent = None


    def init_phase(
            self,
            label,
            SPL,
            dtype,
            out,
            batchsize,
            SPD,
            shelf,
            grading,
            mode="pseudo",
            th=None,
    ):
        """
        (Called by :any:`SampleSets`)

        Initialize sample set for a :any:`Constraint`.

        Arguments:

            label (string):
                constraint label, for issue spot-checking
            SPL (integer):
                The # of samples per measure in the interior of the time cylinder,
                given as a 1-dimensional value (along the time axis).
                So if SPL = 100, then there are 100*SPM samples in a unit
                measure of the time cylinder which has dimension dim(source)+1.
            dtype:
                The dtype propagated from driver, to phase, to here.
            out (:any:`Manager`):
                output manager to log the work.
            batchsize (integer):
                size of batches
            th (optional :any:`TimeHorizon`):
                requested time horizon, or None for a time-independent problem.
            SPD (optional float):
                step division delta, the distance between time samples
                where moments are extracted. Global parameter set in :any:`TopLine`.
            shelf (scalar):
                shelf, parameter for setting up time-dependent sample sets.
            grading (boolean):
                whether grading strategy is used.
            mode (string):
                sampling mode

        """
        out.log(f"Initializing constraint {label}. SPL = {SPL}")
        self.time_dependent = (th is not None)
        if isinstance(self.constraint.source, DataSet):
            # > set the data residual
            self.constraint.residual = DataResidual(
                labels = self.constraint.source.get_labels()
            )
            reference_data_size = self.constraint.source.reference_data_size()
        else:
            reference_data_size = 0
        # > create a base for the cylinder
        if self.constraint.source is None:
            # 0-dimensional case
            base = None
        else:
            # todo this is not well-typed.
            #  make this XNoF.
            base = self.constraint.source(
                SPL=SPL,
                Nmin=batchsize,
                pow2=True if grading else False, # todo review
                convex_hull_contains=True,
            )
        # > set inputs to cylinder
        if not self.time_dependent:
            tinit = None
            stepsize = None
            shelf = None
            N_1d = None
        else:
            # > time-dependent case
            tinit = th.tinit
            stepsize = th.stepsize()
            shelf = shelf
            # > find N_1d
            if SPD:
                # SPD is always a regular partition of the timestep interval.
                # Ordinarily, SPD divides dt. But if it does not, we apply the ceiling
                # to extend off the edge of the step (preferable to undershooting).
                # This produces the integer number of partition intervals of the timestep.
                # To obtain the number of t's from this value, we add one (closed endpoints).
                N_1d = ceil(stepsize/SPD)+1
                out.log(f"Using N = {N_1d} samples on time axis (SPD = {SPD}).")
            else:
                N_1d = ceil(stepsize*SPL)+1
            if grading:
                if SPD:
                    out.log(f"SPD and grading strategy are not perfectly compatible yet! The SPD specification may be violated.")
                N_1d_ = 2
                while N_1d_ < N_1d:
                    N_1d_ *= 2
                if N_1d_ > N_1d:
                    out.log(f"For grading strategy, using N = {N_1d_} samples on time axis from target non-power-of-2 {N_1d}.")
                    N_1d = N_1d_
        if self.version == "v1":
            out.log("Using v1.")
            self.cyl = Cylinder(
                label=label,
                base=base,
                time_dependent=self.time_dependent,
                nsamples_1d=N_1d,
                batchsize=batchsize,
                samples_1d_mode=mode if SPD is None else 'regular',
                custom_batch=self.constraint.custom_batch,
                grading=grading,
                dtype=dtype,
                log=out.log,
                reference_data_size=reference_data_size,
            )
            self.cyl.init(
                tinit=tinit,
                stepsize=stepsize,
                shelf=shelf,
            )
            out.log("")
        elif self.version == "v2":
            raise NotImplementedError
        else:
            raise ValueError


    def deinit(self):
        """
        Called by phase at end of phase.
        """
        self.cyl.deinit()


    def measure(self):
        """
        Measure (volume, length, etc.) of domain.
        Accounts for 1-dimensional time axis.

        Returns:
            measure quantity (scalar)
        """
        M = 1.0
        if self.constraint.source is not None:
            M *= self.constraint.source.measure()
        if self.time_dependent:
            M *= self.cyl.measure_1d()
        return M


    def expand(self):
        self.cyl.expand()


    def contract(self):
        self.cyl.contract()


    def advance(self, hub):
        self.cyl.advance(dt=hub.dt)


    def batch(self):
        XX, QQref = self.cyl.batch()
        return XX, QQref


    def age(self):
        """
        Recall that an epoch of training occurs at the iteration
        where all sample sets are traversed.
        To avoid the confusion caused by making this
        term ambiguous, one sample set's epoch is an 'age'.

        Returns:
            integer
        """
        return self.cyl.age()


    def __str__(self):
        out = ""
        out += f"Constraint: {self.constraint}"
        out += f"version: {self.version}\n"
        out += f"measure: {self.measure()}\n"
        return out


