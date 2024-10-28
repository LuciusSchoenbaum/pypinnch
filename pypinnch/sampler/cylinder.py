# File Cylinder.py Created by Lucius Schoenbaum April 16, 2023




import torch
from math import log2

from numpy import float64 as numpy_float64

from .sampler_impl.sampler import Sampler
from .._impl.types import ispow2
from .unit_hypercube import UnitHypercube




class Cylinder(Sampler):
    """
    CylindricalSampler, or Cylinder for the sake of simplicity.
    Expandable stimulus for time-advancing, graded PINN training method,
    cf. :any:`Grading` . The data is a sample set from a geometric shape that
    is mathematically a cylinder with a base of any kind of geometric
    shape (of any dimension), and a single real-valued dimension
    which extends into the time dimension.
    For a time-dependent problem, the "time cylinder" can be
    extended, contracted, and advanced, along the distinguished time axis.
    For a time-independent problem, the height is zero and the dimensionality
    is reduced by one.

    If the problem is time-independent, then the cylinder is trivially
    initialized having only the base sample set.

    Parameters:

        label (string):
            The label for the constraint managing the cylinder.
            For example, ``bc_left''.
        base ():
            The time-independent sample set on the base of the time cylinder.
            If the time-dependent space is (x,y,z,t), then base has
            dimensions (x,y,z). Note that base corresponds to a *constraint*
            so it might be a sample of a boundary, for example. However,
            all base's must have the same "ambient" dimensions.
            The base may contain reference values, for data-driven training.
        nsamples_1d (integer):
            Multiple controlling density of points in time cylinder.
        batchsize (integer):
            batch size for training.
            The size determines performance (due to potential bottlenecks
            when arrays grow to impact a hardware ceiling)
            as well as accuracy (potentially), since it is the cumulative effect
            of the batch that determines the discrete gradient updates.
            So a bigger batch is (heuristically) more "sensitive".
            Heuristic::

                cpu: 64 (sufficient for 0d+t, 1d, 1d+t problems)
                gpu: 512 or higher

        samples_1d_mode (string):
            Mode used for probabilistic sampling in one
            dimension (for time).
        custom_batch (:any:`CustomBatch`):
            custom_batch for any custom batching todo documentation
        grading (boolean):
            Whether or not grading is used.
        dtype:
            datatype for arrays
        log (optional :any:`Logger`):
            log, for logging, or `None` to print directly to standard output.
        reference_data_size (integer):
            The number of reference columns in the data. Default: 0

    """

    # todo: use XNoF

    # todo: v1 vs v2

    # todo: reference_data_size > 0:
    #  - if it is time-independent, we are ok. (CHECK)
    #  - if there is time-dependence, there is work to do if the time
    #    values should be randomly generated.
    #  - if there are time values set by the data (i.e. a priori),
    #    then this implementation doesn't work, it is a task for the future.
    #    it is a little bit too bad because this case might introduce some
    #    unwanted clutter and between time dependent/time independent
    #    it is already a bit cluttered. Perhaps review.


    def __init__(
            self,
            label,
            base,
            time_dependent,
            nsamples_1d,
            batchsize,
            samples_1d_mode,
            custom_batch,
            grading,
            dtype = numpy_float64,
            log = None,
            reference_data_size = 0,
    ):
        super().__init__()
        self.base = base
        self.nsamples_1d = nsamples_1d
        self.custom_batch = custom_batch
        self.dtype = dtype
        self._log = log
        self.reference_data_size = reference_data_size
        # extent of time cylinder in time dimension up to tfinal
        self.stepsize = None
        # shelf, or extension of time sample past tfinal
        self.shelf = None
        self.epoch_marker = False
        # The data managed by the class:
        # expandable data in the time cylinder
        self.sampleset = None
        # fields for time dependent case
        self.structural_maxlevel = None
        self.sampler = None
        # state of class: level >= 0
        # sets how often the timesample has been expanded/contracted
        self.level = 0
        # pointers for batching
        self.point = 0
        # counters for age (1 age == 1 trip through the dataset)
        self.age_counter = 0
        # > set batchsize
        if self.custom_batch is not None:
            if batchsize % self.custom_batch.divisor != 0:
                raise ValueError(f"batchsize {batchsize} is not divisible by custom batch divisor {self.custom_batch.divisor}.")
            batchsize_ = batchsize // self.custom_batch.divisor
        else:
            batchsize_ = batchsize
        self.batchsize = batchsize_
        # > set the base and find the size
        if base is None:
            # zero-dimensional case.
            self.indim = 0
            size = 1
            # > sanity check
            if not time_dependent:
                raise ValueError(f"Time independent case requested, but base sample set is not found.")
        else:
            # input dimension > 0.
            self.indim = base.shape[1] - self.reference_data_size
            size = base.shape[0]
        # > things to set up or check in time dependent case
        if time_dependent:
            # it is impossible to increase the level (expand) beyond this value.
            self.structural_maxlevel = int(log2(size*self.nsamples_1d))
            self.sampler = UnitHypercube(dimension=1, mode=samples_1d_mode, dtype=dtype)
            # > nsamples_1d must be large enough to include the endpoints (tinit and tfinal for the step).
            if self.nsamples_1d < 2:
                raise ValueError
            # Invariant: nsamples_1d >= 2.
            # > grading requires power of 2 size
            if grading:
                if not ispow2(self.nsamples_1d):
                    raise ValueError(f"nsamples_1d must be a power of 2")
                if size > 2**(int(log2(size))):
                    raise ValueError(f"Sample size {size} must be a power of 2 for constraint {label}")
                # Invariant: nsamples_1d is a power of 2.
                # Invariant: base sample size is a power of 2.
            # > user check
            batches_per_age = int(size*self.nsamples_1d/self.batchsize)
            if batches_per_age < 16:
                self.log(f"[Warning] Only {batches_per_age} batches per age for constraint {label}. ")
                self.log(f"((size = {size}, nsamples_1d = {self.nsamples_1d}))")
                self.log(f"To enlarge, decrease batch size, increase sample size on base, or increase nsamples_1d.")


    def init(
            self,
            tinit = None,
            stepsize = None,
            shelf = None,
            finalize = True
    ):
        """
        Populate the sample in time interval tinit ≤ t ≤ tinit+stepsize.
        Could call again at any point to repopulate sample set,
        keeping the base the same.

        Arguments:

            tinit (optional scalar):
                Initial time, or time at the base of the initial time cylinder,
                or `None` if the problem is time-independent.
            stepsize (optional scalar):
                The extent into the time dimension to the tfinal value of the step,
                or `None` if the problem is time-independent.
            shelf (optional scalar):
                Extent into the time dimension past the tfinal value of the step,
                or `None` if the problem is time-independent.
            finalize (boolean):
                Finalize the initialization by shuffling the data set.
                Unset to unit-test data generation.

        """
        # todo test: call at end of each age (1), end of each step (2), once per stride (3)
        if tinit is not None:
            self.stepsize = stepsize
            self.shelf = shelf
            textent = self.stepsize + self.shelf
            if self.indim == 0:
                X = (textent*self.sampler(self.nsamples_1d, corners=True) + tinit)\
                    .reshape((self.nsamples_1d, 1))
            else:
                # > extend the base into time dimension, including reference values, if any
                if self.sampler.mode == "regular":
                    # case when SPD is set.
                    X = torch.zeros([0, self.indim + 1 + self.reference_data_size])
                    Ts = (textent*self.sampler(self.nsamples_1d, corners=True) + tinit)
                    for t in Ts:
                        times = torch.full([self.base.shape[0], 1], t)
                        X2 = torch.hstack((self.base[:,:self.indim], times, self.base[:,self.indim:]))
                        X = torch.vstack((X, X2))
                else:
                    size = self.base.shape[0]
                    # > create two copies of base at t0, t0+dt
                    X = torch.zeros([0, self.indim + 1 + self.reference_data_size])
                    ts = [tinit, tinit+self.stepsize]
                    for t in ts:
                        times = torch.full([self.base.shape[0],1], t)
                        X2 = torch.hstack((self.base[:,:self.indim], times, self.base[:,self.indim:]))
                        X = torch.vstack((X, X2))
                    # > fill the remainder with random sample in the interior
                    for _ in range(2, self.nsamples_1d):
                        times = (textent*self.sampler(size, corners=False) + tinit)
                        # times[0,0] = t0 + self.dt
                        X2 = torch.hstack((self.base[:,:self.indim], times, self.base[:,self.indim:]))
                        X = torch.vstack((X, X2))
            # todo make this XNoF
            self.sampleset = X
        else:
            # todo make this XNoF
            self.sampleset = self.base
        # > re-init state
        self.point = 0
        self.level = 0
        self.age_counter = 0
        # > finalize
        if finalize:
            self.sampleset = self.shuffle()


    def X(self):
        """
        Return the entire sample set,
        combining the class's basesample
        and timesample objects.
        Can be used by output utilities,
        e.g. :any:`SampleMonitor`.

        Note: recall that the time is the
        indim'th index, ITCINOOD.
        This was a decision made early on
        that did not age well.

        Returns:

            the sample set X

        """
        # todo acquire this op by inherit from XNoF
        return self.sampleset


    def shuffle(self):
        idxs = torch.randperm(self.sampleset.shape[0])
        return self.sampleset[idxs]


    def _impl(self, contract=False):
        """
        Main v1 implementation method.
        """
        dt = self.stepsize
        if contract:
            dt = -dt
        X = self.sampleset
        L = self.level
        N = X.shape[0]
        p = 1 << L
        beg = 0
        end = N >> p
        p = p >> 1
        M = N >> p
        for _ in range(L):
            itime = self.indim
            X[beg:end, itime:itime+1] += p*dt
            beg = beg + M
            end = end + M


    def deinit(self):
        """
        Called at end of phase.
        """
        self.sampleset = None


    def batch(self):
        """
        Get a batch to train on.

        .. note::

            The batching method
            "throws away" the remaining batch,
            if it is not big enough to make a full batch.
            This is acceptable because we shuffle
            after each age, and because we check (in __init__)
            that an age is at least 16 batches, at minimum.

        Returns:

            XX, QQref, where QQref is a set of reference values.
            Since there are no reference values for
            a sampling constraint, QQref is None.

        """
        beg = self.point
        end = self.point + self.batchsize
        XXQQref = self.sampleset[beg:end,:]
        self.point = end
        if self.point + self.batchsize > self.sampleset.shape[0]:
            self.sampleset = self.shuffle()
            self.point = 0
            age_tmp = self.age()
            self.age_counter += 1
            # Invariant: only here is this flag set.
            if self.age() > age_tmp:
                self.epoch_marker = True
        if self.custom_batch is not None:
            XXQQref = self.custom_batch(XXQQref)
        if self.reference_data_size > 0:
            if self.sampler is not None:
                # > time dependent
                XX = XXQQref[:,:self.indim+1]
                QQref = XXQQref[:,self.indim+1:]
            else:
                print(f"[batch (ref data, timedep)] indim {self.indim} shape XXQQref {XXQQref.shape}")
                XX = XXQQref[:,:self.indim]
                QQref = XXQQref[:,self.indim:]
        else:
            XX = XXQQref
            QQref = None
        return XX, QQref


    def age(self):
        """
        The current age.
        """
        return self.age_counter


    def measure_1d(self):
        """
        The measure of the time domain currently set to be sampled
        (a one-dimensional domain).

        Returns;

            measure (extent along time axis)
        """
        return self.stepsize + self.shelf


    def size(self):
        return self.sampleset.shape[0]


    def expand(self):
        """
        Expand sample into time dimension.
        NOTE. The shelf de creates some overlap,
        but we ignore it (for now?).
        """
        if self.level == 63:
            # You might catch this exception due to a catastrophic bug.
            raise ValueError(f"Attempted expand past what impl can support.")
        if self.level == self.structural_maxlevel:
            # You might do this accidentally on a test run:
            raise ValueError(f"Attempted expand past maximum level {self.structural_maxlevel}.")
        self.level += 1
        self._impl()


    def contract(self):
        """
        Contract sample into time dimension.
        NOTE. The shelf de creates some overlap,
        but we ignore it (for now?).
        """
        if self.level == 0:
            raise ValueError(f"Attempted contract past minimum level 0.")
        self._impl(contract=True)
        self.level -= 1


    def advance(self, dt = None):
        """
        Advance all sample points along time axis by a step value.

        Arguments:
            dt (optional scalar):
                time step value. If None, use the class's dt.
        """
        # Just a comment: advance is by simple translation.
        # So if you advance the sample set by a custom dt value,
        # the 'width' (in time) of the sample will still be the width that was imposed
        # during the init() method, ITCINOOD.
        if dt is not None:
            deltat = dt
        else:
            deltat = self.stepsize
        # > sanity check for time-independent case
        if self.nsamples_1d is None:
            raise ValueError(f"Cannot advance in time in time-independent case.")
        # > advance
        deltat_vector = torch.full([self.sampleset.shape[0],], deltat)
        self.sampleset[:,self.indim] += deltat_vector
        # > reset age counters on advance
        self.age_counter = 0



    def log(self, msg):
        if self._log is not None:
            self._log(msg)
        else:
            print(msg)





