


from .._impl.types import timed

from scipy.interpolate import griddata

from torch import \
    from_numpy as torch_from_numpy, \
    full as torch_full

from mv1fw import (
    get_fslabels,
    parse_labels,
    parse_fslabels,
)
from mv1fw.fw import XFormat



class MomentSets:
    """
    Used to store/retrieve moments,
    which are implemented as groups,
    each of which has a specific data structure.
    The name :any:`MomentSets`
    is in analogy with :any:`SampleSets`.

    For the sake of clarity, allow us to refer to a **moment set**,
    a convenient name for a data structure created in response to one
    or more :any:`Moment` definitions. .
    The purpose of :any:`MomentSets` is to manage the moment sets
    for a phase of training.
    Each moment set has the form of a map
    from a set of t's to an array having shape [N, indim + outdim].
    For t's considered, N is the number of points kept up to date
    for the purpose of interpolation, and indim is the
    dimension of the geometric region to be interpolated over
    (the target domain), and outdim is the number of moments
    that will utilize this input sample set
    (This is most likely 1, and in fact required to be 1 ITCINOOD).
    In effect, the moment set works like a map t --> D,
    where the D is some data living on the time slice at t. D may be empty,
    in which case the moment(s) are a function of time alone.

    The :any:`MomentSets` class manages one or more
    moment sets, which are accessed in residuals by
    via `any`:Problem.get_moment`.
    The moments are calculated and updated according
    to the definitions specified by the problem description
    using :any:`Moment`.

    """

    ###################################
    #
    # Quick refresher about access:
    #
    # moments = problem.moments
    # for inlabels in moments:
    #     moment = moments[inlabels] # type: Moment
    #     every = moment.every
    #     for lb in moment.resolution:
    #         outlabel = lb
    #         resolution = moment.resolution[lb] # type: integer
    #     for lb in moment.methods:
    #         outlabel = lb
    #         method = moment.methods[lb] # callable(X, problem)
    ###################################


    def __init__(
            self,
    ):
        self.lattices = {}
        self.fslabels_dict = {}
        self.SPD = None
        self.Nts = 0
        self.tinit = None


    def init_phase(
            self,
            problem,
            th,
    ):
        """
        (Called when samplesets are initialized.)

        At the start of a phase,
        a moment constructs its input list
        in preparation for calls to retrieve values.

        Arguments:

            problem (:any:`Problem):
            th (:any:`TimeHorizon`):
                todo document

        """
        if len(problem.moments)  == 0:
            # # > set tinit (dummy value, never used)
            self.tinit = th.tinit
        else:
            if problem.SPD is None:
                raise ValueError(f"Invalid configuration. Moments are defined, but SPD is not set in TopLine.")
            self.SPD = problem.SPD
            # todo replace SPD with step_partition (integer)
            # > find Nts, "number of t's", we wish to have a t
            # for each partition of the timestep in SPD, plus 1 for
            # the end of the step, or a "closed interval" partition.
            # Thus, if SPD = NPart(11), then Nts is 12, and so on.
            t0, _ = th.range()
            self.tinit = t0
            tfinal = t0 + th.stepsize()
            Nts = 0
            ti = t0
            # todo this tolerance:
            #  if the stepsize is small enough,
            #  and the SPD (as integer counting subintervals)
            #  is large enough, then this tolerance
            #  might need to be decreased.
            #  This case should be checked but it would not
            #  arise often (??), I have see no issues when
            #  SPD is as large as 36 subintervals and the stepsize is 0.0625,
            #  this comes to an SPD interval of ~0.0017 or 1.7e-3.
            #  tol needs to be large enough to surround
            #  roundoff error from adding, and small enough
            #  to avoid knocking the arithmetic out of whack.
            #  ...there's something unpretty about it,
            #  perhaps a better setup is needed. review
            tol = 1e-8
            while True:
                ti += self.SPD
                Nts += 1
                if ti > tfinal+tol:
                    break
            self.Nts = Nts
            # > populate stores, lattices
            moments = problem.moments
            for labels in moments:
                moment = moments[labels]
                fslabels = get_fslabels(*parse_labels(labels))
                # > populate fslabels dict
                # todo more space efficient method?
                for lb in moment.methods:
                    self.fslabels_dict[lb] = fslabels
                ranges = []
                resolutions = []
                for lb in moment.resolution:
                    resolutions.append(moment.resolution[lb])
                    ranges.append(problem.p.range(lb))
                # > final resolution, for the moment
                # todo outdim > 1
                outdim = 1
                outdimNts = outdim*Nts
                resolutions += outdimNts*[1]
                # > hack to get the output dimension to initialize to zero
                # todo test/review for other initializations of moments
                ranges += outdimNts*[(0,0)]
                # > create a lattice of nontemporal points where the
                #  moment will be computed, using problem ranges,
                #  resolution, and moment-specific labels
                self.lattices[fslabels] = problem.mesh(
                    ranges=ranges,
                    resolution=resolutions,
                    right_open=False,
                    # todo name of arg is fw_type
                    dtype=problem.driver().config.fw_type,
                )


    def advance(
            self,
            problem,
    ):
        """
        (Called when samplesets are advanced.)

        Advance the moment inputs.
        """
        # There is almost nothing to do,
        # all the information is maintained virtually
        # as regular lattices of nontemporal points
        # that are set at phase init.
        # > push scalar base time
        self.tinit += problem.th.stepsize()


    def deinit(self):
        self.lattices = {}


    @timed("moment_update")
    def update(
            self,
            iteration,
            problem,
    ):
        """
        Intended caller: the training phase.

        Arguments:

            iteration (integer):
                The current training iteration counter number.
            problem (:any:`Problem`):
                Problem instance.

        """
        moments = problem.moments
        for labels in problem.moments:
            moment = moments[labels]
            lbl, indim, with_t = parse_labels(labels)
            fslabels = get_fslabels(lbl, indim, with_t)
            every = moment.every
            if iteration % every == 0:
                # > refresh stored values
                lattice = self.lattices[fslabels]
                XF = XFormat(
                    X=lattice[:,:indim],
                    t=self.tinit,
                    fslabels=get_fslabels(lbl[:indim], indim, with_t),
                )
                for lb in moment.methods:
                    # print(f"[moment:update] iteration {iteration} lb {lb} tinit {self.tinit}")
                    method = moment.methods[lb]
                    for ti in range(self.Nts):
                        # todo outdim > 1
                        outdim = 1
                        outi = 0
                        i = indim + outdim*ti + outi
                        lattice[:,i:i+1] = method(
                            X=XF,
                            problem=problem,
                        )
                        # print(f"[moment:update] ti {ti} SPD {self.SPD} latticei {lattice[:,i:i+1]}")
                        XF.advance(deltat=self.SPD)


    def fslabels(self, outlabel):
        return self.fslabels_dict[outlabel]


    @timed("moment_lookup")
    def lookup(
            self,
            label,
            t,
            X = None,
    ):
        """
        Intended caller: :any:`Problem.get_moment`

        Look up the value of a moment by its label ``label``
        at a time ``t``. Optionally further specify values,
        otherwise all values present on the timeslice
        are defined.

        Arguments:

            label (string): label of a moment defined
                via :any:`Moment` in the problem description.
            t (scalar):
                Single time at which moment values have been requested,
                this time should be a time inside the current timestep.
            X (pytorch array):
                Requested values for lookup.
                The shape of ``X`` is (n, indim) where
                n is the number of points requested for lookup,
                and indim is the non-temporal input dimension
                for the moment. For each point x in X, if the exact
                values of x is not found, the closest value is returned
                via interpolation.
                if ``X`` is not passed in, the entire slice is returned,
                this exposes precisely what the interpolation is based on.

        """
        fslabels = self.fslabels_dict[label]
        lbl, indim, with_t = parse_fslabels(fslabels)
        lattice = self.lattices[fslabels]
        # todo outdim > 1
        outdim = 1
        outi = 0
        # > obtain the integer time
        ti = self._get_integer_time(
            t=t,
            label=label,
        )
        i = indim+outdim*ti+outi
        if X is None:
            # > return the output values
            # todo document what this branch is used for
            out = lattice[:,i:i+1]
        else:
            # > obtain the value
            if indim == 0:
                # The target moment is a scalar
                out = torch_full((X.shape[0], 1), lattice[0,i])
            else:
                # The target moment depends on
                # one or more nontemporal variables,
                # and must be interpolated using the lattice
                points = []
                for j in range(indim):
                    points.append(lattice[:,j])
                values = lattice[:,i]
                # > interpolate n values using m guidevalues
                out = griddata(
                    points=tuple(points), # shape [m,], number: indim
                    values=values, # shape [m,]
                    xi=X, # shape [n, indim]
                    method="linear",
                ) # shape [n,]
                out = out.reshape((-1, 1)) # shape [n, 1]
                out = torch_from_numpy(out)
                # print(f"[moment:lookup] t {t} out {out}")
        return out


    def _get_integer_time(self, t, label):
        tol = 1e-8
        errmsg = f"time {t} not found in the set of possible times while looking up moment {label}."
        # todo binary would be faster in worst case
        t1 = self.tinit
        if t < t1:
            raise ValueError(f"[momentsets:get_integer_time:Range] t1 {t1} {errmsg}")
        ti = 0
        def v(msg):
            # print(msg)
            pass
        v(f"[lookup] t {t}")
        v(f"[lookup] Nts {self.Nts}")
        while t > t1 + tol:
            v(f"[lookup] t {t} t1 {t1}")
            t1 += self.SPD
            ti += 1
        # t <= t1 + tol.
        if t1 - tol > t:
            v(f"[lookup !!!] t1 - tol {t1 - tol} t {t}")
            # t1 and t are not close enough together.
            # It may be a bug or may be due to user input.
            raise ValueError(f"[momentsets:get_integer_time:Tolerance] {errmsg}")
        # t1 - tol <= t <= t1 + tol.
        # ti is the desired integer time.
        if ti >= self.Nts:
            v(f"[lookup !!!] ti {ti} >= Nts {self.Nts}")
            tfinal = self.tinit + self.SPD*self.Nts
            v(f"[lookup] t1 {t1} tfinal {tfinal}")
            if t1 >= tfinal:
                v(f"[lookup] OFFTHEEDGE")
                pass
            # Possibly, t is too large and it fell off the edge.
            raise ValueError(f"[momentsets:get_integer_time:Range] {errmsg}")
        v(f"[lookup] Done. ti {ti}")
        return ti



