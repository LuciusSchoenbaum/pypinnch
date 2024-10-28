



from numpy import \
    array as np_array

# todo fw or numpy
from torch import \
    full as torch_full, \
    zeros as torch_zeros, \
    from_numpy as torch_from_numpy, \
    Tensor as torch_Tensor

from .source_impl import \
    BoundingBox, \
    Union, \
    ConstantDim

from ..sampler.unit_hypercube import UnitHypercube

from math import log2



class Box90(Union):
    """
    Box that has the same dimension as ambient space,
    or else is zero-dimensional only along one or more coordinate axes.
    Simple to implement, easy to debug/test,
    sufficient for many kinds of problems.

    The Box is constant in the ith dimension
    if and only if the ith value of ``proportions``
    is set using :any:`ConstantDim`.
    If ``proportions`` or ``origin`` is callable,
    it must evaluate to a list of the kind specified
    when evaluated on :any:`Problem`'s ``Parameters`` class.
    These lists must express the correct dimension
    (have the correct length).

    Example:
    a 2-dimensional square box with its four boundaries (sides).
    The box is 4x4 and centered at the origin::

            box = Box90(
                proportions = [4.0, 4.0],
                origin = [-2, -2],
            )
            left = Box90(
                proportions = [ConstantDim(-2.0), 4.0],
                origin = [None, -2],
            )
            right = Box90(
                proportions = [ConstantDim(2.0), 4.0],
                origin = [None, -2],
            )
            bottom = Box90(
                proportions = [4.0, ConstantDim(-2.0)],
                origin = [-2, None],
            )
            top = Box90(
                proportions = [4.0, ConstantDim(2.0)],
                origin = [-2, None],
            )

    For a dimension where the shape is zero-dimensional,
    the origin coordinate is unused and can be set to ``None``,
    as in the example above.

    Parameters:

        proportions (list of scalar or :any:`ConstantDim` or callable):
            proportions of the box. If callable, it is evaluated
            on the :any:`Parameters` class of the :any:`Problem`.
            This effectively parametrizes the geometry.
        origin (optional list of optional scalar, or callable):
            origin, the component-wise minimum point.
            If there are constant dimensions, the origin point is
            redundant information, and ``None`` can be passed
            in place of the correct scalar value.
            Callable properties are identical to that of ``proportions``.
            (Default: None, places the origin at the coordinate origin.)
        mode (optional string):
            Choice of sampling algorithm, or None to
            disable sampling for this source.
            At present only pseudo is implemented ITCINOOD.
            (Default: "pseudo")
        SPL (optional positive integer):
            A sample per unit length value which, if set,
            overrides the problem-wide definition.
            (Default: None)

    """

    def __init__(
            self,
            proportions,
            origin = None,
            mode="pseudo",
            SPL=None,
    ):
        super().__init__(
            mode=mode,
            SPL=SPL,
        )
        self.proportions = proportions
        self.origin = origin
        self.constantdims = []
        # memoized during init()
        self.idim = None


    def init_impl(
            self,
            parameters,
    ):
        """
        See :any:`Union`.

        """
        # > evaluate all callables
        if callable(self.proportions):
            self.proportions = self.proportions(parameters)
        if callable(self.origin):
            self.origin = self.origin(parameters)
        if self.origin is None:
            self.origin = len(self.proportions)*[0.0]
        if len(self.proportions) != len(self.origin):
            raise ValueError(f"proportions and origin arguments do not "
                             f"have the same length. The dimension is undefined.")
        # > set dimensions (dim, idim)
        self.dim = len(self.proportions)
        self.idim = len(list(filter(lambda x: not isinstance(x, ConstantDim), self.proportions)))
        # populate constantdims
        # todo - constantdims should also include extended dimensions,
        #  (for cylinder, prism, etc)
        #  suggesting a name change.
        for i, x in enumerate(self.proportions):
            if isinstance(x, ConstantDim):
                self.constantdims.append(x())
                self.proportions[i] = 1.0
            # Note: proportion == 0 can be useful sometimes, so allow it.
            elif x is None or x < 0.0:
                raise ValueError(f"Invalid proportion x = {x}")
            else:
                # x is assumed to be a float representing
                # an extent in the ith dimension.
                # append None to signal not a constant dimension.
                self.constantdims.append(None)
        # remove any None from origin.
        # We do not check the user.
        for i, x in enumerate(self.origin):
            if x is None:
                self.origin[i] = self.constantdims[i]
        # > set up random source
        if self.dim > 0 and self.mode is not None:
            self.sampler = UnitHypercube(
                dimension=self.dim,
                mode=self.mode,
                dtype=self.dtype,
            )
        else:
            self.sampler = None



    def sample_impl(
            self,
            SPL,
            Nmin,
            pow2,
            convex_hull_contains,
            special = None,
    ):
        """
        See :any:`Union`.

        The argument ``special`` (STIUYKB): cf. :any:`Special`.
        """
        if self.dtype is None:
            raise ValueError(f"Uninitialized source")
        # > find N
        SPM = SPL**self.idim if self.SPL is None else self.SPL**self.idim
        meas = self.measure()
        N = int(SPM*meas)
        if Nmin is not None:
            N = Nmin if N < Nmin else N
        if pow2:
            M = 2**(int(log2(N)))
            N = 2*M if M != N else M
        # > populate X
        if self.dim == 0:
            # //\\ the dimension is zero.
            # todo test this branch
            X = torch_full([N,1], self.origin[0])
        elif not self.voids and special is None:
            # //\\ there are no voids and the dimension is not zero.
            # > sample unit hypercube
            X = self.sampler(
                N,
                constantdims=self.constantdims,
            )
            # > scale and translate
            X = X*np_array(self.proportions)
            X += np_array(self.origin)
            # > place constant dimension values
            for i, v in enumerate(self.constantdims):
                if v is not None:
                    X[:, i] = v
        else:
            # //\\ there are voids OR there is a special condition,
            # AND, the dimension is not zero.
            # todo faster way - vectorize? test processing of blocks at a time?
            i = 0
            X = torch_zeros((N,self.dim))
            safety = 1000*N
            safety_count = 0
            while i < N:
                x = self.sampler(1, constantdims=self.constantdims)
                x = x*np_array(self.proportions)
                x += np_array(self.origin)
                for j, v in enumerate(self.constantdims):
                    if v is not None:
                        x[:, j] = v
                # todo test union w/ 2+ voids
                # > convert point to a python list
                # You must do this because ITCINOOD,
                # (1) samplers "think in tensor" whereas unions "think in list".
                # (2) sampler returns something that looks like a list of points, not a single point.
                # It may be worthwhile to (once again) go to using only tensors.
                # ...or make some kind of other change. But this will be ok for now.
                x = x[0,:].tolist()
                good = True
                for void in self.voids:
                    if x in void:
                        good = False
                if special is not None:
                    if not special(x):
                        good = False
                if good:
                    X[i,:] = torch_Tensor(x)
                    i += 1
                safety_count += 1
                if safety_count == safety:
                    raise ValueError(f"[Box90] Sampler is having difficulty sampling outside of voids.")
        if convex_hull_contains:
            # This procedure will work PROVIDED THAT:
            # there are no voids, or any voids that are present
            # are contained inside of the interior of the domain.
            # (I.e., they do not intersect the boundary.)
            # The simplest example where this condition is violated
            # is a "Utah" shape, in which case, this procedure
            # will insert a single point at the ghost corner
            # located near Rock Springs, Wyoming.
            # This is a known issue, violating the documentation,
            # but we choose not to fix it or address it at this time.
            # The reason is that in order to truly fix the issue,
            # we would need to not only remove the ghost point(s),
            # but also take the still further measure of adding points
            # in the new convex hull extremeties. In the Utah example,
            # this would be the two triple state boundary points that
            # are (respectively) due West, and due South of Rock Springs, Wyoming.
            # Note that while this *is* technically an issue, it is not an issue in a
            # large number of cases.
            Xc = self.sampler.get_corners(dim=self.dim, constantdims=self.constantdims)
            Xc_n = Xc.shape[0]
            if Xc_n > N:
                raise ValueError(f"[UnitHypercube] Not enough points {N} for sample set to include corners.")
            # > corners: scale and translate
            Xc = Xc*np_array(self.proportions)
            Xc += np_array(self.origin)
            # > corners: place constant dimension values
            for i, v in enumerate(self.constantdims):
                if v is not None:
                    Xc[:, i] = v
            # overwrite the upper section of the sample set with corner points
            X[:Xc_n,:] = Xc
        return X




    def inside_impl(self, p):
        """
        See :any:`Union`.

        """
        out = True
        tiny = 1e-12
        for prop, cd, org, x in zip(self.proportions, self.constantdims, self.origin, p):
            if cd is not None:
                out &= cd - tiny < x < cd + tiny
            else:
                out &= org <= x <= org + prop
            if not out:
                break
        return out



    def measure_impl(self):
        """
        See :any:`Union`.

        """
        m = 1.0
        for i in range(self.dim):
            if self.constantdims[i] is None:
                m *= self.proportions[i]
        return m



    def bounding_box_impl(self):
        """
        See :any:`Union`.

        """
        out = BoundingBox(self.dim)
        for i in range(self.dim):
            if self.constantdims[i] is not None:
                out.mins.append(self.constantdims[i])
                out.maxs.append(self.constantdims[i])
            else:
                out.mins.append(self.origin[i])
                out.maxs.append(self.origin[i] + self.proportions[i])
        return out


    def internal_dimension_impl(self):
        """
        See :any:`Union`.

        """
        return self.idim




