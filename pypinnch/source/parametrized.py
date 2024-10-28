


from numpy import \
    array as np_array, \
    float64 as np_float64

from torch import \
    hstack as torch_hstack, \
    Tensor as torch_Tensor

from .source_impl import \
    BoundingBox, \
    Source

from ..sampler.unit_hypercube import UnitHypercube

from .._impl.impl2.torch import mesh

from math import log2

from mv1fw.fw import (
    fw_type,
)


class Parametrized(Source):
    """
    A source that is defined by a parametrization.

    Parameters:

        parametrization (callable):
            todo
        ranges (list of optional pairs of scalar):
            List of ranges, implicitly setting the
            dimension of the parametrized source.
        measure (scalar):
            The measure of the geometric object, in its interior dimension.
            It must be specified, but it can be an estimate.
        mode (string):
            sampling mode
        bias (optional callable):
            todo

    """

    # todo improve docstring

    # todo implement bias

    def __init__(
            self,
            parametrization,
            ranges,
            measure,
            mode = 'pseudo',
            SPL = None,
            bias = None,
    ):
        super().__init__()
        self.parametrization = parametrization
        self.ranges = ranges
        self._measure = measure
        self.mode = mode
        self.bias = bias
        self.sampler = None
        self.idim = None
        self.origin = None
        self.proportions = None
        self.SPL = SPL,


    def init(
            self,
            # todo fw float64
            dtype = np_float64,
            parameters = None,
    ):
        if not self.initialized:
            super().init(dtype, parameters)
            # > evaluate on parameters
            # todo I'm going to require that this is set up this way,
            # todo document this requirement,
            #  provide an example in the docstring
            self.parametrization = self.parametrization(parameters)
            if callable(self.ranges):
                self.ranges = self.ranges(parameters)
            # > set the default range, 0..1
            for i, rng in enumerate(self.ranges):
                if callable(rng):
                    self.ranges[i] = rng(parameters)
                elif rng is None:
                    self.ranges[i] = (0.0, 1.0)
            # > set origin (in parameter space) and proportions
            self.origin = [rng[0] for rng in self.ranges]
            self.proportions = [rng[1] - rng[0] for rng in self.ranges]
            # > set dimensions
            self.dim = len(self.parametrization(torch_Tensor(self.origin)))
            self.idim = len(self.ranges)
            if self.dim == 0:
                raise ValueError
            if self.idim == 0:
                raise ValueError
            # > set up random source
            if self.dim > 0 and self.mode is not None:
                self.sampler = UnitHypercube(
                    dimension=self.idim,
                    mode=self.mode,
                    dtype=self.dtype,
                )
            else:
                self.sampler = None


    def sample(
            self,
            SPL,
            Nmin = None,
            pow2 = False,
            convex_hull_contains = True,
    ):
        """

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
        if True:
            # > populate X
            X = self.sampler(N)
            # > scale and translate
            X = X*np_array(self.proportions)
            X += np_array(self.origin)
            X = self.parametrization(X)
            if isinstance(X, tuple):
                # > stitch up X
                X = torch_hstack(X)
        if convex_hull_contains:
            # convex_hull_contains: This argument is now a misnomer. :-/
            Xc = self.sampler.get_corners(dim=self.idim, constantdims=None)
            Xc_n = Xc.shape[0]
            if Xc_n > N:
                raise ValueError(f"[UnitHypercube] Not enough points {N} for sample set to include corners.")
            # > corners: scale and translate
            Xc = Xc*np_array(self.proportions)
            Xc += np_array(self.origin)
            Xc = self.parametrization(Xc)
            if isinstance(Xc, tuple):
                Xc = torch_hstack(Xc)
            # overwrite the upper section of the sample set with corner points
            X[:Xc_n,:] = Xc
        # todo TEST __________
        return X


    def measure(self):
        """

        """
        return self._measure


    def bounding_box(self):
        """

        """
        # todo cache the result

        #### parameters
        resolution = 10
        pad = 10
        out = BoundingBox(dim=self.dim)
        X = mesh(
            ranges=self.ranges,
            resolution=resolution,
            right_open=False,
            dtype=fw_type(self.dtype), # awk
            device=None,
        )
        X = self.parametrization(X)
        if isinstance(X, tuple):
            X = torch_hstack(X)
        # find max and min of bounding box
        Xmin = X[0,:].tolist()
        Xmax = X[0,:].tolist()
        # todo gracefully using slices
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if Xmin[j] > X[i,j]:
                    Xmin[j] = X[i,j]
                if Xmax[j] < X[i,j]:
                    Xmax[j] = X[i,j]
        out.mins += Xmin
        out.maxs += Xmax
        # pad
        for i in range(self.dim):
            factor = float(pad)/100.0
            extenti = out.maxs[i] - out.mins[i]
            out.mins[i] = out.mins[i] - factor*extenti
            out.maxs[i] = out.maxs[i] + factor*extenti
        return out


    def internal_dimension(self):
        """

        """
        return self.idim


