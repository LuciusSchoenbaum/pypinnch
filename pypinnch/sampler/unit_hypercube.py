

from skopt.sampler import \
    Lhs as skopt_Lhs, \
    Halton as skopt_Halton, \
    Hammersly as skopt_Hammersly, \
    Sobol as skopt_Sobol
from numpy import \
    float64 as numpy_float64, \
    ones as numpy_ones, \
    zeros as numpy_zeros, \
    random as numpy_random, \
    linspace as numpy_linspace, \
    asarray as numpy_asarray


from .sampler_impl.sampler import Sampler
from .._impl.types import smallest_nonzero, unset_at_index


from mv1fw.fw import fw_from_numpy



class UnitHypercube(Sampler):
    """

    Parameters:

        dimension (integer):
            dimension of sample points
        mode (string):
            choice of sampler:

                - "regular" regular partition (only possible if dimension is 1).
                - "pseudo" pseudorandom
                - "LHS" Latin hypercube sampling
                - "Halton" Halton sequence
                - "Hammersley" Hammersley sequence
                - "Sobol" Sobol sequence

        dtype (numpy data type):

    """

    def __init__(
            self,
            dimension,
            mode,
            dtype = numpy_float64,
            seed = None,
    ):
        super().__init__()
        self.dimension = dimension
        self.mode = mode
        self.dtype = dtype
        # todo review seed = None case
        self.call_seed = seed if seed is not None else 123



    def __call__(self, n, corners = False, constantdims = None):
        """
        Sample of n values using a standard RNG (pseudorandom)
        or a skopt quasirandom sampler.

        Arguments:

            n (integer):
                number of points in sample
            corners (boolean):
                whether to include the corners of the hypercube
                in the sample set. This can also be done
                in a more customizable way using ``get_corners``.
            constantdims (optional list of optional floats):
                the constant dimensions requested in the sample.
                If None, there are no constant dimensions.
                If None at index i, then i is not a constant dimension.

        Returns:

            Numpy array of shape (n, dim)

        """
        dim = self.dimension
        mode = self.mode
        if mode == "pseudo":
            # Sample using a standard RNG.
            X = numpy_random.random(size=(n, dim)).astype(self.dtype)
        elif mode == "regular":
            # Sample regularly.
            if self.dimension > 1:
                raise ValueError(f"Cannot sample regularly unless dimension is 1.")
            X = numpy_linspace(start=0.0, stop=1.0, num=n, endpoint=True, dtype=self.dtype)
        else:
            if mode == "Lhs":
                sampler = skopt_Lhs()
                skip = 0
            elif mode == "Halton":
                # 1st point: [0, 0, ...]
                # todo I'm leaving this + 1000 as-is for now.
                sampler = skopt_Halton(min_skip=self.call_seed, max_skip=self.call_seed + 1000)
                skip = 0
            elif mode == "Hammersley":
                # 1st point: [0, 0, ...]
                if dim == 1:
                    sampler = skopt_Hammersly(min_skip=self.call_seed, max_skip=self.call_seed)
                    skip = 0
                else:
                    sampler = skopt_Hammersly()
                    skip = 1
            elif mode == "Sobol":
                # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
                sampler = skopt_Sobol(randomize=False)
                if dim < 3:
                    skip = 1
                else:
                    skip = 2
            else:
                raise ValueError(f"sampler not defined")
            # Create a space and sample from it using skopt API:
            space = [(0.0, 1.0)] * dim
            X =  numpy_asarray(
                sampler.generate(space, n + skip)[skip:],
                dtype=self.dtype,
            )
        if corners:
            if mode == "regular":
                # corners are always present in this case.
                pass
            else:
                Xc = self.get_corners(
                    dim=dim,
                    constantdims=constantdims,
                )
                Xc_n = Xc.shape[0]
                if Xc_n > n:
                    raise ValueError(f"[UnitHypercube] Not enough points {n} for sample set to include corners.")
                # overwrite the upper section of the sample set with corner points
                X[:Xc_n,:] = Xc
        self.call_seed += 1
        # todo use fw throughout
        X = fw_from_numpy(X)
        return X


    def get_corners(
            self,
            dim,
            constantdims,
    ):
        """
        Get the corners of the unit hypercube,
        considering constant dimensions.
        Cf. :any:`Box90` for an application.

        This procedure was tested on both x86 and ARM 64-bit, but not rigorously.

        """
        # non-constant dimensions bitfield
        ncd = 0
        # number of non-constant dimensions
        nncd = 0
        if constantdims is not None:
            for i in range(dim):
                if constantdims[i] is None:
                    # ith dimension is not constant
                    ncd += 1<<i
                    nncd += 1
        else:
            ncd = ~ncd
            nncd = dim
        twonncd = 2**nncd
        Xc = numpy_zeros((twonncd, dim)).astype(self.dtype)
        tgt = 0
        for i in range(nncd):
            tgt = smallest_nonzero(ncd, hint = tgt)
            if tgt < 0:
                raise ValueError(f"Error while reading constant dimension bitfield")
            beg = 2**i
            end = 2*beg
            Xc[beg:end,:] = Xc[0:beg,:]
            Xc[beg:end,tgt:tgt+1] = numpy_ones((beg,1)).astype(self.dtype)
            ncd = unset_at_index(ncd, tgt)
            tgt += 1
        # todo use fw throughout
        Xc = fw_from_numpy(Xc)
        return Xc




