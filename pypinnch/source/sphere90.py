



from .source_impl import \
    Union, \
    BoundingBox, \
    ConstantDim

from .box90 import Box90

from math import sqrt, pi, gamma



class Sphere90(Union):
    """
    A hypersphere of ``n`` dimensions,
    either a circle, sphere, 4-sphere, 5-sphere, etc.
    The dimension is implied by the value of ``center``.
    The parametrization does not allow for
    arbitrary embeddings, but does allow
    embeddings aligned with the dimensions of the external space.

    Example:
    A 2-dimensional circle parallel to the x-z plane
    in 3-dimensional space,
    centered at (0, 4, 0) with radius 2::

        circle = Sphere90(
            radius = 2.0,
            center = [0.0, ConstantDim(4.0), 0.0],
        )

    Parameters:

        radius (scalar): radius
        center (mixed list of scalar or :any:`ConstantDim`):
            A list of either scalars or values wrapped in :any:`ConstantDim`,
            whose presence indicates that the embedded dimension of the
            :any:`Sphere90` is smaller than the internal dimension of the :any:`Sphere90`.
        mode (optional string):
            A sampling mode, see :any:`Union`.
        SPL (optional positive integer):
            A sample per unit length value which, if set,
            overrides the problem-wide definition.
            (Default: None)

    """

    def __init__(
            self,
            radius,
            center,
            mode="pseudo",
            SPL=None,
    ):
        super().__init__(
            mode=mode,
            SPL=SPL,
        )
        self.center = center
        self.radius = radius
        self.sampler = None
        # memoized during init()
        self.idim = None


    def init_impl(self, parameters):
        """
        See :any:`Union`.

        """
        if callable(self.center):
            self.center = self.center(parameters)
        if callable(self.radius):
            self.radius = self.radius(parameters)
        self.dim = len(self.center)
        self.idim = len(list(filter(lambda x: not isinstance(x, ConstantDim), self.center)))
        if self.mode:
            origin = self.center.copy()
            # todo awk
            for i in range(len(self.center)):
                origin[i] -= self.radius
            proportions = []
            for coord in self.center:
                if isinstance(coord, ConstantDim):
                    proportions.append(coord)
                else:
                    proportions.append(2*self.radius)
            self.sampler = Box90(
                proportions = proportions,
                origin = origin,
                mode=self.mode,
                SPL=self.SPL,
            )
            self.sampler.init()
            void = Sphere90(
                radius=self.radius,
                center=self.center,
                mode=None,
            )
            void.init()
            void.complement()
            self.sampler -= void


    def sample_impl(
            self,
            SPL,
            Nmin,
            pow2,
            convex_hull_contains,
     ):
        """
        See :any:`Union`.

        """
        return self.sampler.sample(
            SPL,
            Nmin,
            pow2,
            convex_hull_contains,
        )


    def inside_impl(self, p):
        """
        See :any:`Union`.

        """
        # this would be nice, but there are ConstantDims.
        # p_ = np.array(p)
        # c_ = np.array(self.center)
        # pc = p_ - c_
        # rp = np.linalg.norm(pc)
        # return rp <= self.radius
        tiny = 1e-12
        p1 = []
        c1 = []
        if len(p) != len(self.center):
            raise ValueError(f"Something is wrong. Dimensions do not agree for a location comparison.")
        for pi, coord in zip(p, self.center):
            if isinstance(coord, ConstantDim) and not coord() - tiny < pi < coord() + tiny:
                # p is not in the slice where the shape is located
                return False
            else:
                p1.append(pi)
                c1.append(coord)
        rp = 0.0
        for pi, ci in zip(p1, c1):
            pci = pi - ci
            rp += pci*pci
        return rp <= self.radius*self.radius


    def measure_impl(self):
        """
        See :any:`Union`.

        For dimension n::

            Sn = 2*sqrt(pi**n) / gamma(n/2)

        is the surface content of a sphere of unit radius, and::

            Vn = r**n / n * Sn

        is the volumetric content of a sphere of radius r.
        """
        # Sn = 2**((n+1)/2) * pi**((n-1)/2) / (n-2)!! # n odd
        # Sn = 2 * pi**(n/2) / (n/2 - 1)! # n even
        x = pi**self.idim
        x = 2.0*sqrt(x)
        g = gamma(self.idim/2.0)
        Sn = x/g
        Vn = Sn * (self.radius**self.idim) / self.idim
        return Vn


    def bounding_box_impl(self):
        """
        See :any:`Union`.

        """
        if self.sampler:
            return self.sampler.bounding_box_impl()
        else:
            origin = []
            for coord in self.center:
                if isinstance(coord, ConstantDim):
                    origin.append(coord())
                else:
                    origin.append(coord)
            bb = BoundingBox(dim=self.dim)
            bb.mins = origin.copy()
            bb.maxs = origin.copy()
            for i in range(self.dim):
                if not isinstance(self.center[i], ConstantDim):
                    bb.mins[i] -= self.radius
                    bb.maxs[i] += self.radius
            return bb


    def internal_dimension_impl(self):
        """
        See :any:`Union`.

        """
        return self.idim


