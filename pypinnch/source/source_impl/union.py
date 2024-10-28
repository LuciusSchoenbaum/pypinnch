

from numpy import \
    float64

from torch import \
    zeros as torch_zeros, \
    vstack as torch_vstack

from .bounding_box import BoundingBox
from .source import Source, UninitializedSource





class Union(Source):
    """
    A geometric :any:`Source` where solid constructive
    geometry is available.
    For example, :any:`Box90`, :any:`Sphere90`
    are instances of :any:`Union`.

    There is some ambiguity in terminology.
    A concrete or "basic" :any:`Union` is referred to as a
    **basic constituent** or **constituent**,
    and a group of these is referred to as a **generic union**.
    Any geometric sampling object is therefore a :any:`Union`,
    but not every sampling object is a "generic union".
    The reason the class is called :any:`Union` and everything
    is therefore a "union", is because
    the generic union operations are implemented as
    fallback methods that superclasses fully or partially override,
    and the union class is a convenient place for doing generic
    operations on constituents (and generic unions).

    Union and difference are implemented
    as + and - operations, which always produce a
    :any:`Union`. Moreover, + always produces a generic union.
    Recall that in a-b, the `minuend` is a,
    the `subtrahend` is b, and in a+b, both a and b are `summands`.
    With this terminology fixed, the user is responsible for respecting
    these restrictions:

    - **summands.**
        Summands must always be disjoint.
        The reason is not because they will
        fail, but because this will invalidate the measure computations.
        There is an exception at the boundaries,
        which can intersect.

    - **subtrahends.**
        The subtrahend must be fully contained in
        the minuend. The same exception about boundaries
        applies as with the first rule.

    - **minuends.**
        The minuend of
        a difference must be a basic constituent.

    In other words, a generic union
    will not do the work of separating
    the subtrahend and giving the separate pieces
    to each constituent that concerns it.
    That work must be done during preprocessing.

    These rules are restrictive compared to the general
    boolean algebra of set theory, but they
    nevertheless correspond to a certain version of
    intuition, what might be called an intuition of
    "fleshy" shapes, or essentially "fruit logic" .

    Generic unions are formed out of references to the
    constituents, so you can form the generic union,
    and then difference out pieces afterwards, if you wish::

        A = Circle(r=2, ...)
        B = Triangle(...)
        C = A + B
        D = Circle(r=1, ...)
        A -= D # A is an annulus and C is updated.

    Some properties are enforced by checks that deliver errors.
    For example, += operations are only allowed for generic union,
    to avoid generating spurious references to voids.
    Conversely, differences must only be carried out
    on a basic constituent, and to encourage this,
    you are only permitted to perform a difference
    using a -= operation (ITCINOOD, in any case the principle is good).

    The + and - operations are also not guaranteed to
    possess every mathematical property
    of a union or intersection (respectively).
    For resolving questions/issues, consult the
    documentation or access the source code.

    Unions must always be initialized before sampling,
    by calling **init**.
    This is mainly so that they can be parametrized,
    and parameters can be stored in a standard location,
    :any:`Parameters`.
    After initialization, Unions have two
    dimension values, ``dim`` and ``idim``:

    - dim:
        dimension of the external or "ambient" space.
        For example, a flat 2D shape in 3D has dim 3.
    - idim:
        internal dimension.
        For example, a flat 2D shape in 3D has idim 2.

    A :any:`Union` where ``mode`` is None will not be able
    to generate a sample set,
    but it nevertheless can have testable geometry.
    It can be used as a differencing set,
    i.e., to place a hole (or "void") in a sample set,
    a basic technique in solid constructive geometry.
    Any :any:`Union` can be used for this purpose,
    but it is technically more efficient
    to use mode=None sources, because this
    doesn't set up a sampler. It is also a good habit
    to encourage, because then the ``complement``
    operation (possible only on voids)
    will always be available when needed.
    This operation, which only comes up from time to
    time, will now be explained.

    The complement can be useful if, instead of
    removing a shape from a sample, you wish
    to "restrict" the sample to a subset.
    For example::

        C = Cube(...)
        S = Sphere(..., mode = None)
        S.complement()
        C -= S

    Now whenever ``C`` is sampled,
    it will be like sampling ``S``. This can be
    used to construct a basic constituent's sampling
    apparatus.

    The ``complement()`` operation is used
    for this purpose and this purpose alone.
    Thus, in order to avoid confusion,
    the complement operation is only allowed if the ``mode``
    is ``None``, meaning that no sampling is possible
    from this union. (It is certainly not possible to guarantee
    sampling from such an object.)
    There are these additional conditions to observe
    if complements are used:

        - **disjointness.**
            All complements in a :any:`Union`
            must be disjoint.

        - **voids with respect to complements.**
            All voids in the :any:`Union` where
            complements are present must
            lie fully inside of the complements.

    Unlike in the case of minuends (i.e. voids) which must
    lie inside of a single basic constituent, it is possible
    for a void to lie inside only one, or of two or more
    complemented voids.

    The SPL is used to specify a sample set size.
    The absolute sample size ``N`` is derived from this value
    relying on the dimension ``n`` and the measure ``m``::

        N = floor( SPL**n * m )

    A union can contain a mixture of shapes of different
    (internal) dimensions. For example (pseudocode only,
    it can be fully implemented easily using :any:`Box90`)::

        X = Square(...) # 2x2 square
        left = LineSegment(...) # the left-hand side of X
        right = LineSegment(...) # the right-hand side of X
        bottom = # the bottom of X
        top = ... # the top of X
        Y = X + left + right + bottom + top
        m1 = X.measure() # 4
        S1 = X(SPL=3) # size: 3**2*m1
        N1 = size(S1) # 36
        m2 = Y.measure() # 4
        S2 = Y(SPL=3) # size: 3**2*m2 + 3*2 + 3*2 + 3*2 + 3*2
        N2 = size(S2) # 60

    .. note::

        To create a basic constituent class,
        inherit from :any:`Union` and
        implement these "impl" classes:

            - ``init_impl``
            - ``sample_impl``
            - ``inside_impl``
            - ``measure_impl``
            - ``bounding_box_impl``
            - ``internal_dimension_impl``

        Any class can be used as a template,
        for example, :any:`Box90`.
        Also see the warnings sprinkled in the
        :any:`Union` documentation below.

        To implement the ``sample_impl`` method,
        the following arguments should be provided:

            - ``SPL``:
                the number of samples,
                per unit length.
            - ``Nmin``:
                the minimum number of
                samples, if any, to be required.
            - ``pow2``:
                whether to coerce (by rounding up)
                the size N of the sample set
                to a power of 2.
            - ``convex_hull_contains``:
                whether to enforce the condition
                that the convex hull of the sample
                set contains the underlying
                domain. This condition
                can only be met if the
                (underlying) domain is an
                object whose geometric
                boundary is polygonal.
                If this is not true, then the
                condition ``convex_hull_contains``
                is passed over silently.
                For a :any:`DataSet` , the condition
                is considered to be always satisfied.

    Arguments:

        mode (optional string):
            A sampling mode or None, if no sampling is requested.
            The latter is the case for a "void" source
            (one that is present only for the purpose of differencing
            away from another :any:`Union`).
            For the list of possible modes, see :any:`UnitHypercube`.
        SPL (optional positive integer):
            A sample per unit length value which, if set,
            overrides the problem-wide definition.
            (Default: None)

    """

    # todo (April 2024)
    #  in addition to ConstantDim(1.234),
    #  also allow ExtendDim(1.234, 2.345).
    #  This "extends" the shape through an interval
    #  instead of embedding it in a single time slice like ConstantDim.
    #  .....this would allow to construct cylinder, prism, etc.
    #  while avoiding code redundancies.
    #  (i.e. a separate "Cylinder" class is not a long-lasting idea.)
    #  ....There are two ways you could sample:
    #  one is to sample normally, as in from a bounding box sampler.
    #  e.g., for a cylinder in 3D, you might want
    #  3D pseudo-random sampling.
    #  But you could achieve a sample set more cheaply with 2D sampling
    #  "extended" to 3D with a 1D sampler to handle the additional dimension(s).
    #  ...I would like to test to see this - i.e., test effects (real, not imagined)
    #  of "extended" sampling versus the other "embedded" kind.
    #  ...This study would be worth publishing if done carefully,
    #  because regardless of what it proves one way or the other
    #  it would put some questions to rest.
    #  ...To set up for this study, implement Union + ExtendDim
    #  with the freedom to have the expensive "bounding box sampler"
    #  method(s) and the cheaper "rev2" style sampler.
    #  (perhaps terminology 'embedded sampling' vs. 'extended sampling' can be used.)
    #  This code could go into Union and essentially just sit there
    #  forever after that (only updating the documentation with heuristics, advice, etc.),
    #  and it would be very general.


    def __init__(
            self,
            mode = None,
            SPL = None,
    ):
        super().__init__()
        self.sampler = None
        self.mode = mode
        # construct these lists using +, -.
        # This should be maintained as a single point of entry,
        # do not edit these lists manually.
        self.union = []
        self.voids = []
        self.complemented = False
        self.SPL = SPL


    def init(
            self,
            # todo fw float64
            dtype = float64,
            parameters = None,
    ):
        """
        Initialize the geometry by de-parametrizing it
        and setting the dimensions.

        .. warning::

            When implementing a :any:`Union` the ``init_impl`` method must:

                - have signature parameters --> None,
                - handle parametrized sources,
                - set the dimension and internal dimension (``dim`` and ``idim``)
                - define the random sampler, if ``mode`` is set,
                - perform any final checks that the :any:`Union` is well defined.

        Arguments:

            dtype (dtype):
            parameters (:any:`Parameters`):

        """
        if not self.initialized:
            super().init(dtype, parameters)
            dim = None
            if self.is_constituent():
                self.init_impl(parameters)
            else:
                for source in self.union:
                    source.dtype = dtype
                    source.initialized = True
                    source.init_impl(parameters)
                    dim = source.dim
                    for void in source.voids:
                        void.init(
                            dtype=dtype,
                            parameters=parameters,
                        )
                for void in self.voids:
                    void.init(
                        dtype=dtype,
                        parameters=parameters,
                    )
                self.dim = dim


    def __contains__(self, p):
        """
        Alias of ``inside``.

        """
        return self.inside(p)


    def sample(
            self,
            SPL,
            Nmin = None,
            pow2 = False,
            convex_hull_contains = True,
    ):
        """
        Sample the domain. This common operation
        can be performed by calling the :any:`Union`.
        The SPL is used to determine the sample size::

            A = Square(s=2) # 2x2 square, it is just pseudocode
            points = A(SPL=5) # the sample size is 4*5 = 20

        .. warning::

            If implementing a :any:`Union` the ``sample_impl`` method
            must consider the list of voids and avoid
            sampling points from them, because this step is *not*
            handled generically.

        .. note::

            The sample set is not shuffled,
            so it may not be fully randomized as-is.
            This step is taken by :any:`ConstraintSampleSet`.

        Arguments:

            SPL (integer):
                Sample size per unit length. For sources with (internal) dimension
                higher than 1, multiply this by the (internal) dimension
                to obtain the number of samples per unit (area, volume, etc.)
            Nmin (integer):
                Absolute minimum number of points to sample.
                If there are constituents this is applied to each one separately, ITCINOOD.
            pow2 (boolean):
                Whether to require generation of N number of points, N a power of 2.
                This will round up from the SPL.
                TO DO: support for union sources.
            convex_hull_contains (boolean):
                Whether to require that the convex hull of the sample set
                contains the the sample set. If the boundary is curved,
                this requirement may be violated.

        Returns:

            numpy array: sample set

        """
        if not self.initialized:
            raise UninitializedSource(f"Attempted to sample from an uninitialized Source.")
        # know: initialized, so dim is defined.
        if self.is_constituent():
            # > constituent, it can be sampled
            out = self.sample_impl(
                SPL=SPL,
                Nmin=Nmin,
                pow2=pow2,
                convex_hull_contains=convex_hull_contains,
            )
        else:
            # > not a constituent - in order to sample, sample its constituents
            # todo - test calling sample() instead of sample_impl, and see if a
            #  union may contain unions - a test can be a union of a pair of concentric annuli
            out = torch_zeros((0, self.dim))
            for source in self.union:
                S = source.sample_impl(
                    SPL=SPL,
                    Nmin=Nmin,
                    pow2=pow2,
                    convex_hull_contains=convex_hull_contains,
                )
                out = torch_vstack((out, S))
        return out


    def inside(
            self,
            p,
    ):
        """
        Whether point ``p`` is inside the region. It is
        most convenient to call this using builtin ``in``::

            S = Union(...)
            if p in S:
                # ...

        .. warning::

            When implementing a :any:`Union`
            the ``inside_impl`` method does not need
            to consider any voids present.
            That step is handled generically.

        Arguments:

            p (list of scalar):
                Point to be evaluated.

        Returns:

            boolean

        """
        out = False
        checklist = [self] if self.is_constituent() else self.union
        for source in checklist:
            found = source.inside_impl(p)
            if found:
                for void in self.voids:
                    found &= not void.inside_impl(p)
            if found:
                out = True
                break
        return (not out) if self.complemented else out





    ######################################################
    # Constructive Geometry Operations: +, -, +=, -=
    #
    # IF:
    # You do not edit self.union, self.voids manually.
    # THEN:
    # Invariants:
    # - the union list of any :any:`Union` is 100% constituents.
    # - the voids list of any :any:`Union` is 100% constituents.
    # - generic unions have voids == [].


    def __add__(self, b):
        out = Union(
            mode = None,
        )
        out.union += [self] if self.is_constituent() else self.union
        out.union += [b] if b.is_constituent() else b.union
        return out

    def __iadd__(self, b):
        if self.is_constituent():
            raise ValueError(f"Addition in place can only be performed on a generic Union.")
        self.union += [b] if b.is_constituent() else b.union
        return self

    def __sub__(self, b):
        raise ValueError(f"Differencing of unions must be performed in place: use (-=) and not (-).")

    def __isub__(self, b):
        self.voids += [b] if b.is_constituent() else b.union
        return self
        # optimize:
        #  check bounding boxes,
        #  if source is outside,
        #  do not add to list.


    def measure(self):
        """
        The measure (volume, length, etc) of the source.

        This procedure proceeds by computing all
        constituent measures with the internal dimension
        of the source, and subtracting from this
        all void measures with the internal dimension
        of the source.

        .. warning::

            When implementing a :any:`Union`,
            the ``measure_impl`` method does not need to consider
            any voids present - that step is handled generically.

        Returns:

            nonnegative scalar

        """
        if not self.initialized:
            raise UninitializedSource(f"Attempted to get measure from an uninitialized Source.")
        if self.is_constituent():
            out = self.constituent_measure()
        else:
            out = 0.0
            idim = self.internal_dimension()
            for source in self.union:
                if source.internal_dimension_impl() == idim:
                    out += source.constituent_measure()
        return out


    def bounding_box(self):
        """
        A bounding box for the source's
        geometric domain.

        .. warning::

            Not guaranteed to be minimal, but it will be
            provided that all constituent classes used return
            minimal bounding boxes.

        Returns:

            :any:`BoundingBox`

        """
        if not self.initialized:
            raise UninitializedSource(f"Attempted to get bounding box from an uninitialized Source.")
        out = BoundingBox(dim=self.dim)
        if self.is_constituent():
            out += self.bounding_box_impl()
        else:
            for source in self.union:
                out += source.bounding_box_impl()
        return out



    def internal_dimension(self):
        """
        Internal dimension for a union,
        immediate based on the values from
        ``internal_dimension_impl`` returned by
        the constituents.

        Returns:

            scalar ≥ 0

        """
        if self.is_constituent():
            out = self.internal_dimension_impl()
        else:
            out = max([u.internal_dimension() for u in self.union])
        return out






    def complement(self):
        """
        A restricted complement operation.
        Triggers an inversion of the containment test.
        For user's safety/sanity, use of this operation
        is forbidden unless the sampling mode is
        set to ``None``.

        """
        if self.mode:
            raise ValueError(f"Cannot take the complement of a Union if a sampling mode is set (mode = {self.mode})")
        self.complemented = not self.complemented


    def is_constituent(self):
        """
        Test if the :any:`Union` is a basic constituent.

        :meta private:
        """
        out = 'init_impl' in dir(self)
        return out


    def constituent_measure(self):
        """
        Private helper for measure().

        Returns:

            scalar

        :meta private:
        """
        shell = self.measure_impl()
        fill = 0.0
        comp = 0.0
        for void in self.voids:
            if void.idim == self.internal_dimension_impl():
                if void.complemented:
                    fill += void.measure_impl()
                else:
                    comp += void.measure_impl()
        out = fill - comp if fill > 0.0 else shell - comp
        return out


    def __str__(self):
        # draft
        out = str(self.union) + ", " + str(self.voids)
        return out


