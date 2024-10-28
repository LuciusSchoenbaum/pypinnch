

from numpy import \
    float64 as np_float64


class UninitializedSource(Exception):
    pass


class Source:
    """
    A computational device for generating a sample set,
    or in other words, "a source of data".
    In PINN training, there are typically multiple sources of
    data (for example, interior domain and boundaries), hence
    the concept of "source" is more general than the concept
    of a dataset.
    So a :any:`Source` is either a dataset (the more traditional
    type of source), or something more mathematical in nature.

    Sources must always be initialized before sampling,
    by calling **init**.
    This is mainly so that they can be parametrized,
    and parameters can be stored in a standard location,
    :any:`Parameters`.

    """

    # todo I'm pretty certain
    #  all Sources should
    #  have sample() and __call__,
    #  review, cf. Parametrized

    # todo as far as a good base to build on,
    #  I think that Source should be a general
    #  base class with no SPL, then there should
    #  be, say, GenSource with SPL,
    #  then after that is Union, with Parametrized
    #  becoming a GenSource and DataSet becoming
    #  a plain Source.

    def __init__(
            self,
    ):
        self.initialized = False
        self.dtype = None
        self.dim = None

    def init(
            self,
            dtype = np_float64,
            parameters = None,
    ):
        """
        Initialize source.

        .. warning::

            A :any:`Source` init can have default values of None
            for dtype, parameters.
            During :any:`Source` init:

                - handle parametrized sources,
                - set the dimension (``dim``)
                - set the data type (dtype),
                - perform any final checks that the :any:`Source` is well defined.

        Arguments:

            dtype (dtype):
            parameters (:any:`Parameters`):

        """
        # > some boilerplate
        self.initialized = True
        self.dtype = dtype if dtype is not None else np_float64


    def __call__(
            self,
            SPL,
            Nmin = None,
            pow2 = False,
            convex_hull_contains = True,
    ):
        """
        Alias of ``sample``.

        """
        return self.sample(
            SPL,
            Nmin,
            pow2,
            convex_hull_contains,
        )


    def sample(
            self,
            SPL,
            Nmin = None,
            pow2 = False,
            convex_hull_contains = True,
    ):
        raise NotImplementedError



    def dimension(self):
        """
        The dimension of the points in the source,
        as a data set. Mathematically, the dimension
        of the space where the geometry is embedded.

        This method is guaranteed for all Source objects
        as part of an API for actions. The field ``dim``
        can also be used::

            S = Triangle(...) # triangle in 3D
            d = S.dim # 3
            d = S.dimension() # 3

        Returns:

            scalar

        """
        return self.dim


    def measure_term(self):
        """
        Interpretation of measure for the given dimension
        (printing helper method).

        Returns:

             string

        :meta private:
        """
        idim = self.internal_dimension()
        if idim == 0:
            out = "point (measure 1.0)"
        elif idim == 1:
            out = "length"
        elif idim == 2:
            out = "area"
        elif idim == 3:
            out = "volume"
        else:
            out = "measure"
        return out



    def internal_dimension(self):
        """
        The dimension of the dataset, which may be distinct
        from the dimension of the ambient space.
        For example, a 2D shape (e.g., a boundary)
        embedded in a 3D space has internal dimension 2.

        This method is guaranteed for all Source objects
        as part of an API for actions.

        Returns:

            scalar

        """
        raise NotImplementedError


    def bounding_box(self):
        """
        Partly for the purpose of visualization,
        a bounding box must be defined.

        """
        raise NotImplementedError


    def __str__(self):
        # stub
        out = str(self.initialized) + ", " + str(self.dtype)
        return out




