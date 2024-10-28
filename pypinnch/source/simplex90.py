

from .source_impl import \
    Union, \
    BoundingBox, \
    ConstantDim





class Simplex90(Union):
    """
    An n-dimensional shape that is completely
    characterized as the convex hull of a set of points.
    This is a special kind of convex `polytope <https://en.wikipedia.org/wiki/Polytope>`_
    known as a `simplex <https://en.wikipedia.org/wiki/Simplex>`_.

    Similarly to :any:`Box90` and :any:`Sphere90`,
    a :any:`Simplex90` can be embedded in a higher-dimensional space,
    but only along coordinate axes.

    To create a shape, you must pass a list of coordinates,
    not a list of vertices.

    .. note::
        This is a stub.

    Example: a triangle in 2D space with vertices at (0,0), (0,1), (1,0)::

        triangle = Simplex90(
            # x coordinates, then y coordinates
            coordinates = [[0,0,1], [0,1,0]],
        )

    The same triangle, embedded in the x-y plane in 3D space::

        triangle_emb = Simplex90(
            coordinates = [[0,0,1], [0,1,0], ConstantDim(0)],
        )

    Parameters:

        coordinates (list of list of scalar or :any:`ConstantDim`):
            Vertices of the shape, arranged by ascending coordinate values.

    """

    def __init__(
            self,
            coordinates,
            mode = 'pseudo',
            SPL = None,
    ):
        super().__init__(
            mode=mode,
            SPL=SPL,
        )
        self.coordinates = coordinates
        self.idim = None

    def init_impl(self, parameters):
        """
        See :any:`Union`.

        """
        if callable(self.coordinates):
            self.coordinates = self.coordinates(parameters)
        dimension = len(self.coordinates)
        npt = len(self.coordinates[0])
        if npt != dimension+1:
            raise ValueError(f"Simplex90 of dimension {dimension} must have exactly {dimension+1} points, however, some may be defined using ConstantDim.")
        self.dim = dimension
        self.idim = len(list(filter(lambda x: not isinstance(x, ConstantDim), self.coordinates)))
        raise NotImplementedError("Simplex90 has not been implemented yet!")


    def sample_impl(
            self,
            SPL,
            Nmin,
            pow2,
            corners,
    ):
        """
        See :any:`Union`.

        """


    def inside_impl(self, p):
        """
        See :any:`Union`.

        """


    def measure_impl(self):
        """
        See :any:`Union`.

        """


    def bounding_box_impl(self):
        """
        See :any:`Union`.

        """


    def internal_dimension_impl(self):
        """
        See :any:`Union`.

        """
        return self.idim



