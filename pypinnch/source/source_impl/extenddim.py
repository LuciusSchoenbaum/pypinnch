





class ExtendDim:
    """
    Used to establish an extension of a geometry
    into a coordinate dimension.

    A common elementary example is a cylinder
    that extends a circle from 2D into 3D.
    Here is an example where the cylinder
    is extended from the x-y plane into the z axis::

        cylinder = Sphere90(
            radius = 1.0,
            center = [0.0, 0.0, ExtendDim(0.0, 3.0)],
        )

    Parameters:

        x (scalar):
            The lower limit
            defining the extended geometry.
        y (scalar):
            The upper limit
            defining the extended geometry.

    """
    # A way of thinking suggests
    # that ConstantDim and ExtendDim are
    # are super/subclasses under inheritance,
    # but I remain skeptical that this wouldn't
    # be more susceptible to human error.

    def __init__(
            self,
            x,
            y,
    ):
        if x >= y:
            raise ValueError(f"ExtendDim must have upper and lower bounds, but {x} >= {y}.")
        self.x = x
        self.y = y

    def __call__(self):
        return self.x, self.y

    def str(self):
        return f"({self.x}, {self.y})"




