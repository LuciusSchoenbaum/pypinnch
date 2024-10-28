





class ConstantDim:
    """
    Frequently used when setting up "90" sources.

    Parameters:

        x (scalar):
            The value to which to pin the dimension.

    """

    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x





