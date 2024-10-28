






class Strategy:
    """
    Base class for strategies,
    mainly for organizational purposes.

    """

    def __init__(self, name):
        self.name = name

    def init(self, phase):
        raise NotImplementedError(f"[Strategy:{self.name}] init method undefined")

    def __str__(self):
        out = f"Strategy:{self.name}"
        return out




