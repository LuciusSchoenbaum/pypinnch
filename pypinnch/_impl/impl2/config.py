




class Config:
    """

    Base class for a config that
    accepts arguments, using ``config_args``
    in PyPinnchRun.

    """

    def __init__(self, engine):
        # > user's choice
        self.engine = engine
        self.e = engine



