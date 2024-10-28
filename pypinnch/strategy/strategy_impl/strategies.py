





class Strategies:
    """
    Houses all the available strategies for :any:`Phase` or :any:`Engine`.

    Supports integer-lookup (like a list) and ``for..in`` loops.

    Parameters:

        strategies (optional list of :any:`Strategy`):
            A list of one or more strategies.

    """

    def __init__(
            self,
            strategies=None,
    ):
        if strategies is None:
            strategies = []
        for strat in strategies:
            self.__setattr__(strat.name, strat)
        self._i = 0

    def __len__(self):
        return len(self.__dict__) - 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration
        else:
            i = self._i
            self._i += 1
            return self[i]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise LookupError(f"[Strategies] No {idx}th strategy")
        i = 0
        x = None
        for y in self.__dict__:
            if i == idx:
                x = y
                break
            i += 1
        return self.__dict__[x]

    def using(self, strategy):
        """
        Whether or not training is using the requested strategy.
        Allows code to be more generic.

        Arguments:
            strategy (string):

        """
        return hasattr(self, strategy)

    def __str__(self):
        out = ""
        for strat in self.__dict__:
            out += str(strat)
        return out




