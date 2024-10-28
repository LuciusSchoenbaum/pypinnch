





class Encoding:
    """
    Base class for encodings.

    """
    def __init__(self):
        self._indim = None

    def init(self, indim):
        self._indim = indim

    def indim(self):
        return self.indim

    def outdim(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class BasicEncoding(Encoding):
    """
    Basic encoding from arXiv:2006.10739

    """
    def __init__(self):
        super().__init__()

    def init(self, indim):
        super().init(indim=indim)

    def outdim(self):
        return 2*self._indim



class PositionalEncoding(Encoding):
    """
    Positional encoding from arXiv:2006.10739

    Parameters:

        m (optional scalar):
            todo

    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def init(self, indim):
        super().init(indim=indim)

    def outdim(self):
        return 2*self.m*self._indim


class GaussianEncoding(Encoding):
    """
    Gaussian encoding from arXiv:2006.10739

    Parameters:

        sigma (optional scalar):
            standard deviation of a default ``b`` matrix.
            This default ``b`` is a matrix of size (m, indim) sampled
            from from N(0, \sigma^2), a standard normal distribution
            with std.dev. sigma.
            Heuristically, ``sigma=1`` for low frequency bias,
            ``sigma=10`` for high frequency bias appropriate for a natural image,
            ``sigma=100`` for very high frequency bias.
        m (optional scalar):
            See ``sigma``. Measures the density of selected dimensions.
            A problem-dependent parameter to be tuned. Insufficient density
            may affect results, over-sufficient density may be unnecessary.
        b (optional matrix-shaped tensor):
            custom ``b''. See source code/paper cited for guidance.
            Use if there exist directions of interest where
            high frequency effects are expected, STIUYKB.
            Note:  if ``b`` is set, ``m`` and ``sigma``, if set, are not used.

    """
    def __init__(
            self,
            sigma = None,
            m = None,
            b = None,
    ):
        super().__init__()
        self.sigma = sigma
        self.m = m
        self.b = b

    def init(self, indim):
        super().init(indim=indim)
        if self.b is not None:
            # nothing to do
            pass
        else:
            if self.sigma is None or self.m is None:
                raise ValueError(f"To use the standard/default b, require inputs sigma and m")

    def outdim(self):
        return 2*self.m


