






from math import cosh
from torch import (
    tanh as torch_tanh
)

def _sech2(x, a = 1.0):
    """
    secant squared
    """
    c = cosh(a*x)
    return 1.0/(c*c)

def _arcsech2(y, a = 1.0):
    """
    inverse of secant squared
    """
    tol = 1e-9
    if y > 1.0 or y < 0.0:
        return None
    if y == 1.0:
        return 0.0
    if y <= tol:
        return 1e10
    delta = 1e-1
    t = delta
    while delta > tol:
        # check t
        if y < _sech2(t, a):
            # > we're not past it
            t = t + delta
        else:
            # > we're past it
            # step back by delta
            t = t - delta
            # push down delta
            delta = delta * 1e-1
            # start checking again
    return t

default_epsilon = 0.001

def _get_a(T, epsilon):
    """
    Equation coefficient a from the input scalar T.
    """
    A = _arcsech2(epsilon, 1) if epsilon != default_epsilon else 4.146774726
    a = 2*A/T
    return a



class SmoothStep:
    """
    A smooth step function, built from tanh
    and parametrized. The function is strictly increasing
    and very close to a constant outside of the transient
    region, whose width is specified during initialization.

    Arguments:

        x0 (optional scalar):
            Where the transition occurs. Default: 0
        y1 (optional scalar):
            Lower value (to transition from) as x increases. Default: 0
        y2 (optional scalar):
            Upper value (to transition to) as x increases. Default: 1
        T (optional scalar):
            A width (x-interval) inside of which the transient
            region is bounded. Outside of this region,
            the slope of the step function is within a
            tolerance of zero, defined by the argument ``epsilon``.
            Default: 0.1
        epsilon (optional scalar):
            tolerance magnitude defining the stable regions where
            the smooth step function is approximately constant.
            Default: 0.001

    """
    def __init__(
            self,
            x0 = 0.0,
            y1 = 0.0,
            y2 = 1.0,
            T = 0.1,
            epsilon = default_epsilon,
    ):
        self.x0 = x0
        self.y1 = y1
        self.y2 = y2
        self.a = _get_a(T, epsilon)

    def init(
            self,
            x0 = 0.0,
            y1 = 0.0,
            y2 = 1.0,
            T = 0.1,
            epsilon = default_epsilon,
    ):
        """
        An optional init() routine for __init__/init parameter setting.
        """
        self.x0 = x0
        self.y1 = y1
        self.y2 = y2
        self.a = _get_a(T, epsilon)


    def __call__(self, x):
        y1, y2, x0, a = self.y1, self.y2, self.x0, self.a
        return (y2-y1)/2.0*(torch_tanh(a*(x-x0))+1.0)+y1


    def arcsech2(self, x, a = 1.0):
        """
        Scalar-to-scalar function, for testing.
        """
        return _arcsech2(x, a)


