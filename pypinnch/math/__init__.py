__all__ = [
    "scalar",
    "eq",
    "pi",
    "exp1",
    "sqrtpi",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "sqrt",
    "exp",
    "sum",
    "arctan",
    "arctan2",
    "where",
    "ones_like",
    "zeros_like",
    "constant_like",
    "hstack",
    "vstack",
    "lor",
    "land",
    "min",
    "max",
    "SmoothStep",
]

from . import scalar

### convenience

pi = 3.141592653589793
sqrtpi = 1.772453850905515
exp1 = 2.718281828459045

# This allows for implementation of other backends.
# todo push back to fw?
from mv1fw.fw import XFormat

from torch import sin, cos, tan, arctan, arctan2
from torch import sinh, cosh, tanh
from torch import sqrt, exp
# where(a < b, if_true, if_false)
from torch import where
from torch import \
    ones_like as torch_ones_like, \
    zeros_like as torch_zeros_like, \
    full_like as torch_full_like
from torch import hstack, vstack
from torch import logical_or as lor
from torch import logical_and as land

from torch import max as torch_max
from torch import min as torch_min

from torch import sum as torch_sum

def max(x):
    return torch_max(x).reshape((1,1))

def min(x):
    return torch_min(x).reshape((1,1))

def sum(x):
    return torch_sum(x).reshape((1,1))

def ones_like(x):
    x_ = x.X() if isinstance(x, XFormat) else x
    return torch_ones_like(x_[:,0:1])

def zeros_like(x):
    x_ = x.X() if isinstance(x, XFormat) else x
    return torch_zeros_like(x_[:,0:1])

def constant_like(x, value):
    x_ = x.X() if isinstance(x, XFormat) else x
    return torch_full_like(x_[:,0:1], value)


def eq(LHS, RHS=0.0):
    """
    Convenience function to avoid
    copy errors due to minus signs.
    """
    return LHS - RHS


from .smoothstep import SmoothStep


# physical constants (SI)

boltzmann = 1.3807e-23 # J K-1
planck = 6.6261e-34 # J s
plankpi = 1.0546e-34 # J s, h/2pi
clight = 2.9979e8 # m s-1
#
echarge = 1.6022e-19 # C
emass = 9.1094e-31 # kg
pmass = 1.6726e-27 # kg
#
avogadro = 6.0221e23 # mol-1
gasR = 8.3145 # J K-1 mol-1
calorie = 4.1868 # J
gravg = 9.8067 # m s-2
# proton/electron mass ratio
pmoem = 1.8362e3 # 1

# plasma physics (work in progress)

egyro = 2.80e6 # B Hz
egyroo = 1.76e7 # B rad/sec
igyro = 1.52e3 # Z mu-1 B Hz
igyroo = 9.58e3 # Z mu-1 B rad/sec

efreq = 8.98e3 # ne 1/2 Hz
ifreq = 2.10e2 # Z mu-1/2 ni 1/2 Hz


