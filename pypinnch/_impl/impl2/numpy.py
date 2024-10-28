


# todo review


from torch import (
    Tensor as torch_Tensor,
    zeros_like as torch_zeros_like,
    ones_like as torch_ones_like,
    maximum as torch_maximum,
)
import numpy as np



def mesh_numpy(
        ranges,
        resolution,
        right_open,
        dtype = None,
        device = None,
):
    """
    Construct a regular array of points
    using a standard meshgrid method,
    formatted as an :any:`XFormat` input.

     Arguments:

        ranges (nonempty list of pair of scalar):
            length is ``d``, the dimension of the mesh,
            defines the ranges as well as the order
            of the coordinates.
        resolution (integer or list of integers):
            resolution ``n``, the number of points
            in the regular mesh along each dimension.
        right_open (boolean):
            whether to create a right-open regular mesh.
        dtype (optional torch dtype): dtype
        device (optional torch device): device

    Returns:

        numpy array
            an array of shape ``(n**d, d)``.

    """
    # todo fw

    # todo review, out does not require grad
    # > set up resolution list
    if isinstance(resolution, list):
        resns = resolution
    else:
        resns = len(ranges)*[resolution]
    # > build inputs to meshgrid
    sps = ()
    for rn, resn in zip(ranges, resns):
        if right_open:
            # It is surprising that this is so much trouble, but
            # the torch.linspace API is fairly rigid, ITCINOOD.
            delta = (rn[1] - rn[0])/resn
            end = rn[1] - delta
        else:
            end = rn[1]
        sp = np.linspace(start=rn[0], stop=end, num=resn, endpoint=True, dtype=dtype)
        sps += (sp,)
    Xout = np.meshgrid(*sps, indexing='ij')
    # > reshape/reformat Xout, I am not sure what is the most efficient way
    Xlist = ()
    for v in Xout:
        Xlist += (v.reshape((-1, 1)),)
    out = np.hstack(Xlist)
    return out





def get_taw_curve(B):
    """
    Subroutine to get a series showing the time adaptive weighting
    being applied, used by StrideMonitor, TAWeightingClinic
    :param B: action bundle
    :return: numpy array to pass to pyplot
    """
    taw = B.phase.strategies.taweighting
    weighting = B.phase.strategies.weighting
    tinit = B.phase.samplesets.icbase.t
    tfinal = tinit + B.phase.th.stepsize()
    num = 64
    T = np.linspace(tinit, tfinal, num=num, endpoint=True).reshape((num,1))
    if weighting.preemption_state:
        W = np.zeros_like(T)
    elif taw.finished():
        W = np.ones_like(T)
    else:
        W = taw.w(T)
    return np.hstack((T, W))



# todo this mixture of numpy/torch preceded move to torch,
#  quite possible that now it is no longer needed.


def maxgen(a, b = None):
    """
    General maximum function for
    number types, numpy arrays, torch tensors.

    Arguments:
        a:
        b:
    Returns:
        max(a,b), pointwise if needed
    """
    if isinstance(a, float):
        if b is None:
            b = 0.0
        return max(a, b)
    elif isinstance(a, np.ndarray):
        if b is None:
            b = np.zeros_like(a)
        return np.maximum(a, b)
    elif isinstance(a, torch_Tensor):
        if b is None:
            b = torch_zeros_like(a)
        return torch_maximum(a,b)
    else:
        raise ValueError(f"Unrecognized type for maximum.")


def onesgen(a):
    if isinstance(a, float):
        return 1.0
    elif isinstance(a, np.ndarray):
        return np.ones_like(a)
    elif isinstance(a, torch_Tensor):
        return torch_ones_like(a)
    else:
        raise ValueError(f"Unrecognized type for ones.")


# This was used at one point but no longer ITCINOOD
def zerosgen(a):
    if isinstance(a, float):
        return 0.0
    elif isinstance(a, np.ndarray):
        return np.zeros_like(a)
    elif isinstance(a, torch_Tensor):
        return torch_zeros_like(a)
    else:
        raise ValueError(f"Unrecognized type for zeros.")





def wgen(order, t0 = None, dt = None):
    """
    Generates temporal weighting functions,
    with an option to set the order
    for parabolic or cubic weighting.
    Weight function monotonically decreases
    from one at value t0
    until reaching zero at value t0+dt.
    Assumes sample points that weights are applied to
    satisfy t >= t0.

    Arguments:

        t0: time
        dt: time extent
        order: constant, linear, parabolic, cubic: 0, 1, 2, 3

    Returns:

        w: R -> R compatible with numpy/pytorch array/tensors.

    """
    if order < 0:
        def w(t):
            return zerosgen(t)
    elif order == 0:
        def w(t):
            return onesgen(t)
    elif order == 1:
        def w(t):
            s = (t - t0)/dt
            return maxgen(-s + 1.0)
    elif order == 2:
        def w(t):
            s = (t - t0)/dt
            return maxgen(-s*s + 1.0)
    elif order == 3:
        def w(t):
            s = (t - t0)/dt
            return maxgen(-s*s*s + 1.0)
    else:
        raise ValueError(f"wgen: order too high!")
    return w








