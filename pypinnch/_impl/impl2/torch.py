

from torch import \
    hstack as torch_hstack, \
    linspace as torch_linspace, \
    meshgrid as torch_meshgrid



def mesh(
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

        torch.tensor
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
        sp = torch_linspace(rn[0], end, steps=resn, dtype=dtype)
        sps += (sp,)
    Xout = torch_meshgrid(*sps, indexing='ij')
    # > reshape/reformat Xout, I am not sure what is the most efficient way
    Xlist = ()
    for v in Xout:
        Xlist += (v.reshape((-1, 1)),)
    out = torch_hstack(Xlist).to(device)
    return out





