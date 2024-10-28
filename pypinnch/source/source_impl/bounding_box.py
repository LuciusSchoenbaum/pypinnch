





class BoundingBox:
    """
    A type providing a bounding box for a geometric source,
    and operations on these. A bounding box can be
    summed, like a :any:`Union`, but a :any:`Union` preserves
    the information of constituents (as well as voids).
    A :any:`BoundingBox` destroys this information,
    making it much more efficient at scale.

    The default bounding box is empty.
    Given a list of bounding boxes such an uninitialized
    bounding box can be easily accumulated::

        bb_total = BoundingBox(d)
        for bb in some_bounding_boxes:
            bb_total += bb

    Arguments:

        dim (integer): dimension

    """

    def __init__(self, dim):
        self.dim = dim
        self.mins = []
        self.maxs = []


    def __add__(self, bb):
        if self.dim != bb.dim:
            raise ValueError(f"Bounding boxes cannot be combined, dim {self.dim} != dim {bb.dim}")
        else:
            out = BoundingBox(dim = self.dim)
            if not self.mins:
                out.mins += bb.mins
                out.maxs += bb.maxs
            else:
                out.mins += self.mins
                out.maxs += self.maxs
                for i in range(self.dim):
                    if self.mins[i] > bb.mins[i]:
                        out.mins[i] = bb.mins[i]
                    if self.maxs[i] < bb.maxs[i]:
                        out.maxs[i] = bb.maxs[i]
        return out


    def __iadd__(self, bb):
        if self.dim != bb.dim:
            raise ValueError(f"Bounding boxes cannot be combined, dim {self.dim} != dim {bb.dim}")
        else:
            if not self.mins:
                self.mins = bb.mins
                self.maxs = bb.maxs
            else:
                for i in range(self.dim):
                    if self.mins[i] > bb.mins[i]:
                        self.mins[i] = bb.mins[i]
                    if self.maxs[i] < bb.maxs[i]:
                        self.maxs[i] = bb.maxs[i]
        return self


    def __str__(self):
        """
        The range in each dimension, e.g.,
        [range in x, range in y, range in z]
        """
        return f"[{', '.join([f'({x}, {y})' for x, y in zip(self.mins, self.maxs)])}]"

