






class RotateDict:
    """
    A RotateDict is effectively a list, and works like :any:`RotateList`.
    It may be rotated by simply making use of a additional zero counter.
    The underlying dict is the field ``dict``.
    The key for an item from its list-index is the field ``key``.
    Given RotateDict ``x``, the value for integer i may be obtained ``x[i]``,
    and the key for integer i may be obtained ``x.key[i]``.
    Intended use is with small dicts housing
    singleton data structures as values.

    Off the end errors will not occur as per normal for a list,
    although the input is not treated strictly like a modular integer.
    Off the end occurs at 2n instead of n, unless it's rotated.

    Parameters:

        x (dict):
            input

    """

    def __init__(self, x):
        self.dict = x
        self.key = []
        for lb in self.dict:
            self.key.append(lb)
        # iter counter
        self._i = 0
        # zero pointer
        self._z = 0


    def rotate(self, back = False):
        """
        Rotate the list:
        iterations and indexes
        will be moved left or right.
        """
        length = len(self.dict)
        if not back:
            self._z += 1
            if self._z == length:
                self._z -= length
        else:
            self._z -= 1
            if self._z == -1:
                self._z += length


    def __iter__(self):
        return self


    def __next__(self):
        if self._i >= len(self.dict):
            self._i = 0
            raise StopIteration
        else:
            i = self._z + self._i
            length = len(self.dict)
            if i >= length:
                i -= length
            self._i += 1
            return self.dict[self.key[i]]


    def __len__(self):
        return len(self.dict)


    def __getitem__(self, idx):
        """
        Get the ith counting from the
        zeroth item.
        """
        i = self._z + idx
        length = len(self.dict)
        if i >= length:
            i -= length
        return self.dict[self.key[i]]


    def __str__(self):
        out = ""
        for x in self:
            out += str(x)
        return out

