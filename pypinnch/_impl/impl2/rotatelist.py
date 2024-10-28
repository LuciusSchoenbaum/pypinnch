






class RotateList:

    def __init__(self, xs):
        self.list = xs
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

        length = len(self.list)
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
        if self._i >= len(self.list):
            self._i = 0
            raise StopIteration
        else:
            i = self._z + self._i
            length = len(self.list)
            if i >= length:
                i -= length
            self._i += 1
            return self.list[i]


    def __len__(self):
        return len(self.list)


    def __getitem__(self, idx):
        """
        Get the ith counting from the
        zeroth item.
        """
        i = self._z + idx
        length = len(self.list)
        if i >= length:
            i -= length
        return self.list[i]


    def __str__(self):
        out = ""
        for x in self:
            out += str(x)
        return out

