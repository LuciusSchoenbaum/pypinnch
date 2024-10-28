



from .source_impl.source import Source

from numpy import \
    loadtxt as numpy_loadtxt
from torch import \
    from_numpy as torch_from_numpy

from numpy import \
    float64

from mv1fw import (
    parse_labels,
)


class DataSet(Source):
    """
    A data set implemented as a NxD array,
    where D is the dimension and N is the data set size.

    Parameters:

        labels (string):
            labels string that formats data.
        numpy_data (optional Numpy array):
            target data, already set up.
        filename (optional string):
            filename where data is stored.

    """

    def __init__(
            self,
            labels,
            filename,
    ):
        super().__init__()
        self.labels = labels
        self.filename = filename
        self.data = None
        # if numpy_data is not None:
        #     self.data = numpy_data
        # elif filename is not None:
        #     self.data = np.loadtxt(filename)
        # else:
        #     raise ValueError


    def init(
            self,
            dtype = float64,
            parameters = None,
    ):
        super().init(dtype, parameters)
        self.data = numpy_loadtxt(fname=self.filename, dtype=dtype)


    def internal_dimension(self):
        _, indim, _ = parse_labels(self.labels)
        return indim


    def measure(self):
        return 1.0


    def get_labels(self):
        """
        (Not called by user.)

        Return the output labels targeted
        by the dataset.
        """
        lbl, indim, with_t = parse_labels(self.labels)
        if with_t:
            out = ','.join(lbl[indim+1:])
        else:
            out = ','.join(lbl[indim:])
        return out


    def reference_data_size(self):
        lbl, indim, with_t = parse_labels(self.labels)
        return len(lbl) - indim


    def __call__(
            self,
            SPL,
            Nmin,
            pow2,
            convex_hull_contains,
    ):
        """
        Return a time-independent sample set
        from which training batches can be taken.

        Arguments:

            SPL (integer):
            Nmin (integer):
            pow2 (boolean):
            convex_hull_contains (boolean):

        Returns:

            array

        """

        # todo
        #  Arguments (SPL, Nmin, pow2, etc) are ignored
        #  for the sake of simplicity

        # > check whether reformatting is necessary
        # todo
        # > reformat
        # todo
        out = torch_from_numpy(self.data)
        return out



