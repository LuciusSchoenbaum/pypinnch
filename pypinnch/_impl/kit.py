



from numpy import \
    zeros as numpy_zeros



class Kit:
    """
    A record class for the optimizer parameters
    for a training procedure using iterative gradient descent.
    The default values are (perhaps) good typical values.

    Arguments:

        tolerance (floating point):
            tolerance to apply.
            Can be a list of tolerances in the case of graded training.
            todo review
        max_iterations: (floating point):
            Can be an integer of a list of integers
            (applied in the case of graded training).
            If it is a list, it must have length depth,
            where depth = log2(steps_per_stride).
        learning_rate (floating point):
            Learning rate's initial value.
        gamma (floating point):
            Learning rate's exponential decay rate.
            todo review
        lbfgs:
            A dict for LBFGS parameters.
        adamw:
            A dict for other AdamW and AMSGrad parameters.

    """


    def __init__(self,
        max_iterations = 10000,
        tolerance = 1e-10,
        learning_rate = 1e-03,
        gamma = 0.9,
        lbfgs = None,
        adamw = None,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lbfgs = {} if lbfgs is None else lbfgs
        self.adamw = {} if adamw is None else adamw


    def __len__(self):
        return 4


    def as_np(self, dtype):
        out = numpy_zeros((1,len(self))).astype(dtype)
        out[0,0] = float(self.max_iterations)
        out[0,1] = float(self.tolerance)
        out[0,2] = float(self.learning_rate)
        out[0,3] = float(self.gamma)
        return out


    def header(self):
        return "max_iterations, tolerance, learning_rate, gamma"


    def __str__(self):
        out = ""
        out += f"max_iterations: {self.max_iterations}\n"
        out += f"tolerance: {self.tolerance}\n"
        out += f"learning_rate: {self.learning_rate}\n"
        out += f"gamma: {self.gamma}\n"
        return out



