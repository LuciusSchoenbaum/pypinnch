


# todo fw
from torch import \
    zeros

from .types import \
    parse_every

from mv1fw import (
    get_fslabels,
    parse_labels,
    parse_fslabels,
    get_labels,
)
from mv1fw.fw import (
    XFormat,
)



class Solution:
    """
    Houses a solution to be computed and stored
    by the solver. In precise terms, a :any:`Solution` is a
    time-independent quantity or set of quantities, to be
    computed at a timestep. A script can define any number of solutions
    in a dict format. Normally a solution method is a callable,
    but exceptional cases are the string 'direct', and a value ``None``.
    The string 'direct' must only be used when a label is a problem output label,
    i.e., it would be obtained as direct output from the solver.
    Here are a few example scenarios::

        # u, v are direct output, Q is a derived output
        solutions = {
            'x,y,t': Solution({'u': 'direct'', 'v': 'direct', 'Q': calculate_Q}, every=100),
        }
        # I would like to calculate Q more frequently than u, v
        solutions = {
            'x,y,t': Solution({'u': 'direct'', 'v': 'direct'}, every=100),
            'x,y,t': Solution({'Q': calculate_Q}, every=10),
        }
        # Q only depends on input variable y and not on x
        solutions = {
            'x,y,t': Solution({'u': 'direct'', 'v': 'direct'}, every=100),
            'y,t': Solution({'Q': calculate_Q}, every=100),
        }
        # I have validation data labeled x, y, t; Q, R but I don't want to solve for R
        solutions = {
            'x,y,t': Solution({'u': 'direct'', 'v': 'direct'}, every=100),
            'x,y,t': Solution({'Q': calculate_Q, 'R': None}, every=10),
        }

    Time series solutions can also be created. For example, to
    generate a block with only a time value column::

        solutions = {
            'x,y,t': Solution(every=10, {'u, v': 'direct'}),
            't': Solution(every=1, {'P': calculate_P}),
        }

    If a solution method is ``None``, data will be stored, but
    it will be meaningless (for example, zeros);
    this is done just so that result processing has stronger assumptions.
    If there is no validation data, there is probably no reason to have a
    method value of ``None``.
    A script can also define all labels at once, if they
    share a common method, for example, if they are all direct
    solver outputs, or if there is only one output to consider::

        solutions = {
            'x,y,t': Solution({'u, v': 'direct'}, every=100),
        }

    You must separate such labels by a comma; whitespace does not matter.

    Each :any:`Solution` will be stored as a block where inputs and
    outputs are stored (abstractly, as column data).
    In general, each such block can be any mixture
    of direct outputs, callable outputs, and 'skipped' or trivial outputs.
    This computation is executed and a new block of data is stored
    according to a common timestep stride, defined
    by the parameter ``every``.
    For example, if the timestep is ``0.01``, and it is sufficient
    to have data for every 0.1 interval of time, then you may
    choose to set ``every = 10``.

    Parameters:

        every (integer >= 1 or floating point):
            How many steps between each time the evaluation procedures are executed.
            Will execute before the 1st step (IC's) and after the every'th step.
            It is possible to pass fractions, e.g., 1/2 (twice per step),
            1/4 (four times per step), 1/100 (100 times per step), and so on.
            Note: the fractional evaluations are all applied at the end of the step.
            Instead of a fraction, the ``substep`` argument can be used, it is
            up to your preference.
        methods (dict, label -> None, callable, or string 'direct'):
            Assignment of evaluation procedures to individual labels.

            - None:
                The case when method is None is a case where
                a label set includes an output from a reference solver
                that we don't want to compute right now.
                We want to be able to tell the solver, "calculate A and B
                but don't worry about C" because this is less effort
                than reformatting the data from the solver, or modifying how
                we handle the data coming from that side.
            - callable:
                A callable method with signature (X, problem) ---> Q
                generating the target solution outputs from solver's direct outputs
                on a timeslice. The method should use the X parameter of the
                problem.get() method, for example::

                    def mymethod(X, problem):
                        x, y, z, t = problem.get("x, y, z, t", X=X)
                        q = ...
                        return q

            - string 'direct':
                Tells the solver that this solution is a direct output from the solver.
        resolution (optional integer or list of integer):

        substep (optional integer):
            If set, the substep is set, that is, how many times, per timestep,
            the solution is exported from the solver.
            Instead of a ``substep``, this command can be set via a fractional
            ``every`` value, it is up to your preference.
            Either way, if it is left unset, it will inherit the substep value set
            in the :any:`TopLine` instance.
        features (optional dict):
            associate features to solutions.
            Map an output label to a dict (string -> object).
            Possibilities: 'streamlines' to callable. To do: documentation for streamlines

    """

    def __init__(
            self,
            every = 1,
            methods = None,
            resolution = None,
            substep = None,
            features = None,
    ):
        whole, part = parse_every(every)
        self.every = whole
        if part > 1:
            if isinstance(substep, int):
                # substep is defined twice.
                # > raise error during init
                self.substep = part, substep
            else:
                self.substep = part
        elif isinstance(substep, int):
            self.substep = substep
        else:
            self.substep = None
        self.methods = methods
        if self.methods is not None and not isinstance(self.methods, dict):
            raise ValueError("A solution must specify at least one output label via methods argument, a dict string -> method.")
        self.fslabels = None
        self.features = {} if features is None else features
        self.resolution = resolution if resolution is None or isinstance(resolution, list) else [resolution]


    def init(
            self,
            inlabels,
    ):
        lbl, indim, with_t = parse_labels(inlabels)
        # > methods not specified, use a default value of direct evaluation
        if self.methods is None:
            self.methods = {}
            for lb in lbl[indim:]:
                self.methods[lb] = 'direct'
        # > modify label list
        # only extract the inputs from the key label string,
        # get the output list from the methods dict.
        lbl = lbl[:indim]
        for lb in self.methods:
            lbl.append(lb)
        # > set fslabels, a unique identifier string
        self.fslabels = get_fslabels(lbl, indim, with_t)
        # > type check the substep
        if isinstance(self.substep, tuple):
            raise ValueError(f"Solution {get_labels(lbl, indim, with_t)}: Substep multiply defined: {self.substep}")
        #> ensure self.methods is a dict mapping to {boolean, callable}
        # where False: skip, True: direct output, callable: indirect output.
        for lb in lbl[indim:]:
            #> adjust values
            m = self.methods[lb]
            if m == 'direct':
                self.methods[lb] = True
            elif m is None:
                self.methods[lb] = False


    def __call__(
            self,
            X,
            problem,
            QQref = None,
     ):
        return self.evaluate(X, problem, QQref)


    def evaluate(
            self,
            X,
            problem,
            driver,
    ):
        """
        (Not called by user.)

        Evaluate solution methods as a group on
        the input X, and include reference output,
        if there is any available.

        .. note::

            This method is called by :any:`Result`
            as part of the core code for result processing.
            It is not called elsewhere, or by any
            other actions - except for :any:`BaseMonitor` ITCINOOD.

        Arguments:

            X (:any:`XFormat`):
                Contains the sample set (input data) on which to evaluate the solution.
                X will be modified via the :any:`XFormat.append` method.
            problem (:any:`Problem`):
            driver (:any:`Driver`):

        """
        # device = driver.config.device
        # dtype = Ttype(driver.config.dtype)
        lbl, indim, with_t = parse_fslabels(self.fslabels)
        UU = None
        lblUU = None
        methods = self.methods
        for lb in lbl[indim:]:
            method = methods[lb]
            if callable(method):
                # > evaluate solution method on Xtgt
                Q = method(X=X, problem=problem)
            else:
                if method:
                    # direct output
                    # > evaluate the model(s)
                    # todo revise driver.evaluate and improve this procedure
                    if UU is None:
                        # > model hasn't been evaluated yet, evaluate
                        # todo multiple models
                        UU, lblUU = driver.evaluate_output(
                            X = X,
                        )
                    i = -1
                    for i0, lb0 in enumerate(lblUU):
                        if lb0 == lb:
                            i = i0
                            break
                    Q = UU[:,i:i+1]
                else:
                    # skipped output
                    # > create dummy space
                    Q = zeros((X.X().shape[0], 1))
            X.append(Q, lb)



    def __str__(self):
        out = ""
        # draft
        out += f"Solution {self.fslabels}\n"
        return out


