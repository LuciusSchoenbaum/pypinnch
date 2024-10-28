


from mv1fw import \
    get_fslabels, \
    parse_labels, \
    get_labels





class Moment:
    """
    The :any:`Moment` class provides support for a moment, where
    a **moment** is specifically defined as **a
    quantity appearing in a residual, which depends in a nonlocal
    way on non-temporal variables, but depends only on a pointwise
    (local) value of time.**
    Such non-temporal variables might be configuration variables,
    or phase space variables. In practice, a moment is derived from
    mathematical expression appearing in a
    partial differential equation.

    This concept is based on the concept of a
    moment as it is defined in kinetic theory (not the same as
    the concept of moment of angular rotation.)

    The :any:`Moment` class is defined similar to a the way that a
    :any:`Solution` or :any:`Reference` is defined, via a dict.
    If we compare a :any:`Moment` to a :any:`Solution`,
    for example, we can see the basic similarity which associates
    a functional signature with a computational method that the user
    is able to control/manage, a "callback", in other terms.

    The most easily noticed qualitative difference between a :any:`Moment`
    and a :any:`Solution` or a :any:`Reference` is that the latter two
    always goes to the :any:`Result` class for expression
    as output, while a :any:`Moment` does not.
    If you want it to go out of the solver (as output),
    you may do so by passing the moment directly through
    a :any:`Solution` .
    The functional difference (which is more fundamental)
    between a :any:`Moment` and a :any:`Solution` or a :any:`Reference`
    is that a :any:`Moment` is always necessary for "self-consistency"
    of training, whereas in the case of a :any:`Solution` or :any:`Reference,
    this is never the case. A :any:`Solution` cannot appear inside of a
    residual, and a computation that occurs to
    generate a :any:`Solution` is independent from the residual values
    computed during training. It is only possible to set up a
    :any:`Solution` to occur at timesteps, between timesteps,
    or periodically as timesteps advance.
    A :any:`Moment`, on the other hand, is a computation
    that impacrs the residual's value *during* training---without the
    moment, the neural network cannot be trained.
    In other words (in a certain sense) it is an input to the pde, which itself
    depends on the output to the pde. This indicates that a moment
    has the potential to add significant computational complexity to
    the training procedure.

    The interval between updates to the moment, measured in number
    of iterations, is set by the ``every`` parameter.
    Another parameter that affects how :any:`Moment` is computed
    is ``SPD``, set in :any:`TopLine`. The SPD is the **step partition delta**,
    or the chosen gap between slices where moments are calculated.
    There is an expected tradeoff between and small SPD (better accuracy)
    and a larger SPD (faster to compute).

    Example of a scalar-valued :any:`Moment` that might
    appear in a :any:`Problem` description::

        moments = {
            't; f0': Moment(
                every=1,
                methods={
                    'f0': f0_method,
                }
            ),
        }

    An example of a scalar-valued :any:`Moment`
    with an input variable 'x' which will program a calculation
    of the moment as a function of x in 30 regularly-spaced positions::

            moments = {
                'x, t; g0': Moment(
                    every=1,
                    methods={
                        'g0': g0_method,
                    },
                    resolution={
                        'x': 30,
                    }
                ),
            }

    The names ``f0_method``, ``g0_method``, etc. can be chosen arbitrarily.

    Parameters:

        methods (dict, labels -> callable):
            A dict associating an output label to a callback method.
            The output label could resemble the symbol ordinarily
            associated to the moment; we will use ``U`` as a placeholder.
            The callback method will be called every time the moment
            is updated. This callback has the form ``X, problem --> U`` .
            The input variable ``X`` permits the user to extract
            every variable specified in the input list,
            including time::

                def U_method(X, problem):
                    x = problem.get('x', X=X)
                    ...
                    t = problem.get('t', X=X)
                    ...
                    U = ...
                    return U

            This defines the moment computation
            and embeds it within a problem description.
        every (integer >= 1):
            How many iterations between each time the
            moment is updated, during training phases.
            The moment will always be updated
            at the beginning of the zeroth training phase (so that the
            moment is always defined), and after that
            at the beginning of the ``every``th numbered
            training phase.
        resolution (integer or dict string -> integer):
            Either a general, common resolution for all non-temporal
            input variables, or an assignment
            of distinct resolutions to each such input variable.
            Interpolation is done between a regular
            array of size ``resolution``, ITCINOOD.

            .. note::

                For the time dimension,
                you cannot set a "resolution" here.
                You can do this by setting
                the ``SPD`` parameter of
                :any:`TopLine`. This is because that parameter,
                unlike the ones called "resolution" that
                you set here, affects all of the sample sets.

    """

    # Note: this class simply formats user input.
    # the computation of moments is performed by
    #  :any:`MomentSets` and :any:`Problem`

    # todo: document the signature of a moment calculation method,
    #  and explain how to interpret `X` and `problem`

    # todo: document available tools in a moment calculation method


    def __init__(
            self,
            every = 1,
            methods = None,
            resolution = None,
    ):
        self.methods = methods if methods is not None else {}
        self.every = every
        self.resolution = resolution if resolution is not None else {}
        self.fslabels = None


    def init(
            self,
            labels,
    ):
        """
        (Not called by user.)

        Initialize the moment prior to training.

        """
        lbl, indim, with_t = parse_labels(labels)
        # > only extract the inputs from the key label string.
        lbl = lbl[:indim]
        # > get the output list straight from the methods dict
        for lb in self.methods:
            lbl.append(lb)
        # > set fslabels, a unique identifier string
        self.fslabels = get_fslabels(lbl, indim, with_t)
        # > the labels dict contains 't'.
        # otherwise, it is not a moment.
        if not with_t:
            raise ValueError(f"Moment inputs must include time (moment {get_labels(lbl, indim, with_t)}).")
        # > we forbid more than one outlabel
        #  for now, it is easier to implement.
        if len(lbl[indim:]) > 1:
            raise ValueError(f"A Moment must have only one output (moment {get_labels(lbl, indim, with_t)}).")
        # > populate resolution as dict
        if isinstance(self.resolution, int):
            res = self.resolution
            self.resolution = {}
            for lb in lbl[:indim]:
                self.resolution[lb] = res
        else:
            for lb in lbl[:indim]:
                if lb not in self.resolution:
                    raise ValueError(f"A moment has input without defining resolution (moment {get_labels(lbl, indim, with_t)}).")
            for lb in self.resolution:
                if lb not in lbl[:indim]:
                    raise ValueError(f"A moment resolution must define a resolution for the moment's input variable, not other problem variables (moment {get_labels(lbl, indim, with_t)}).")



    def __str__(self):
            out = ""
            # draft
            out += f"Moment {self.fslabels}\n"
            return out


