





class DataResidual:
    """
    Residual for a data-driven constraint.
    This class is only used internally, in order to allow
    the solver to detect a data-driven constraint.

    Arguments:

        labels (optional string):
            Sets the format of the output data.
            Output labels can be higher derivatives.

    """

    def __init__(
            self,
            labels = None,
            # label (optional string):
            # The same as ``labels``, for readability
            # and (mainly) backwards compatibility.
            label = None,
    ):
        self.labels = labels if labels is not None else label



class Periodic(DataResidual):
    """
    Residual for periodic constraint,
    a type of data-driven constraint.

    For a periodic constraint, the
    labels defined here are pinned on the source
    and target of the callable transform.

    .. note::

        A periodic constraint with respect to time, say,
        ``u(x, y, tinit) = u(x, y, tfinal)``
        cannot be solved this way, however you can model this
        by defining a ``false'' time label such as ``s``, say,
        and using a :any:`TimeIndependent` engine.
        The label ``t`` is reserved for time-stepping.

    Arguments:

        transform (callable):
            Transform of the source to the target region
            for periodicity. This transform should map equivalent
            points to their analogs in the target region.
            Signature: ``(x, problem)`` where ``x`` is the
            problem inputs, minus time.

        labels (string):
            A labels string.

    """

    def __init__(
            self,
            transform,
            labels = None,
            label = None,
    ):
        super().__init__(labels, label)
        self.transform = transform



