





from .residual import (
    DataResidual,
    Periodic,
)
from .types import is_boundary_constraint



class Constraint:
    """
    Mathematical constraint consisting of
    a :any:`Source` and a residual which expresses
    the constraint on ML network outputs.

    Example: an ODE F(x,t,x',x'') = f(x,t) is applied
    to constrain a solution u on the the interior of a domain D.
    The source is the interior subdomain of D.
    Then the residual is F(x,t,x',x'') - f(x,t).

    During training, a :any:`Constraint` is unpacked
    and applied, in the order specified by the user,
    by applying batches of size given by the batchsize parameter.

    A label for each constraint is set via the ``constraints`` argument
    of :any:`Problem`.

    .. note::

        For a periodic constraint, you can use the :any:`Periodic` class.
        When you create a periodic constraint, you have in mind two
        sources, A and B, which are constrained as one source A, with B
        obtained by transforming A. We call this a "double source", AB for short.
        For example, suppose A is one bounding surface, and B is another bounding surface.
        Then let AB, geometrically, be the union of the two walls.

            - set the label to what you want to be equal on the double source AB.
                For example: label="u, u_x" means "I want u and u_x to be identical on AB."
                So the first order periodic constraint for a 1d output would be "u", where u is the output label,
                and you can further strengthen the condition by adding variables.
            - set the residual equal to a Period instance, and follow the guidelines for
                setting up a Period instance provided in the :any:`Periodic` class.

    Arguments:

        residual (callable):
            The callable constraint on ML outputs, usually derived from a PDE,
            or, in case of a periodic constraint, a callable Period instance.
            See example scripts.

        source (:any:`Source`):
            A source of data, or inputs to ML network.
            (None, in case of zero-dimensional problem.)

        custom_batch (:any:`CustomBatch`):
            An instance of CustomBatch
            that sets a transform and divisor
            for use in setting a custom batch;
            STIUYKB.

    """

    def __init__(
            self,
            residual = None,
            source = None,
            custom_batch = None,
    ):
        self.source = source
        self.residual = residual
        self.custom_batch = custom_batch
        self.is_boundary_constraint = None


    def init(self, label):
        """
        Called by :any:`Problem`.

        Arguments:

            label (string):

        """
        # todo this is kludgy but I'm not sure what would be better
        # boolean used for the generic "bc" weight.
        self.is_boundary_constraint = is_boundary_constraint(label)


    def measure(self):
        """
        Measure (volume, length, etc.) of domain.
        Does not account for 1-dimensional time axis.
        and in the exceptional 0-dimensional source case
        it returns 1.

        Returns:

            M (scalar):

        """
        M = self.source.measure() if self.source is not None else 1.0
        return M


    def transform(self, XX, problem):
        """
        Transform XX according to a Period instance's transform.
        To be called in case of a periodic constraint,
        otherwise a call to this method will fail (and is not needed).

        Arguments:

            XX: data to be transformed
            problem (:any:`Problem`):

        Returns:

            XX, transformed to pointwise equivalent
                values under the periodic condition.

        """
        return self.residual.transform(XX, problem)


    def __str__(self):
        out = ""
        if isinstance(self.residual, DataResidual):
            if isinstance(self.residual, Periodic):
                out += "Periodic:\n"
            else:
                out += "Data:\n"
        if self.source is not None:
            out += f"{self.source.measure_term()}: {self.source.measure()}\n"
        return out



