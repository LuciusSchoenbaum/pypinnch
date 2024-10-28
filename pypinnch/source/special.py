






from copy import deepcopy

from .source_impl import \
    Union



class Special(Union):
    """
    A custom source that is interoperable with
    other Union sources, like :any:`Box90` and :any:`Sphere90`.

    Parameters:

        box (:any:`Box`):
            A :any:`Box90` source that serves as the
            bounding box for the source.

            .. warning::

                Try to set the internal dimension
                of the bounding box equal to the
                source's internal dimension.
                If this is not possible because the bounding
                box is a Box90, contact the developers
                with a feature request.

        inside (callable):
            A method that defines the inside of the
            source, which accepts a vector described (simply) as a Python list.
            Only points inside the bounding box are considered.
        measure (scalar):
            The measure of the geometric object, in its interior dimension.
            It must be specified, but it can be an estimate.
        mode (optional string):
            Choice of sampling algorithm, or None
            to disable sampling for this source.
            At present only pseudo is implemented ITCINOOD.
            (Default: "pseudo")
        SPL (optional positive integer):
            A sample per unit length value which, if set,
            overrides the problem-wide definition.
            (Default: None)

    """

    # To really get this right, a general Box must finally be implemented,
    # dropping the '90'.
    #
    # I will now pause to try to explain, in the lingo of the source code
    # below, why this is, and argue that it is not a pressing issue.
    # The internal dimension of the source (ITCINOOD)
    # is inherited from the box, like almost everything else.
    # This, however, is not correct in general.
    # For example, if the desired Special source is, say,
    # a 2D outline of an elephant that is not aligned with an axis -
    # if you like, imagine that it is walking from point (10, 0, 0) to
    # point (0, 10, 0), then it is not aligned with any of the axis planes -
    # then this code cannot produce the special source
    # without using a 3D bounding box. Indeed, if you tried this it
    # would not even work at all, because the sampling would be
    # a lost cause -- but that's beside the point! The point is that
    # the internal dimension is actually *incorrect*.
    # Therefore this code is actually *wrong*, until the issue
    # above is addressed. However, this is not a pressing issue,
    # because there is plenty we can do just with a Box90-wrapped
    # Special source. As long as the internal dimension of the
    # Box90 matches that of the Special, we are fine.
    #
    # One final comment (sorry): the Box itself
    # can be built using affine transforms of Box90
    # with little extra effort and little extra code.
    # So actually the thing to do is implement Box90
    # as a special case of Box, and change the name of
    # Box90's source to become the Box source.
    # I personally see little value in allowing for constructors of
    # any kind not based on a Box90 pattern, so this
    # would be an easy, painless update.
    # We would just have to mind whether
    # Box90's code, having been updated with all these
    # little affine transformations, would not break
    # something, somewhere---so a code review would
    # be necessary.


    # todo add an 'Example' to docstring as in Box90

    # todo finesse the parametrizations:
    #  - allow parametrized (callable) measure input
    #  - decorator pattern for parametrized inside() ???
    #  - document, document, document


    def __init__(
            self,
            box,
            inside,
            measure,
            mode = 'pseudo',
            SPL = None,
    ):
        super().__init__(
            mode=mode,
            SPL=SPL,
        )
        # > be careful, the user might by chance re-use the box
        self.box=deepcopy(box)
        # > propagate the mode to the bounding box,
        #  as it will do the sampling under the hood
        self.box.mode = self.mode
        self.inside=inside
        self._measure = measure


    def init_impl(
            self,
            parameters,
    ):
        """
        See :any:`Union`.

        """
        # > initialize the bounding box.
        self.box.initialized = True
        self.box.dtype = self.dtype
        self.box.init_impl(parameters)
        # todo I'm going to require that this is set up this way,
        # todo document this requirement,
        #  provide an example in the docstring
        self.inside = self.inside(parameters)
        # awk
        self.dim = self.box.dim


    def sample_impl(
            self,
            SPL,
            Nmin = None,
            pow2 = False,
            convex_hull_contains = False,
    ):
        """
        See :any:`Union`.

        """
        return self.box.sample_impl(
            SPL=SPL if self.SPL is None else self.SPL,
            Nmin=Nmin,
            pow2=pow2,
            convex_hull_contains=False,
            special=self.inside,
        )


    def inside_impl(self, p):
        """
        See :any:`Union`.

        """
        return self.inside(p)


    def measure_impl(self):
        """
        See :any:`Union`.

        """
        return self._measure


    def bounding_box_impl(self):
        """
        See :any:`Union`.

        """
        return self.box.bounding_box_impl()


    def internal_dimension_impl(self):
        """
        See :any:`Union`.

        """
        return self.box.internal_dimension_impl()

