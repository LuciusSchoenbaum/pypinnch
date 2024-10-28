





# todo fw
from torch import \
    full_like as torch_full_like, \
    full as torch_full, \
    zeros as torch_zeros, \
    hstack as torch_hstack, \
    vstack as torch_vstack, \
    linspace as torch_linspace, \
    tile as torch_tile, \
    tensor as torch_tensor

from ..source import BoundingBox
from ..source.source_impl.source \
    import UninitializedSource
from .types import \
    timed, \
    get_index, \
    indexlist_to_gaps
from .impl2.grad import Grad
from .impl2.torch import mesh
from sys import \
    stderr

from mv1fw import (
    get_labels,
    parse_labels,
    get_fslabels,
    parse_fslabels,
    sortdown,
)
from mv1fw.fw import XFormat

hub_error_message = "A non-hub X appears where a hub was expected. " \
    "Did you write (problem, X) when you meant to write (problem, X=X)?"



class Problem:
    """
    Houses the problem inputs, including all constraints,
    and hosts various methods that can be performed using
    only this information.

    .. note::

        Currently, IC Constraints can only be scalar quantities, as in::

            {'u': u0} # 'u' must be a scalar quantity

        So if you have two outputs, say u and v, you must do them separately,
        for example::

            {'u': u0, 'v': v0}

        Future: allow IC Constraints to set vector quantities.
        There's nothing difficult about this, but it requires some easy
        refactoring and thinking about how to implement an
        interface that simulation scripts and problem.get can use
        to receive/return vector quantities.
        (That is, values with shape (N, M) with M > 1).

    Arguments:

        labels (string):
            Labels defining the problem's inputs and outputs.
            Labels are what could also be called variables, names, or identifiers.
            Labels assign a name using commas (csv, comma-separated values)
            and a semicolon::

                <inputs csv> ; <outputs csv>

            Time label is always "t".
            It is not possible to use an underscore "_" or a hyphen "-" character in
            any label (or the program will not run).
            Whitespace in a label is theoretically possible, but not recommended.
            The same goes for punctuation characters or other unusual nomenclature.
            Example: ``"x, y, t; u"``. The inputs are x, y, t and the output is u.
        Parameters:
            The parameters type defined by the user. For example::

                class Parameters:
                def __init__(self):
                alpha = 1.234
                Q = 2.345
                v_init = 3.456
                # etc.

            This class is never used internally,
            and can always be accessed via the ``problem.p`` field.
        constraints (dict of labels: :any:`Constraint`):
            The constraints defined for the problem.
        solutions (dict of labels: :any:`Solution`):
            Directives for calculating quantities from the solver, or "solutions".
            These computations are carried out by the :any:`Result` action.
        references (optional dict of labels: :any:`Reference`):
            Reference solutions.
        moments (optional dict of labels: :any:`Moment`):
            Directives for calculating moments, if any are needed
            during training and the calculation of batch residuals.
        ic_constraints (dict):
            dict of the form {string: ic}, where ic is either a scalar or a callable.
            Note: in spite of sharing a similar name with the `constraints` dict,
            the `ic_constraints` dict is actually more comparable to the `solutions` dict
            or the `references` dict.
        ic_source (optional :any:`Source`):
            A source that describes the sample on which to
            train the IC's. Using :any:`Union` operations,
            it can be modified. It is None if the problem is time-independent,
            and conversely, a time-dependent problem requires an ic source.
        finalizers (optional list of callable):
            Callable routines that will be called as the solver closes.
            Can be used for ad-hoc data harvesting.
            Signature should be one argument, the engine,
            similar to a config method.
        file (string):
            Pass `__file__` if the problem is isolated in its own file.
            Otherwise, you may ignore this option.

    """


    def __init__(self,
        labels,
        Parameters,
        constraints = None,
        solutions = None,
        references = None,
        moments = None,
        ic_source = None,
        ic_constraints = None,
        finalizers = None,
        file = None,
    ):
        self.file = file
        self.background = None
        self.grad_ = Grad()
        self.p = Parameters()
        lbl, indim, with_t = parse_labels(labels)
        self.fslabels = get_fslabels(lbl, indim, with_t)
        self.lbl = lbl
        self.indim = indim
        self.with_t = with_t
        self.constraints = {} if constraints is None else constraints
        self.ic_constraints = {} if ic_constraints is None else ic_constraints
        self.solutions = {} if solutions is None else solutions
        self.references = {} if references is None else references
        self.moments = {} if moments is None else moments
        # todo deprecated, remove soon
        # self.handle = 'None'
        self.finalizers = finalizers if finalizers is not None else []
        self._review_labels()
        self._review_icc_labels()
        dict_fields = [self.constraints, self.ic_constraints, self.solutions, self.references, self.moments]
        dict_names = ['constraints', 'ic constraints', 'solutions', 'references', 'moments']
        for field, name in zip(dict_fields, dict_names):
            if not isinstance(field, dict):
                raise ValueError(f"{name} must have dict type.")
        self.th = None
        self.SPD = None
        # Remind: problem has copy of .out
        # so that it can be examined via clinics,
        # which is convenient so don't remove it.
        self.out = None
        self.ic_source = ic_source
        #####################
        # Driver-dependent fields
        # todo remind, the strategy for multi-driver ITDCINOOD
        #  is to allow problem to be deep-copied by each driver,
        #  who can associate himself to the problem by updating
        #  this reference in the copy. Each deep-copy of problem
        #  has its own grad memoizing system - therefore "morally"
        #  the grad memoizing system i.e. self.grad_ should be
        #  in the driver class and accessed via this little routing switch.
        #  This update should be made early in multi-driver stage.
        self._driver = None


    def driver(self):
        # todo multi-driver case
        return self._driver


    def labels(self):
        """
        Get the labels string.

        Returns:
            string
        """
        out = get_labels(self.lbl, self.indim, self.with_t)
        return out


    def set_labels(self, labels):
        """
        Config method. (Called by user.)
        For using out-of-box problems with user labels,
        the labels set can be changed.
        Example: given an out-of-box problem with labels "x, y, z; u",
        the user may change the labels to (say) "u, v, w; Q".

        Arguments:

            labels (string):

        """
        # todo note that the user needs to modify the p.ranges before this method is called.
        #     should this step be included in this method? probably.
        lbl, indim, with_t = parse_labels(labels)
        self.fslabels = get_fslabels(lbl, indim, with_t)
        self.lbl = lbl
        self.indim = indim
        self.with_t = with_t
        self._review_labels()
        raise NotImplementedError


    ###################################################


    def init(self, background, out):
        """
        (Not called by user.)

        :meta private:
        """
        self.background = background
        # todo threadsafe?
        self.out = out
        # > init sources and type-check labels
        for lb in self.constraints:
            c = self.constraints[lb]
            c.init(lb)
            if c.source is not None:
                c.source.init(
                    dtype=background.dtype(),
                    parameters=self.p,
                )
            # todo type-check labels in constraints
        if self.ic_source is not None:
            self.ic_source.init(
                dtype=background.dtype(),
                parameters=self.p,
            )
        # > init solutions and sanity-check the target solutions and moments,
        # you cannot compute one thing in two ways.
        target_sols = []
        target_moments = []
        # todo inlabels variable is a misnomer? review
        for inlabels in self.solutions:
            sol = self.solutions[inlabels]
            sol.init(
                inlabels=inlabels,
            )
            # > check user:
            for lb in sol.methods:
                if sol.methods[lb] is not None:
                    target_sols.append(lb)
        # > init references
        reference_path = self.background.reference_absolute_root_directory
        for labels in self.references:
            ref = self.references[labels]
            try:
                ref.init(
                    labels=labels,
                    # todo awk? threadsafe?
                    log=self.out.log,
                    root_path=reference_path,
                    # hard-coded, for now
                    tolerance=1e-4,
                )
            except FileNotFoundError:
                print(f"[Problem] Could not obtain reference solution for labels '{labels}' using reference root path {reference_path}", file=stderr)
                print("[Problem] Exiting early.")
                exit(0)
        for labels in self.moments:
            moment = self.moments[labels]
            moment.init(
                labels=labels,
            )
            # > check user
            for lb in moment.methods:
                target_moments.append(lb)
        if len(target_sols) > len(set(target_sols)):
            raise ValueError(f"Multiple solving methods detected: target solutions {target_sols}.")
        if len(target_moments) > len(set(target_moments)):
            raise ValueError(f"Multiple solving methods detected: target moments {target_moments}.")



    def deinit(self, engine):
        for finalizer in self.finalizers:
            finalizer(engine)


    # todo deprecated
    def outdim(self):
        return len(self.lbl) - self.indim


    # todo deprecated
    def outlabels(self):
        return self.lbl[self.indim:]


    ##########################################################
    # User Utilities
    # For use in ic's, bc's, residuals.


    def get(
            self,
            labels,
            hub = None,
            X = None,
            requires_grad = True,
            cpu = False,
    ):
        """
        Returns a label derived from the
        input and output tensors via derivative operations.
        Examples:
        Compute the value of "u_x", the derivative of
        label v with respect to x.
        Compute the value of "v_t_t", the second derivative of
        label u with respect to t.
        These can be performed simultaneously, whether
        you are implementing methods for use in training models,
        or methods for producing outputs::

            # when computing pde residuals (during model training)
            u_x, v_tt = problem.get("u_x, v_t_t", hub=hub)
            # when computing ICs, references, or indirect outputs
            u_x, v_tt = problem.get("u_x, v_t_t", X=X)

        Exactly one of ``hub`` or ``X`` must be passed.
        The inputs ``hub`` and ``X`` each express the data
        needed for the computation, and are normally received directly
        from the inputs to the environment where problem.get is called.
        For higher derivatives, use the underscore like a derivative operator ∂/∂,
        thus u_x_y denotes ∂(∂u/∂x)/∂y. The notation u_xy will be interpreted as
        ∂u/∂(xy), not ∂^2u/(∂x∂y). Individual labels cannot contain underscores,
        so this notation is always well-defined.

        Arguments:

            labels (string):
                List of requested labels.
            hub (optional :any:`Hub`):
                Instance of Hub that provides the data contained
                in a single training batch.
            X (optional :any:`XFormat`):
                Formatted X array containing input data, with optional
                labels. If labels are not provided, :any:`Problem` will
                rely on its own labels to read X.
            requires_grad (optional boolean):
                Whether to deliver a tensor attached to the computational
                graph; the default behavior is context dependent.
                Not a very robust feature ITCINOOD.
                STIUYKB.
            cpu (boolean):
                Return a tensor located on the cpu device,
                if this is not set then the device of the X or hub,
                whichever is received, is used. Default: False

        Returns:

            tuple of arrays or tensors

        """
        # todo review requires_grad property of X in XFormat case (use X.X().requires_grad :: boolean)
        # > translate inputs into internal variables _x, _u, _t
        if X is not None:
            _x = X.X()
            _t = X.t()
            fslabels = X.fslabels()
            if fslabels is None:
                lbl, indim, with_t = self.lbl, self.indim, self.with_t
            else:
                # > no choice but to parse
                lbl, indim, with_t = parse_fslabels(fslabels)
            if _t is None and with_t:
                # todo deprecated (clear_algebraic_bug)
                # _t = _x[:,indim:indim+1]
                _u = _x[:,indim+1:]
            else:
                # t is a float, or a time independent problem.
                _u = _x[:,indim:]
        else:
            if not hasattr(hub, '_x'):
                raise ValueError(hub_error_message)
            lbl, indim, with_t = self.lbl, self.indim, self.with_t
            _x = hub._x if requires_grad else hub._x.clone().detach()
            _t = None
            _u = hub._u if requires_grad else hub._u.clone().detach()
        # > process _x, _u, _t with label list
        out = ()
        varlist = [x.strip() for x in labels.split(",")]
        if len(varlist) == 0:
            raise ValueError(f"Empty request to find label/gradient from input labels {labels}.")
        for v in varlist:
            # derivatives
            dlist = [x.strip() for x in v.split("_")]
            for x in dlist:
                if x == "":
                    raise ValueError(f"Invalid syntax in label {v}. Use a single underscore to indicate a partial derivative. For example, u_x is the derivative of u with respect to x and u_x_t is the partial derivative of u_x with respect to t.")
            yi, isinput = get_index(dlist[0], lbl, indim, with_t)
            if len(dlist) == 1:
                # ordinary label
                if isinput and yi == indim:
                    # if yi is indim, and the label is input with no derivative,
                    # then the index is time: the problem is time-dependent because
                    # otherwise the maximum index is indim-1.
                    # (Also if the input X is an XFormat, then its maximum index in _x must be
                    # less than or equal to the length of the problem labels.)
                    # Either _t is a float, or _t is a tensor from either hub or from XFormat.
                    _t = torch_full_like(_x[:,0:1], _t) if isinstance(_t, float) else _x[:,yi:yi+1]
                    out += (_t,)
                else:
                    if yi is None:
                        raise ValueError(f"Not found: label {v}.")
                    if isinput:
                        A = _x
                    else:
                        if _u is None:
                            raise ValueError(f"Not found: label {v} in labels {get_labels(lbl, indim, with_t)}")
                        A = _u
                    out += (A[:,yi:yi+1],)
            else:
                # higher label
                if isinput:
                    raise ValueError(f"Detected attempt to take partial derivative of an input label {dlist[0]} from requested label {v}.")
                if _u.shape[1] == 1 and yi == 0:
                    y = _u
                else:
                    # todo case of multiple output variables: experiment with
                    #  taking grad of entire u wrt entire x and then slice. it is like a
                    #  jacobian matrix.
                    if yi is None:
                        raise ValueError(f"Not found: label {dlist[0]} for requested label {v}.")
                    y = _u[:,yi:yi+1]
                for i in range(1, len(dlist)):
                    xi, isinput = get_index(dlist[i], lbl, indim, with_t)
                    if xi is None:
                        raise ValueError(f"Not found: label {dlist[i]} in requested label {v}.")
                    if not isinput:
                        raise ValueError(f"Detected attempt to take partial derivative with respect to an output label {dlist[i]} from requested label {v}.")
                    if i < len(dlist)-1:
                        higher = True
                    else:
                        higher = False
                    y = self.grad_(
                        _x=_x,
                        _y=y,
                        higher=higher,
                    )
                    y = y[:,xi:xi+1]
                out += (y,)
        # todo: work on the cpu if cpu is set
        if cpu:
            out2 = ()
            for i in range(len(out)):
                out2 += (out[i].cpu(),)
            out = out2
        # return a tensor/array if singleton
        if len(out) == 1:
            out = out[0]
        # self.out.after_problem_get(valuesin=_x, varlist=varlist, values=out)
        return out


    @timed("problem_get_moment")
    def get_moment(
            self,
            label,
            hub = None,
            X = None,
    ):
        """
        Extract a moment with respect to a
        batch during training, or in a method.

        Arguments:

            label (string):
                Label of a moment (only one can be requested),
                defined in the problem description.
            hub (optional :any:`Hub`):
                Hub instance, contains the batch.
                If hub is not passed then X must be passed, and vice versa.
            X (optional :any:`XFormat`):
                Passthrough data structure used
                during a method callback.

        Returns:

            array of shape [n, 1] where n is the
            batch size or sample set size, depending on whether `hub` or `X` is passed,
            which gives the requested moment's value at the
            simulation's current timestep.
            This moment is calculated either based on the sample set
            (in the case when ``X`` is used) or based on the value of the moment
            at each point in the Hub's (i.e., batch's) input point set
            (in the case when ``hub`` is used).

        """
        # > get source array information
        if X is not None:
            X0 = X.X()
            # solutions will arrive
            device_ = X0.device
            X0 = X0.cpu()
            t = X.t()
        else:
            device_ = self.driver().config.device
            # Case of a mixed t from a training batch
            X0 = hub._x.cpu().clone().detach()
            t = None
        # >> Assumption: past this point, we are cpu-computing.
        n = X0.shape[0]
        # > get target array information
        fslabels = self.driver().phase.momentsets.fslabels(label)
        lbl, indim, with_t = parse_fslabels(fslabels)
        # > get the input labels
        inlbl = lbl[:indim]
        if not with_t:
            # Note: the code after this point can be modified
            #  when demand requires, to support this case.
            raise NotImplementedError
        if X is not None:
            Xlabels = ','.join(inlbl+['t']) if t is None else ','.join(inlbl)
            tup = self.get(labels=','.join(Xlabels), X=X, requires_grad=False, cpu=True)
        else:
            tup = self.get(labels=','.join(inlbl+['t']), hub=hub, requires_grad=False, cpu=True)
        # This step is necessary so that scripts look nice (and because Python), cf. problem.get()
        if not isinstance(tup, tuple):
            tup = (tup,)
        # > the hub (or X) array with all irrelevant labels (columns) removed
        X_ = torch_hstack(tup)
        if t is None:
            # > paint sortable scalars on to X_, for convenience integers 0..n-1 inclusive
            integers = torch_linspace(0.0, n-1.0, steps=n, dtype=hub.fw_type).reshape((-1,1))
            X_ext = torch_hstack((X_, integers))
            # print(f"[get_moment] X_ext pre sort {X_ext}")
            tidx = indim # the time axis
            iidx = indim+1 # the integer axis, ENSURE: last idx
            # > sort, destructively, by increasing time values
            X_ext = sortdown(X=X_ext, k=tidx)
            # print(f"[get_moment] X_ext post sort {X_ext}")
            # print(f"[get_moment] tidx {tidx} iidx {iidx}")
            # > initialize output tensor
            out_ext = torch_hstack((torch_zeros([n, 1]), X_ext[:,iidx:iidx+1]))
            # print(f"[get_moment] out_ext {out_ext.shape}")
            # > write going piece by piece
            beg = 0
            end = 1
            t_piece = X_ext[beg, tidx]
            while end < n:
                t_test = X_ext[end, tidx]
                # comparison should actually be ok, but...
                if t_piece - 1e-12 < t_test < t_piece + 1e-12:
                    end += 1
                else:
                    # > write a piece
                    # print(f"[get_moment] write a piece beg {beg} end {end}")
                    out_piece = self.driver().phase.momentsets.lookup(
                        label,
                        t=t_piece,
                        X=X_ext[beg:end,:tidx],
                    )
                    # print(f"[get_moment] out_piece {out_piece.shape}")
                    out_ext[beg:end,0:1] = out_piece
                    # > set up next piece
                    beg = end
                    end = beg+1
                    t_piece = X_ext[beg, tidx]
            # > remainder piece
            # print(f"[get_moment] remainder piece beg {beg} end {end}")
            out_ext[beg:end,0:1] = self.driver().phase.momentsets.lookup(
                label,
                t=t_piece,
                X=X_ext[beg:end,:tidx],
            )
            # > unsort
            out_ext = sortdown(X=out_ext, k=1)
            out = out_ext[:,0:1]
        else:
            out = self.driver().phase.momentsets.lookup(
                label,
                t=t,
                X=X_,
            )
        out = out.to(device_)
        return out


    def get_moment_resolution(self, moment_label, label):
        """
        Get a resolution for a moment, via a convenient interface.

        Arguments:

            moment_label (string):
                A target moment label.
            label (string):
                A non-temporal input to the target moment.

        Returns:

            integer, resolution.

        """
        out = 0
        good = False
        for labels in self.moments:
            lbl, indim, with_t = parse_labels(labels)
            if moment_label in lbl[indim:]:
                moment = self.moments[labels]
                if label in moment.resolution:
                    out = moment.resolution[label]
                    good = True
            break
        if not good:
            errmsg = f"Cannot find resolution for variable {label} as input to moment {moment_label}."
            raise ValueError(errmsg)
        return out


    def set_moment_resolution(self, moment_label, label, value = None, multiple = None):
        """
        (Config method.)

        Change or set a resolution for an input to a moment.

        Arguments:

            moment_label (string):
                A target moment label.
            label (string):
                A non-temporal input to the target moment.
            value (optional integer):
            multiple (optional scalar):
                Use instead of value to increase/decrease by a
                percentage, defined as a multiple
                of the existing resolution. E.g., use
                1.5 for 50% increase in resolution, or
                use 0.5 for 50% decrease in resolution.

        """
        good = False
        for labels in self.moments:
            lbl, indim, with_t = parse_labels(labels)
            if moment_label in lbl[indim:]:
                moment = self.moments[labels]
                if label in moment.resolution:
                    if value is not None:
                        moment.resolution[label] = value
                        good = True
                    elif multiple is not None:
                        res = moment.resolution[label]
                        moment.resolution[label] = int(res*multiple)
                        good = True
                break
        if not good:
            errmsg = f"Cannot find resolution for variable {label} as input to moment {moment_label}."
            raise ValueError(errmsg)


    def get_constant(
            self,
            value,
            hub = None,
            X = None,
    ):
        """
        Get a constant value compatible with
        the input ``hub`` or ``X`` data.

        Arguments:

            value (scalar or list of scalar):
                Constant or list of constants to generate.
            hub (optional :any:`Hub`):
            X (optional :any:`XFormat`):

        Returns:

            array or tuple of arrays

        """
        if X is None and hub is None:
            raise ValueError(f"[Problem:get_constant] Missing argument. Needs either hub or X.")
        if X is None:
            device_ = self.driver().config.device
        else:
            device_ = X.X().device
        N = self.get_size(hub=hub, X=X)
        if not isinstance(value, list):
            out = torch_full([N, 1], fill_value=value).to(device_)
        else:
            out = ()
            for v in value:
                y = torch_full([N, 1], fill_value=v).to(device_)
                out += (y,)
        return out


    def get_size(self, hub=None, X=None):
        if X is None:
            if not hasattr(hub, '_x'):
                raise ValueError(hub_error_message)
            return hub._x.shape[0]
        else:
            return X.X().shape[0]



    # todo - deprecated

    # def put(self, *args):
    #     """
    #     Reassamble labels to form a multi-array.
    #     Idea: "put" is the opposite of "get",
    #     while "get" takes _x and creates x, y, z,
    #     "put" takes x, y, z and creates _x.
    #     Expected use in periodic residuals.
    #
    #     For now, simply combine the args into an array,
    #     nothing more complex is provided.
    #
    #     :meta private:
    #     """
    #     return torch_hstack(args)



    def format(
            self,
            X,
            resolution,
            right_open = False,
    ):
        """
        Reformat argument ``X`` to the problem labels,
        considering only inputs.
        Missing inputs will be filled in using a regular meshgrid
        with resolution given by the input `resolution`.

        The behavior of :any:`problem.regular`
        and :any:`problem.slice` are intentionally similar,
        while :any:`problem.format` works differently,
        providing only inputs, and the guarantee that its
        output has been formatted in a way that
        allows it to be evaluated on the model(s).

        .. note::

            Output labels cannot be generated this method,
            it generates values for inputs. The output
            from format can be evaluated using :any:`problem.evaluate`.

        Arguments:

            X (:any:`XFormat`):
                X is not modified.
            resolution (integer): 1-dimensional size of grid.
                E.g., if resolution is 100 and dimension is 2, grid is 100x100.
            right_open (boolean):
                Whether to generate inputs as right-open intervals,
                excluding the right (maximum) endpoints.
                Default: False

        Returns:

            Xout (:any:`XFormat`): reformatted X.

        """
        # todo remove my print messages soon

        # todo: review behavior when X is mixed in time.
        # do not clone yet
        X_ = X.X()
        fslabels = X.fslabels()
        tout = X.t()
        dtype_ = X_.dtype
        device_ = X_.device
        # todo review fslabels None option - should probably be deprecated but first cf. XFormat
        if fslabels is None:
            lbl, indim, with_t = self.lbl, self.indim, self.with_t
        else:
            lbl, indim, with_t = parse_fslabels(fslabels)
        if with_t and tout is None:
            # > get the t column
            raise NotImplementedError(f"Calling format() on a data set with mixed time values. This operation is not supported (yet).")
        # > find missing input columns
        missing = []
        if indim == 0:
            # The missing labels are all of them.
            for i, lb in enumerate(self.lbl[:self.indim]):
                missing.append((i, lb))
        else:
            # > find the missing labels
            for i, lb in enumerate(self.lbl[:self.indim]):
                found = False
                for lb_ in lbl[:indim]:
                    if lb == lb_:
                        found = True
                if not found:
                    missing.append((i, lb))
        if len(missing) == 0:
            # There are no missing labels.
            # > nothing to do
            Xout = X_.clone().detach()
        else:
            # Some or all labels are missing.
            # > build ranges list for missing
            ranges = []
            for _, lb in missing:
                ranges.append(self.p.ranges[lb])
            Xout = self.mesh(
                ranges = ranges,
                resolution = resolution,
                right_open = right_open,
                dtype = dtype_,
                device = device_,
            )
            # If all inputs were missing, we are done.
            if Xout.shape[1] != self.indim:
                # Case: missing is not [] and not all.
                # We have a regular grid, and we must now stamp copies of it for
                # each point we have received in the data set X.
                # > painful reformatting step, the cost of using meshgrid.
                # > divide the work into gaps - this cannot be the empty list
                indexlist = []
                for i, _ in missing:
                    indexlist.append(i)
                gaps = indexlist_to_gaps(indexlist)
                # print(f"[format] indexlist {indexlist} gaps {gaps}")
                slices = torch_zeros((0, self.indim)).to(device_)
                N = Xout.shape[0]
                M = len(gaps)
                P = self.indim
                Q = X_.shape[0]
                # print(f"[format] N {N} M {M} P {P} Q {Q}")
                # > create Q NxP slices one by one
                for i in range(Q):
                    # Xout column index
                    ii = 0
                    # X_ column index
                    jj = 0
                    # gaps index
                    j = 0
                    # builder for the ith slice
                    slice1 = torch_zeros((N, 0)).to(device_)
                    # > start building slice
                    gap0 = gaps[0][0]
                    if gap0 != 0:
                        beg = 0
                        end = gap0
                        # > constant gap
                        fill = torch_tile(input=X_[i:i+1,beg:end], dims=(N, 1))
                        # print(f"[format] fill {fill}")
                        jj += end-beg
                        slice1 = torch_hstack((slice1, fill))
                    while True:
                        beg = gaps[j][0]
                        end = gaps[j][1]
                        # > nonconstant gap
                        # print(f"[format] beg {beg} end {end}")
                        slice1 = torch_hstack((slice1, Xout[:,ii:ii+(end-beg)]))
                        # print(f"[format] slice1 {slice1}")
                        ii += end-beg
                        if end == P:
                            break
                        beg = end
                        end = P if j+1 == M else gaps[j+1][0]
                        # > constant gap
                        # print(f"[format] beg {beg} end {end}")
                        fill = torch_tile(input=X_[i:i+1,jj:jj+(end-beg)], dims=(N, 1))
                        jj += end-beg
                        slice1 = torch_hstack((slice1, fill))
                        # print(f"[format] slice1 {slice1}")
                        if end == P:
                            break
                        j += 1
                    # > add the slice to the stack of slices
                    # print(f"[format] stack slices {slices.shape} slice1 {slice1.shape}")
                    slices = torch_vstack((slices, slice1))
                # > replace Xout with slices,
                # which has all of the copies of Xout (throw away the original)
                Xout = slices
        out = XFormat(
            X=Xout,
            t=tout,
            # > update fslabels
            fslabels=get_fslabels(self.lbl[:self.indim], self.indim, self.with_t),
        )
        return out


    def slice(
            self,
            values,
            resolution,
            right_open = False,
            pre = None,
            evaluate = False,
            force_evaluate = False,
    ):
        """
        Create a "slice" at the specified
        "point" given by the ``values``
        argument, in a format already
        suitable for :any:`problem.get`.

        The default, simplest
        behavior is to expand around a single
        "point" using a regular meshgrid
        on remaining inputs,
        and evaluating every problem output
        variable on the resulting "slice".

        A slice's point must specify a point in time,
        ITCINOOD. (It must be a timeslice.)

        The behavior of :any:`problem.regular`
        and :any:`problem.slice` are similar.
        Cf. :any:`problem.format`.

        Arguments:

            values (dict of string to float):
                A dict of label-to-value assignments, specifying
                the constant dimensions of the slice.
            resolution (integer):
                The resolution of the slice. The slice consist of a regular mesh.
            right_open (boolean):
                Whether to create the regular mesh with a "right open" mesh,
                it will then be right open with respect to all mesh dimensions.
            pre (:any:`XFormat`):
                An existing object produced
                by the same call to slice(), such as
                appearing in a loop. If not None,
                the prepared slice will be modified
                instead of generating a new data structure.
                If prepared is None, resolution and right_open
                arguments will not be used.
            evaluate (boolean):
                Evaluate on inputs to populate the data with outputs.
                In situations where outputs are not needed,
                this step can be skipped. Default: False
            force_evaluate (boolean):
                Evaluate on inputs to populate the data with outputs,
                like ``evaluate``, but still further,
                force the evaluation to evaluate on the model,
                not interpolate based on IC's. Default: False

        Returns:

             :any:`XFormat`

        """
        if 't' not in values:
            # todo later
            raise NotImplementedError(f"Cannot create a slice without specifying a time value.")
        # > build fslabels and constructor list
        with_t = True
        values_ = []
        lbl = []
        t_ = values['t']
        for lb in values:
            if lb != 't':
                values_.append(float(values[lb]))
                lbl.append(lb)
        if pre is None:
            # > create XFormat
            X_ = torch_tensor(
                values_,
                dtype=self.driver().config.fw_type,
            ).reshape((1,-1)) #.to(self.driver().config.device)
            fslabels_ = get_fslabels(lbl, len(lbl), with_t)
            # > build XFormat
            # todo review: can we reserve space at this step?
            X = XFormat(
                X=X_,
                t=t_,
                fslabels=fslabels_,
                reserve=self.outdim(),
            )
            # > call problem.format
            # This will build the inputs of the slice.
            # print(f"[slice] X shape before {X.X().shape} fslabels {X.fslabels()}")
            Xout = self.format(
                X = X,
                resolution = resolution,
                right_open = right_open,
            )
            # print(f"[slice] X shape after {Xout.X().shape} fslabels {Xout.fslabels()}")
            # > evaluate
            if evaluate or force_evaluate:
                Xout = self.driver().evaluate(X=Xout, force_evaluate=force_evaluate)
        else:
            # > find indices for the requested labels
            idxs = []
            for lb1 in lbl:
                found = False
                for i, lb in enumerate(self.lbl[:self.indim]):
                    if lb1 == lb:
                        idxs.append(i)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Could not find label {lb1}")
            # > modify pre
            Xpre = pre.X()
            N = Xpre.shape[0]
            for i, idx in enumerate(idxs):
                value = values_[i]
                Xpre[:,idx:idx+1] = torch_full([N,1], value)
            # print(f"[slice] Xpre after reset {Xout.X()}")
            # print(f"[slice] Xpre labels after reset {Xout.fslabels()}")
            # > wipe output, it is invalid now
            pre.reset()
            if evaluate or force_evaluate:
                Xout = self.driver().evaluate(
                    X=pre,
                    force_evaluate=force_evaluate,
                )
            else:
                Xout = pre
            # print(f"[slice] Xpre after eval {Xout.X()}")
            # print(f"[slice] Xpre labels after eval {Xout.fslabels()}")
        return Xout



    def regular(
            self,
            X,
            resolution,
    ):
        """
        Emit a regular mesh in a format usable in problem.get().
        Using the output mesh, problem.get() will deliver
        values from a regular grid on the current base timeslice.

        The behavior of :any:`problem.regular`
        and :any:`problem.slice` are intentionally similar.
        Cf. :any:`problem.format`.

        Arguments:

            X (:any:`XFormat`):
                It is possible to pass ``X`` as it appears as an
                argument in :any:`Solution` methods.
            resolution (integer):
                per dimension resolution, or number of points to generate.
                Total size will be resolution^indim.

        Returns:

            :any:`XFormat`: a structure that can be passed to problem.get for labeled values.

        """
        # todo broken!!! in minor ways.

        # todo right_open argument

        raise ValueError("This method is out of date, to substitute for it use slice().")
        # > generate inputs by calling format on a blank XFormat
        # todo pick this up again when we are ready to dig back into XFormat and fw
        out = XFormat(
            X=torch_zeros((0,0), dtype=self.driver().config.fw_type).to(self.driver().config.device),
            labels='t;',
            t=X.t(),
        )
        out = self.format(
            X=out,
            resolution=resolution,
        )
        # > generate outputs
        out = self.driver().evaluate(
            X=out,
        )
        return out



    def regular_partial(
            self,
            fslabels,
            resolution,
            t,
            right_open = False,
            reserve = None,
    ):
        """
        Generate regular array of inputs just like :any:`Problem.regular`,
        but go no further, and drop the stronger constraint
        that the returned :any:`XFormat` can be used as-is
        inside of :any:`Problem.get`.

        Intended use case is to pass the output straight
        on to a :any:`Solution` method, cf. :any:`Result`.

        Arguments:

            fslabels (string):
                labels to use, only subsets of :any:`Problem` inputs
                are possible.
            resolution (integer):
                regular array size, down each dimension
            t (optional scalar):
                time
            right_open (boolean):
                Whether to generate right-open arrays.
            reserve (optional integer):
                If used, reserve this many additional unlabeled
                data in the output.

        Returns:

            :any:`XFormat`

        """
        lbl, indim, with_t = parse_fslabels(fslabels)
        # > we only use the input labels
        lbl = lbl[:indim]
        # > build ranges
        ranges = []
        for lb in lbl:
            rng = self.p.ranges[lb]
            ranges.append(rng)
        X = self.mesh(
            ranges=ranges,
            resolution=resolution,
            right_open=right_open,
            dtype = self.driver().config.fw_type,
            device = self.driver().config.device,
        ) if len(ranges) > 0 else \
            torch_zeros((1,0)) # todo awk
        out = XFormat(
            X = X,
            t = t,
            fslabels = get_fslabels(lbl, indim, with_t),
            reserve = reserve if reserve is not None else 0,
        )
        return out


    def evaluate(
            self,
            labels,
            X,
    ):
        """
        Evaluate the input ``X`` on the models
        to produce the specified label values.

        .. note::

            It is only possible to evaluate all
            the problem labels ITCINOOD,
            but this will be relaxed in a future update.

        Arguments:

            labels (string):
                comma-separated list of labels to add to the
                data structure ``X`` via evaluation.
            X (:any:`XFormat`):

        Returns:

            XU (:any:`XFormat`):
                X with appended values.

        """
        lbl, indim, with_t = parse_labels(labels)
        # todo review, cf. pinn::Sandbox/pdetest/pdetest1d/EHP1
        # > sanity check
        # if with_t or len(lbl) != indim:
        #     raise ValueError
        # if set(lbl) != set(self.lbl[self.indim:]):
        #     raise NotImplementedError(f"It is only possible to evaluate all the problem output labels at this time.")
        XU = self.driver().evaluate(X=X)
        return XU



    def log(self, msg):
        """
        Add a message to log.
        Works like Python print(),
        but output will be stored.

        Arguments:

            msg (string)

        :meta private:
        """
        self.out.log(msg)


    #################################################



    def clear_gradients(self):
        """
        Call when gradients are no longer needed,
        e.g. at conclusion of training with a given batch.
        Equivalent to grad.clear().
        """
        # todo call to grad should depend on the driver used.
        self.grad_.clear()



    def bounding_box(self):
        """
        Generate a bounding box based
        on the list of constraints (specifically the sources
        from each constraint, which may typ. be interior,
        or boundary constraints.)

        Remind: the idea of this method, right or wrong,
        is to avoid storing a lot of redundant information
        in memory. The information is kept in the sources.
        The method exposed here computes the
        derived information on the fly.

        Returns:

             :any:`BoundingBox`
        """
        out = BoundingBox(self.indim)
        for lb in self.constraints:
            c = self.constraints[lb]
            out += c.source.bounding_box()
        return out



    ##########################################################
    # Calculus



    def grad(self, ylabel, xlabels, hub):
        """
        Helper to return the gradient of a single variable ylabel
        with respect to a list of variables xlabels.
        Examples:
        grad = problem.grad("u", "x,y,z", hub)
        X = "x, y, z" # create a reusable packet of labels
        P_x, P_y, P_z = problem.grad("P", X, hub)
        gradT = problem.grad("T", X, hub)

        Arguments:

            ylabel:
                One input variable
            xlabels:
            hub:

        Returns:
            tuple of derived values
        """
        varlist = [x.strip() for x in xlabels]
        labellist = [ylabel + "_" + v for v in varlist]
        labels = ", ".join(x for x in labellist)
        return self.get(labels=labels, hub=hub)


    def partial(self, v, xlabel, hub, higher=False):
        """
        Find the partial of a value v, possibly obtained
        algebraically and/or via differentiation from problem inputs,
        with respect to a problem label xlabel.

        Arguments:

            v:
            xlabel:
            hub:
            higher: whether to keep the graph
                in case another derivative of the output vector will be requested.

        Returns:

            y:

        """
        xi, isinput = get_index(xlabel, self.lbl, self.indim, self.with_t)
        if xi is None:
            raise ValueError(f"Not found: requested label {xlabel}.")
        if not isinput:
            raise ValueError(f"Detected attempt to take partial derivative with respect to an output label {xlabel}.")
        y = self.grad_(
            _x=hub._x,
            _y=v,
            higher=higher,
        )
        y = y[:,xi:xi+1]
        return y


    def div(self, ylabel, xlabels, hub):
        """
        Helper to return the divergence of a single variable ylabel
        with respect to a list of variables xlabels.

        Arguments:

            ylabel:
            xlabels:
            hub:

        Returns:

            tuple of derived values

        """
        G = self.grad(ylabel=ylabel, xlabels=xlabels, hub=hub)
        for i in range(1, len(G)):
            G[0] += G[i]
        return G[0]


    def curl(self, ylabel, xlabels, hub):
        """
        Helper to return the divergence of a single variable ylabel
        with respect to a list of variables xlabels.

        Arguments:

            ylabel:
            xlabels:
            hub:

        Returns:

            tuple of derived values
        """
        G = self.grad(ylabel=ylabel, xlabels=xlabels, hub=hub)
        if len(G) == 2:
            return G[1] - G[0]
        else: # len(G) == 3:
            return G[2] - G[1], G[0] - G[2], G[1] - G[0]


    def materialD(self, ylabel, vel, spc, hub):
        """
        Material derivative Dy/Dt,
        a convenient construct in fluid mechanics and related fields.
        The number of labels in vel and spc must be the same;
        this is not checked.

        Arguments:

            ylabel:
                Target variable.
            vel:
                Labels for velocities, e.g., "vx, vy, vz"
            spc:
                Labels for space, e.g., "x, y, z"
            hub:
                access to input batch or sample set _x,
                output batch or sample set _u

        Returns:
            M: material derivative
        """
        u_t = self.get(ylabel+"_t", hub)
        V = self.get(vel, hub)
        G = self.grad(ylabel=ylabel, xlabels=spc, hub=hub)
        M = u_t
        for i in range(len(V)):
            M += V[i]*G[i]
        return M




    def mesh(
            self,
            ranges,
            resolution,
            right_open,
            dtype = None,
            device = None,
    ):
        """
        A very general mesh-builder.
        Construct a regular array of points
        using a standard meshgrid method,
        formatted as an :any:`XFormat` input.

         Arguments:

            ranges (nonempty list of pair of scalar):
                length is ``d``, the dimension of the mesh,
                defines the ranges as well as the order
                of the coordinates.
            resolution (integer or list of integers):
                resolution ``n``, the number of points
                in the regular mesh along each dimension.
            right_open (boolean):
                whether to create a right-open regular mesh.
            dtype (optional torch dtype): dtype
            device (optional torch device): device

        Returns:

            torch.tensor
                an array of shape ``(n**d, d)``.
        """
        # todo fw
        return mesh(
            ranges=ranges,
            resolution=resolution,
            right_open=right_open,
            dtype=dtype,
            device=device,
        )





    ##########################################################
    # Private methods and utilities





    def _review_labels(self):
        """
        (Not called by user.)
        """
        lbl, indim, with_t = self.lbl, self.indim, self.with_t
        for l in self.lbl:
            if '_' in l:
                raise ValueError(f"Label {l}: labels cannot contain underscore (_).")
            if '-' in l:
                raise ValueError(f"Label {l}: labels cannot contain hyphen (-).")
            if l[-3:] == 'ref':
                raise ValueError(f"Label {l}: labels cannot end in 'ref'.")
            if l[-3:] == 'val':
                raise ValueError(f"Label {l}: labels cannot end in 'val'.")
            if l[-3:] == 'err':
                raise ValueError(f"Label {l}: labels cannot end in 'err'.")
        for lb in lbl[:indim]:
            if lb not in self.p.ranges:
                raise ValueError(f"Input label {lb} range must specified: provide a lower and upper bound ({lb}min, {lb}max).")
        for lb in lbl[indim:]:
            if lb not in self.p.ranges:
                raise ValueError(f"Output label {lb} range not specified. If unknown, set the range to None.")
        # outdim = len(lbl)-indim
        # Nicc = len(self.ic_constraints)
        # if Nicc > 0:
        #     # the problem is time dependent
        #     if outdim != Nicc:
        #         raise ValueError(f"The number of ic constraints {Nicc} does not match the number {outdim} of problem outputs, but the problem is time dependent.")

    def _review_icc_labels(self):
        """
        During initialization, review user-chosen labels. Not bulletproof.
        """
        lbls = []
        for label in self.ic_constraints:
            if label is None or label == "":
                raise ValueError(f"Invalid ic label {label}. Choose a label (nonempty string).")
            if label == "thumb":
                raise ValueError(f"Invalid ic label {label}. Reserved label.")
            lbls.append(label)
        if len(lbls) != len(set(lbls)):
            raise ValueError(f"Problem-wide, ic labels must be unique.")



    def __str__(self):
        out = ""
        out += f"labels: {get_labels(*parse_fslabels(self.fslabels))}\n"
        out += f"Parameters: {str(self.p)}\n"
        out += f"IC Constraints (number: {len(self.ic_constraints)}):\n"
        out += "\n".join([str(x) for x in self.ic_constraints]) + "\n"
        # todo print sources - info about geometry?
        out += f"Constraints (number: {len(self.constraints)}):\n"
        out += "\n".join([str(x) for x in self.constraints]) + "\n"
        out += f"Solutions:\n"
        for labels in self.solutions:
            out += labels + "\n"
        out += f"References:\n"
        for labels in self.references:
            out += labels + "\n"
        out += str(self.th)
        return out

