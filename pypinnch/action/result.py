

from numpy import savetxt as numpy_savetxt
from math import sqrt


from . import Action
from .._impl.impl2 import Ledger, ResultPlotter

from .._impl.types import get_reference

from mv1fw import (
    get_labels,
    get_fslabels,
    parse_fslabels,
)



class Result(Action):
    """
    Result storage, validation, and plotting.

    For each stride, operations are performed
    after steps (including before the 0th step),
    and before and after each phase.
    The order of loops is {stride, phase, step}.
    So each phase executes its steps,
    for the given stride, before the
    next phase, until all phases have completed.
    The final phase's output, for its steps,
    consist of the PINN's "final" result.

    While processing, the class interacts only with the :any:`Solution`
    and :any:`Reference` classes's data fields, together with
    :any:`Problem` for basic inputs such as
    ranges of variables that are set in the :any:`Parameters` class.

    Many different things that a solver can generate as output
    conveniently fall under the general-purpose header "result".
    For the :any:`Result` class's utilities, these are classified as follows:

        - validation (comparison of solver output with a reference solution)
        - storage of data (raw data sets, either in some text or some binary format)
        - plots (for review/analysis/assessment/etc.)
        - animations (for further review, at a cost in processing time)

    If we label these A, B, C, and D, we can say the following:

        - If there is a reference, it is always used for validation.
            This is a fixed behavior that cannot be turned off.
            Validation, in this sense, includes a 1d-1d plot for
            time-dependent problems, of validation over time,
            broken up by stride and recorded at each step (and substep,
            if substeps are used).
        - If there are plots (C) then there must be data stored (B).
        - If there are movies (D) then there must be plots (C).

    Together, this implies that there are five possible "modes" for
    the :any:`Result` class::

        - 0000, a trivial action (we set this case aside, hereafter).
        - A000, validation only, or "pure validation".
        - AB00, validation and data.
        - ABC0, validation, data, plots, but no animations.
        - ABCD, validation, data, plots, animations.

    The default behavior is ABCD, the richest output possible.
    So to define other :any:`Result` action modes, you have the following
    controlling, restrictive flags:

        - `pure_validation`: set to `True` for mode A000 (default `False`).
        - `skip_plots`: set to `True` for mode AB00 (default `False`).
        - `skip_animations`: set to `True` for mode ABC0 (default `False`).

    If more than one of these flags are set,
    the stronger restriction will silently override the weaker one.

    .. note::

        If the Parameters class has a member ``alternate_ranges``,
        this should be a dict just like ``ranges``, and it can be
        used to set ranges that are plotted by :any:`Result` in addition
        to the "standard" ranges which are usually fed into the problem
        via the :any:`Constraint` s.
        The alternate_ranges feature is in early stages ITCINOOD.

    .. note::

        Too large vresolution can create artificial
        effects in the visualization.

    Parameters:

        final (boolean):
            Whether the result is final, or preliminary.
        vresolution (list of integer):
            List of resolutions to utilize when generating (series, heatmap, ...) plots.
            Implemented as a list for testing, but you will probably wish
            to find a value that works and rely on it. Each resolution integer
            will add performance cost and may generate a certain number of files.
            Default: [100]
        log_validation (boolean):
            Whether to log (to the log, via messages) the validation information
            for solutions that have references. Regardless of whether they are logged,
            they will be stored and plots will be generated. (Default: False)
            todo: review the 0d behavior
        pure_validation (boolean):
            Set for the action to only perform validation.
        skip_plots (boolean):
            Set for the action to generate no plots/animations.
        skip_animations (boolean):
            Set for the action to generate plots, but no animations.

    """
    # todo
    #  For now, do everything in each phase
    #  and let the user interpret as desired. In the future,
    #  treat the output in the case self.final = True differently?

    # todo I wonder if the impact to performance from separately processing
    #  the validation (depending on a *resolution*) and visualization
    #  (depending on a *vresolution*) is worth the degree of freedom
    #  that the solver would obtain. There is a case to be made that in fact
    #  that idea is incorrect. After all, you also want to *see* what you are
    #  basing your validation on! ...But the problems with that way of
    #  looking at it stack up too high.
    #  Do resolutions given by Solutions apply to validation? visualization? both?

    # todo binary format optional
    #  (cf. numpy_savetxt)


    def __init__(
            self,
            final = False,
            vresolution = None,
            log_validation = False,
            pure_validation = False,
            skip_plots = False,
            skip_animations = False,
    ):
        super().__init__()
        self.final = final
        # stem for placing error plots
        self.error_stem = "error"
        # resolution ("pixellation") of result plots (2d/3d cases only)
        self.vresolution = [100] if vresolution is None else vresolution
        # ensure self.vresolution[-1] is the largest resolution
        if len(self.vresolution) > 1:
            self.vresolution.sort()
        self.log_validation = log_validation
        if pure_validation:
            # > pure validation mode
            self.store_data = False
            self.plotter = None
        elif skip_plots:
            # > no-plots mode
            self.store_data = True
            self.plotter = None
        else:
            # > rich output mode, optionally skip the animations
            self.store_data = True
            self.plotter = ResultPlotter(
                result = self,
                skip_animations=skip_animations
            )
        self.ledgers = {}
        self.validation_ledgers = {}
        # counters for every
        self.counter = {}
        # mark for existing artifacts from a phase
        self.val_artifacts = {}
        self.data_artifacts = {}


    def gate_strideloop(self, B):
        if B.problem.with_t:
            # > set counters
            # should be done here and never again
            # so that results don't show "seams" at strides
            # in general.
            for inlabels in B.problem.solutions:
                sol = B.problem.solutions[inlabels]
                self.counter[sol.fslabels] = 0
        else:
            pass


    def on_phase(self, B):
        # > create ledgers
        nstep = B.phase.th.Nstep()
        for inlabels in B.problem.solutions:
            sol = B.problem.solutions[inlabels]
            fslabels = sol.fslabels
            lbl, indim, with_t = parse_fslabels(fslabels)
            self.val_artifacts[fslabels] = False
            self.data_artifacts[fslabels] = False
            outdim = len(lbl) - indim
            ref = get_reference(fslabels=fslabels, references=B.problem.references)
            if ref is not None:
                # > create validation ledger, to record scalar validation for each output
                vallbl = [x+"val" for x in lbl[indim:]]
                if with_t:
                    # t, uval, vval, wval
                    vallbl = ['t'] + vallbl
                    with_t_size = 1
                else:
                    with_t_size = 0
                labels_ = ', '.join(vallbl)
                # Reminder: There is a separate validation ledger for each phase.
                # todo document why self.vresolution
                self.validation_ledgers[fslabels] = Ledger(
                    # todo nstep+1 because I get an off the edge error, I don't know why, review
                    nstep=(nstep+1)*sol.substep*len(self.vresolution),
                    every=sol.every,
                    size=with_t_size+outdim,
                    # todo deprecate?
                    labels=labels_,
                )
            if self.store_data and indim == 0:
                # indim==0 implies time dependent.
                # > create ledger for time series
                # t, u, v, w
                size_ = 1+outdim
                if ref is not None:
                    lbl += [x+"ref" for x in lbl]
                    size_ += outdim
                self.ledgers[get_fslabels(lbl, indim, with_t)] = Ledger(
                    nstep=(nstep+1)*sol.substep*len(self.vresolution),
                    every=sol.every,
                    size=size_,
                    # todo deprecate?
                    labels=get_labels(lbl, indim, with_t),
                )


    def after_slice(self, B):
        """
        Process result and emit data + figures
        for each model.

        Arguments:

            B (:any:`Bundle`):

        """
        if B.sj == 0:
            if B.problem.with_t:
                # > process the 0th timeslice, when t = tinit
                for inlabels in B.problem.solutions:
                    sol = B.problem.solutions[inlabels]
                    q = None
                    t0 = B.phase.samplesets.icbase.t
                    n = 1
                    self.regular_grid_processing(B=B, sol=sol, q=q, t=t0, n=n)
            else:
                # > do nothing on 0th slice of time independent problem
                pass
        else:
            for inlabels in B.problem.solutions:
                sol = B.problem.solutions[inlabels]
                fslabels = sol.fslabels
                if not B.problem.with_t:
                    q = None
                    t0 = None
                    n = 1
                    self.regular_grid_processing(B=B, sol=sol, q=q, t=t0, n=n)
                else:
                    # check the solution's user-defined period of emission `every`
                    counter = self.counter[fslabels]
                    if counter == 0:
                        # todo icbase is XNoF
                        t = B.phase.samplesets.icbase.t
                        if sol.substep > 1:
                            stepsize = B.phase.th.stepsize()
                            q = stepsize/sol.substep
                            t0 = t - stepsize + q
                            n = sol.substep
                        else:
                            q = None
                            t0 = t
                            n = 1
                        self.regular_grid_processing(B=B, sol=sol, q=q, t=t0, n=n)
                    counter += 1
                    self.counter[fslabels] = 0 if counter == sol.every else counter


    def regular_grid_processing(self, B, sol, q, t, n):
        """
        (Called by :any:`Result`)

        Generate and process a regular grid of raw output.

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        ref = get_reference(fslabels=fslabels, references=B.problem.references)
        with_sk = (sol.substep > 1)
        if sol.resolution is not None:
            # > resolution for a solution user may wish to compute
            # faster by reducing resolution
            vresolution = sol.resolution
        else:
            # > default/general visualization resolutions,
            # typ. feasible for direct outputs
            vresolution = self.vresolution
        # > loop through visualization resolutions
        for vres in vresolution:
            # > create regular grid on input domain
            reserve = len(lbl[indim:])
            reserve = 2*reserve if ref else reserve
            Xreg = B.problem.regular_partial(
                fslabels=fslabels,
                resolution=vres,
                t=t,
                right_open=False,
                reserve = reserve,
            )
            for ti_ in range(n):
                # > evaluate model, or use ic's if the model is not ready yet
                sol.evaluate(
                    X=Xreg,
                    problem=B.problem,
                    driver=B.driver,
                )
                if ref:
                    ref.evaluate(
                        X=Xreg,
                        problem=B.problem,
                    )
                # > move to export subroutines
                Xreg.enter_pitstop()
                # > substep counter
                sk = ti_ if with_sk else None
                if ref is not None:
                    # > perform validation (whenever possible i.e. whenever there is a reference solution)
                    # todo see comment in next branch
                    self.after_subslice_validate(B=B, sol=sol, Xreg=Xreg, sk=sk)
                    self.val_artifacts[fslabels] = True
                if self.store_data:
                    self.after_subslice_store(B=B, X=Xreg, sk=sk)
                    self.data_artifacts[fslabels] = True
                if self.plotter:
                    # Note related to XFormat, version 0.3.1:
                    # After this point in the pitstop, we assume that Xreg
                    # has the form [x1, x2, ..., xn, t, u1, u2, ..., um, u1ref, u2ref, ..., umref]
                    # if there are references (if ref is not None), otherwise
                    # [x1, x2, ..., xn, t, u1, u2, ..., um].
                    # This hard binary assumption was made before XFormat was introduced,
                    # and I believe removing this "hard" assumption and
                    # doing processing "out of" XFormat instead of inside of Result
                    # using hard assumptions is preferable. For one thing,
                    # the Result code is too large and much of it is redundant.
                    # ...This would involve making the Result processing "smart" (in a very humble sense)
                    # in the sense that the Result should be able to "understand" that
                    #  - this data is 2d, or 3d, or...,
                    #  - this output is reference for that output (due to a syntactic rule X <--> Xref),
                    #  - ...
                    # and process it accordingly. In this way, the verbose code can be broken down into manageable,
                    # testable parts - the plotting routines become directly related to XFormat, and no longer
                    # associated with a larger machinery like Solution/Reference/Result.
                    # ...but this also suggests that XFormat has *some* high-level information about the
                    # nature of the plotted data. (This can be defined in Solution.)
                    #
                    # At this time, I am unable to make this change due to higher priority updates pending.
                    # todo update - it is precisely for this sort of collision/mess that XFormat was created... :/ I will fix this soon.
                    #  to start on a fix, modify _after_subslice_validate to accept an XFormat instead of breaking it apart.
                    self.plotter.after_subslice(
                        # todo update for time independent case (branch on B.problem.with_t)
                        B=B,
                        sol=sol,
                        Xreg=Xreg,
                        sk=sk,
                        # todo review
                        ti_=ti_,
                        vres=vres,
                    )
                # > reset and advance
                Xreg.exit_pitstop()
                if n > 1:
                    Xreg.advance(deltat=q)


    def after_phase(self, B):
        """
        Post-processing at the end of the phase,
        e.g., making animations from generated frames.

        Arguments:
            B (:any:`Bundle`):

        """
        for inlabels in B.problem.solutions:
            sol = B.problem.solutions[inlabels]
            fslabels = sol.fslabels
            _, indim, _ = parse_fslabels(fslabels)
            if self.val_artifacts[fslabels]:
                self.after_phase_validate(B, sol)
            if self.store_data and indim == 0:
                self.after_phase_store(B, sol)
            if self.plotter and self.data_artifacts[fslabels]:
                self.plotter.after_phase(B, sol)


    ######################################################


    def after_phase_validate(self, B, sol):
        """
        Store the validation ledger, and make a plot.

        Arguments:
            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        if fslabels in self.validation_ledgers:
            ledger = self.validation_ledgers[fslabels]
            lbl, indim, with_t = parse_fslabels(fslabels)
            lblval = [x+"val" for x in lbl[indim:]]
            fslabelsval = get_fslabels(lblval, 0, with_t)
            X = ledger.retrieve()
            filename = self.cog.filename(
                action=self,
                handle=fslabelsval,
                phasei=B.phasei,
                stem="dat",
                ti=B.ti,
            )
            numpy_savetxt(
                fname = filename,
                X = X,
            )
            if with_t:
                # > make plot
                filename = self.cog.filename(
                    action=self,
                    handle=fslabelsval,
                    driveri=B.driveri,
                    phasei=B.phasei,
                    stem=self.error_stem,
                    ti=B.ti,
                    ending="png",
                )
                title = self.cog.title(
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=B.ti,
                    level=B.L,
                    tag = ', '.join(lblval) + "(t)",
                )
                self.fig.series(
                    filename = filename,
                    X = X,
                    inlabel = 't',
                    inidx = 0,
                    outlabels = lblval,
                    outidxs = range(1,X.shape[1]),
                    title = title,
                    text = None,
                    xlim = B.phase.th.range(),
                    ylim = None,
                    t = None,
                )



    def after_phase_store(self, B, sol):
        """
        Store the ledger, plot the ledger to a "multiseries" time series plot.

        Arguments:
            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        ref = get_reference(fslabels=fslabels, references=B.problem.references)
        if ref:
            lbl += [x+'ref' for x in lbl[indim:]]
            fslabels = get_fslabels(lbl, indim, with_t)
        filename = self.cog.filename(
            action=self,
            handle=fslabels,
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
        )
        ledger = self.ledgers[fslabels]
        X = ledger.retrieve()
        header = 't, ' if with_t else ''
        header += ', '.join(lbl)
        numpy_savetxt(
            fname = filename,
            X = X,
            header = header
        )


    ####################################################
    # Subslice Methods


    def after_subslice_store(
            self,
            B,
            X,
            sk,
    ):
        """
        Store the result for the solution with given labels.

        We store the reference values
        even if these are already stored elsewhere.
        We do this because:
        - it is easy to implement.
        - it is convenient for post-processing.

        Arguments:

            B (:any:`Bundle`):
            X (:any:`XFormat`):
                It must be pitstopped by caller.
            sk (optional integer):
                The substep integer, if substepping is used,
                otherwise None must be passed.

        """
        # todo remind, we assume that after XFormat update,
        #  the X enters with the "fref, gref" etc labels and columns present
        #  and we no longer deal with the incredible hassle that caused.
        X0 = X.X()
        fslabels = X.fslabels()
        lbl, indim, with_t = parse_fslabels(fslabels)
        outdim = len(lbl[indim:])
        if with_t:
            t = X.t()
            if indim == 0:
                item = [t]
                # > write to ledger like: [t, u1, ..., un, u1ref, ..., unref]
                for i in range(outdim):
                    # evaluate at 0th row for 0d case.
                    itemi = X0[0,i]
                    item.append(itemi)
                self.ledgers[fslabels].add(item=item)
            if indim > 0:
                filename = self.cog.filename(
                    action=self,
                    handle=fslabels,
                    driveri=B.driveri,
                    phasei=B.phasei,
                    stem="dat",
                    ti=B.ti,
                    sj=B.sj,
                    sk=sk,
                )
                header = ', '.join(lbl) + f", t = {t:.8f}"
                numpy_savetxt(fname=filename, X=X0, header=header)
        else:
            filename = self.cog.filename(
                action=self,
                handle=fslabels,
                driveri=B.driveri,
                phasei=B.phasei,
                stem="dat",
            )
            header = ', '.join(lbl)
            numpy_savetxt(fname=filename, X=X0, header=header)


    def after_subslice_validate(
            self,
            B,
            sol,
            Xreg,
            sk,
    ):
        """
        Validate solution by comparing to reference, if any.
        Store in ledger, write messages to log.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):
            Xreg (:any:`XFormat`):
            sk (optional integer):
                The substep integer, if substepping is used,
                otherwise None must be passed.

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        outdim = len(lbl) - indim
        methods = sol.methods
        t = Xreg.t()
        XQQref = Xreg.X()
        # > push an item to validation ledger
        item = [t] if with_t else []
        if indim == 0:
            for i in range(outdim):
                lb = lbl[i]
                method = methods[lb]
                # > gate a skipped solution
                # callables are truthy, "or" is for readability.
                if method or callable(method):
                    v_pred = XQQref[0, i]
                    v_true = XQQref[0, len(lbl)+i]
                    mse = v_pred - v_true
                    # 0d: sqrt(mean((v_pred - v_true)**2)) = abs(v_pred - v_true)
                    sqrt_mse = -mse if mse < 0 else mse
                    item.append(sqrt_mse)
                else:
                    item.append(-1.0)
        else:
            for i in range(outdim):
                lb = lbl[indim+i]
                method = methods[lb]
                # > gate a skipped solution
                # callables are truthy, "or" is for readability.
                if method or callable(method):
                    v_pred = XQQref[:,indim+i]
                    v_true = XQQref[:,len(lbl)+i]
                    mse = ((v_pred-v_true)**2).mean()
                    # todo: can the computation be re-used?
                    # todo: variance of error?
                    # todo other kinds of analysis besides mse and variance?
                    sqrt_mse = sqrt(float(mse))
                    item.append(sqrt_mse)
                    if self.log_validation:
                        self._log_validation(B, sk=sk, lb=lb, inlb=', '.join(lbl[:indim]), t=t, sqrt_mse=sqrt_mse)
                else:
                    item.append(-1.0)
        self.validation_ledgers[fslabels].add(item=item)



    def _log_validation(self, B, sk, lb, inlb, t, sqrt_mse):
            msg = ""
            if t is None:
                msg += f"[Validation] {lb}({inlb}) √mse {sqrt_mse}"
            else:
                msg += f"[Validation] ti {B.ti} sj {B.sj} sk {sk} t = {t:.2f}\n"
                msg += f"[Validation] {lb}({inlb}, t=t) √mse {sqrt_mse}"
            self.log(msg)

