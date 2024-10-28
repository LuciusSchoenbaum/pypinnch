


# todo fw
import numpy as np
# todo optional binary format
from numpy import savetxt as numpy_savetxt


from .. import Action
from ..._impl.impl2 import Ledger
from ..._impl.types import get_time_from_header


from mv1fw import (
    get_labels,
    get_fslabels,
    parse_labels,
    parse_fslabels,
    sortdown,
)
from mv1fw.visutil import Animation






class BaseMonitor(Action):
    """

    :any:`Monitor` that creates output similar to
    the Result class, but utilizes the solver's Base
    the sample set for IC training.
    These artifacts allow the user to review the
    solver's behavior and assess questions
    like whether the Base is sufficiently
    populated with sample points, and similar issues.
    Because a randomly distributed
    sample set (like IC base) is inconvenient for generating
    plots, the Result class uses regular grids
    that it generates for itself.
    So the Result class does not provide this
    kind of diagnostic information.

    Parameters:

        vresolution (list of integer):
            List of resolutions to utilize when generating heatmap plots. Default: ``[100]``

            .. note::

                Note: too large resolution can create artificial
                effects in the visualization.
                Implemented as a list for testing, but you will probably wish
                to find a value that works and rely on it, as each resolution
                will add performance cost and may generate a number of files.

        store_data (boolean):
            If set, the data used to generate plots will be stored.
            If you store data via Result class, this data is somewhat redundant,
            so you should only set store_data if you have a particular reason for doing so.

    """

    # todo this is in stasis and not finished ------ i believe it is nearly ready.


    def __init__(
            self,
            vresolution = None,
            store_data = False,
    ):
        super().__init__()
        self.ledgers = {}
        # frame duration for animation artifacts
        self.frame_duration = 1000
        # resolution ("pixellation") of result plots (2d/3d cases only)
        self.vresolution = [100] if vresolution is None else vresolution
        self.store_data = store_data




    def on_phase(self, B):
        # > create ledgers
        nstep = B.phase.th.Nstep()
        for labels in B.problem.solutions:
            sol = B.problem.solutions[labels]
            lbl, indim, with_t = parse_labels(labels)
            if with_t:
                if indim == 0:
                    # > create ledger for time series
                    outdim = len(lbl)
                    # ensure consistent labels formatting
                    labels_ = get_labels(lbl, indim, with_t)
                    size_ = 1+outdim
                    # > search for a reference time series
                    ref = None
                    for rlabels in B.problem.references:
                        ref_ = B.problem.references[rlabels]
                        if ref_.fslabels == sol.fslabels:
                            ref = ref_
                            break
                    if ref is not None:
                        labels_ += ", " + ", ".join([x+"ref" for x in lbl])
                        size_ += outdim
                    self.ledgers[sol.fslabels] = Ledger(
                        nstep=nstep,
                        # BaseMonitor generates artifacts after every timestep
                        every=1,
                        size=size_,
                        labels=labels_,
                    )
            else:
                raise NotImplementedError



    def after_slice(self, B):
        """
        Process result and emit data + figures
        for each model.
        :param B: action bundle
        """
        for labels in B.problem.solutions:
            sol = B.problem.solutions[labels]
            ref = None
            for rlabels in B.problem.references:
                ref_ = B.problem.references[rlabels]
                if ref_.fslabels == sol.fslabels:
                    ref = ref_
                    break
            lbl, indim, with_t = parse_labels(labels)
            if with_t:
                X = B.phase.samplesets.base.X
                t = B.phase.samplesets.base.t
                X_, QQref = ref.evaluate(X=(X,t), problem=B.problem) \
                    if ref is not None else (None, None)
                # todo deprecated, done in ref.evaluate()
                # QQref = torch.from_numpy(QQref) if QQref is not None else None
                # X_ = torch.from_numpy(X_) if X_ is not None else None
                if X_ is None:
                    X_ = X
                else:
                    X_ = B.problem.format(
                        X=X_,
                        fslabels=ref.fslabels,
                        resolution=ref.resolution,
                    )
                    X_ = B.driver.format(
                        X_,
                        fslabels=sol.fslabels,
                    )
                XQQref = sol.evaluate(X=(X_, t), problem=B.problem, QQref=QQref)
                with_ref = (QQref is not None)
                if self.store_data:
                    self._after_slice_store(B, sol, XQQref, with_ref)
                # todo no validation in BaseMonitor - not the point here.......
                # self._after_slice_validate(B, sol, XQQref, with_ref)
                self._after_slice_plot(B, sol, XQQref, with_ref)
            else: # not with_t
                raise NotImplementedError






    def after_phase(self, B):
        """
        Post-processing at the end of the phase,
        e.g., making animations from generated frames.

        :param B: action bundle
        """
        for labels in B.problem.solutions:
            sol = B.problem.solutions[labels]
            lbl, indim, with_t = parse_labels(labels)
            if with_t:
                if indim == 0:
                    self._after_phase_0d(B, sol)
                elif indim == 1:
                    self._after_phase_1d(B, sol)
                elif indim == 2:
                    self._after_phase_2d(B, sol)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError




    ####################################################
    # Private Methods



    def _after_slice_store(
            self,
            B,
            sol,
            XQQref,
            with_ref,
    ):
        """
        Store the result for the solution with given labels.

        todo review
        We store the QQref and the Xref
        even if these are already stored elsewhere.
        We do this because:
        - it is easy to implement.
        - it is convenient for post-processing.

        :param B:
        :param sol:
        :param XQQref:
        :param with_ref: boolean
        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        if with_t:
            if indim > 0:
                filename = self.cog.filename(
                    action=self,
                    handle=fslabels,
                    driveri=B.driveri,
                    phasei=B.phasei,
                    stem="dat",
                    ti=B.ti,
                    sj=B.sj,
                )
                t = B.phase.samplesets.base.t
                header = ", ".join(lbl)
                if with_ref:
                    header += ", " + ", ".join([x+"ref" for x in lbl[indim:]])
                header = f"{header}, t = {t:.8f}"
                numpy_savetxt(fname=filename, X=XQQref.numpy(), header=header)
        else:
            # time independent case
            # ...after_slice is called only once?
            raise NotImplementedError



    def _after_slice_plot(
            self,
            B,
            sol,
            XQQref,
            with_ref,
    ):
        """
        Plotting routines to produce generic plots.

        :param B:
        :param sol:
        :param XQQref:
        :param with_ref: boolean
        """
        _, indim, with_t = parse_fslabels(sol.fslabels)
        if with_t:
            if indim == 0:
                self._after_slice_plot_0d(
                    B,
                    sol,
                    XQQref,
                    with_ref,
                )
            elif indim == 1:
                self._after_slice_plot_1d(
                    B,
                    sol,
                    XQQref,
                    with_ref,
                )
            elif indim == 2:
                # todo review
                XQQref = XQQref.detach().cpu().numpy()
                self._after_slice_plot_2d(
                    B,
                    sol,
                    XQQref,
                    with_ref,
                )
            elif indim == 3:
                raise NotImplementedError
            else: # > 3
                raise NotImplementedError
        else:
            # time independent
            raise NotImplementedError


    def _after_slice_plot_0d(
            self,
            B,
            sol,
            XQQref,
            with_ref,
    ):
        """
        Store time series update in a ledger.

        :param B:
        :param sol:
        :param XQQref:
        :param with_ref: boolean
        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        dim = len(lbl)
        t = B.phase.samplesets.base.t
        # write to ledger
        ledger = self.ledgers[fslabels]
        item = [t]
        # > write to ledger like: [t, u1, ..., un, u1ref, ..., unref]
        for i in range(dim):
            # evaluate at 0th row for 0d case.
            itemi = XQQref[0,i]
            item.append(itemi)
        if with_ref:
            for i in range(dim):
                itemi = XQQref[0,dim+i]
                item.append(itemi)
        ledger.add(item=item)


    def _after_slice_plot_1d(
            self,
            B,
            sol,
            XQQref,
            with_ref,
    ):
        """
        Generate a line plot for the 1d outputs and
        references, if any, for the current timeslice.

        :param B:
        :param sol:
        :param XQQref:
        :param with_ref: boolean
        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        t = B.phase.samplesets.base.t
        XQQref = sortdown(XQQref, 0)
        if not with_ref:
            outidxs = range(1, len(lbl))
        else:
            outidxs = range(1, 2*len(lbl) - 1)
        filename = self.cog.filename(
            action=self,
            handle=fslabels,
            stem="fig",
            ending="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            sj=B.sj,
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            sj=B.sj,
        )
        self.fig.series(
            filename = filename,
            X = XQQref,
            inlabel = lbl[0],
            inidx = 0,
            outlabels = lbl[1:] + [x+"ref" for x in lbl[1:]],
            outidxs = outidxs,
            title = title,
            text = None,
            t = t,
            xlim = B.problem.p.range(lbl[0]),
            ylim = B.problem.p.range(lbl[1]),
            half_linestyle = "dashed" if with_ref else None,
            marker = 'x',
            half_marker = '.',
        )


    def _after_slice_plot_2d(
            self,
            B,
            sol,
            XQQref,
            with_ref,
    ):
        """
        Generate heatmap visualizations of input vs. output at time t.

        Note on interpolation (linear, cubic, ...) when generating heatmaps:
        On an irregular dataset (like problem.base.X) there will be
        irregular curvature around the outline of Voronoi cells that risks
        confusing users who may mistake it for solver behavior.
        So we only use nearest interpolation on this point set (these
        heatmaps are tagged "ss" for sample set). This (nearest-interpolation)
        output can be inspected for information, in case the sample set has
        some useful, immediately apparent features. The reader who wishes
        to investigate further might like to uncomment the list of sizes [*]
        and inspect the plots that are produced.

        In order to produce deliverable outputs, we must first interpolate
        onto a regular grid, then pass this to the plotting routine,
        which interpolates a second time. This may be necessary due to
        the backend interpolation algorithm working better on a
        regular input, or at worst inputs with regularly-shaped Voronoi
        cells (this may be the case), but we just aren't sure.

        :param B: bundle
        :param sol:
        :param XQQref:
        :param with_ref: boolean
        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        t = B.phase.samplesets.base.t
        outdim = len(lbl) - indim
        xrange = B.problem.p.range(lbl[0])
        yrange = B.problem.p.range(lbl[1])
        xmin = xrange[0]
        xmax = xrange[1]
        ymin = yrange[0]
        ymax = yrange[1]
        # todo pass the ref through silly! excise boolean with_ref.
        ref = None
        for rlabels in B.problem.references:
            ref_ = B.problem.references[rlabels]
            if ref_.fslabels == fslabels:
                ref = ref_
                break
        for i in range(indim, XQQref.shape[1]):
            def is_ref():
                return i >= indim+outdim
            reftag = "ref" if is_ref() else ""
            outlb = lbl[i-outdim] if is_ref() else lbl[i]
            filename = self.cog.filename(
                action=self,
                handle=get_fslabels(lbl[:indim]+[outlb], indim, with_t),
                phasei=B.phasei,
                stem="fig",
                ending="png",
                ti=B.ti,
                sj=B.sj,
                tag= reftag if is_ref() else None
            )
            title = self.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                sj=B.sj,
                tag = f"{outlb}({lbl[0]}, {lbl[1]}) t {t:.4f} " + reftag,
            )
            value_range = B.problem.p.range(outlb)
            self.fig.heatmap(
                filename = filename,
                X = XQQref,
                in1 = 0,
                in2 = 1,
                out1 = i,
                lbl = lbl + [x + "ref" for x in lbl[indim:]],
                title = title,
                value_range = value_range,
                color_label = outlb,
                # plot xray only once
                plot_xray = True if i == 2 and B.ti == 0 and B.sj == 0 else False,
                method = "nearest",
                xlim=(xmin, xmax),
                ylim=(ymin, ymax),
            )
            # sic - uncomment list to inspect cubic interpolated plots [*]
            for sz in []: # [50, 100, 200, 300, 400]:
                filenamec = self.cog.tag_filename(filename=filename, insert=f"c{sz}")
                self.fig.heatmap(
                    filename = filenamec,
                    X = XQQref,
                    in1 = 0,
                    in2 = 1,
                    out1 = i,
                    lbl = lbl,
                    title = title,
                    value_range = value_range,
                    color_label = lbl[i],
                    plot_xray = False,
                    method = "cubic",
                    xlim=(xmin, xmax),
                    ylim=(ymin, ymax),
                    size_interpolate_2D=sz,
                )




    def after_slice_plot_3d(self, XQQref):
        """

        :param XQQref:
        :return:
        """
        raise NotImplementedError








    ########################################################
    # After phase post-processing




    def _slice_loadtxt(self, B, sj, handle):
        """
        Dimension-independent code for loading the base sample set.
        Used to make collective artifacts (as time varies).

        """
        filename = self.cog.filename(
            action=self,
            handle=handle,
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
            sj=sj,
        )
        t = get_time_from_header(filename)
        return np.loadtxt(filename), t



    def _make_a_movie(self, B, fslabels):
        """
        Create an animation from a set of generated plots.

        :param B:
        :param fslabels:
        """
        lbl, indim, with_t = parse_fslabels(fslabels)
        outdim = len(lbl) - indim
        anim = Animation()
        if indim == 1:
            # 1d case:
            tags = [None]
            fc_limit = 55
        elif indim == 2:
            # 2d case
            tags0 = ["sol", "ref", "err"]
            tags = []
            for vres in self.vresolution:
                tags += [f"v{vres}." + x for x in tags0]
            fc_limit = 30
        else:
            raise NotImplementedError
        for i in range(indim, indim+outdim):
            for tag in tags:
                frame_glob = self.cog.filename(
                    action=self,
                    handle=fslabels,
                    phasei=B.phasei,
                    stem="fig",
                    ending="png",
                    ti=B.ti,
                    sj="*",
                    tag=tag,
                )
                filename = self.cog.filename(
                    action=self,
                    handle=fslabels,
                    stem="fig",
                    ending="gif",
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=B.ti,
                    tag=tag,
                )
                if anim.from_frame_glob(
                        frame_glob=frame_glob,
                        filename=filename,
                        duration=self.frame_duration,
                        frame_count_limit=fc_limit,
                ):
                    save = True
                    # save = False
                    B.out.log("Movie was created from files:", save = save)
                    B.out.log(frame_glob, save = save)
                else:
                    B.out.log("[Output] Movie could not be created from files.")
                    B.out.log(frame_glob)



    def _after_phase_0d(self, B, sol):
        """
        Store the ledger, plot the ledger to a "multiseries" time series plot,
        perform validation if there is a reference.

        # todo store and validate stages: merge with other routines above?

        :param B:
        :param sol:
        """
        fslabels = sol.fslabels
        ledger = self.ledgers[fslabels]
        lbl, indim, with_t = parse_fslabels(fslabels)
        X = ledger.retrieve()
        outlabels = ', '.join(lbl)
        with_ref = (X.shape[1] == 2*len(lbl)+1)
        header = f"t, " + outlabels
        if with_ref:
            lblref = [x+"ref" for x in lbl]
            header += ", " + ', '.join(lblref)
        else:
            lblref = []
        filename = self.cog.filename(
            action=self,
            handle=fslabels,
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
        )
        np.savetxt(
            fname = filename,
            X = X,
            header = header
        )
        filename = self.cog.filename(
            action=self,
            handle=fslabels,
            phasei=B.phasei,
            stem="fig",
            ti=B.ti,
            ending="png",
        )
        if len(lbl) == 1:
            title = f"{outlabels}(t)"
            yrange = B.problem.p.range(lbl[0]) \
                if lbl[0] in B.problem.p.ranges else None
        else:
            title = f"[{outlabels}](t)"
            yrange = None
        self.fig.series(
            filename = filename,
            X = X,
            inlabel = "t",
            inidx = 0,
            outlabels = lbl+lblref,
            outidxs = range(1,X.shape[1]),
            title = title,
            text = None,
            xlim = B.problem.p.range('t'),
            ylim = yrange,
            t = None,
        )
        if with_ref:
            validate_msg = ""
            for i, lb in enumerate(lbl):
                Q = X[:,i+1:i+2]
                Qref = X[:,i+len(lbl)+1:i+len(lbl)+2]
                mse = ((Q - Qref)**2).mean()
                validate_msg += f"[Validation] {lb}(t): mse = {float(mse)}\n"
            self.log(validate_msg, end="")
            # todo store validation






    def _after_phase_1d(self, B, sol):
        """
        Create heatmap plots as time varies
        and create animations from the series plots
        for a more detailed, less colorful picture of
        evolution over time.

        :param B:
        :param sol:
        """
        lbl, indim, with_t = parse_fslabels(sol.fslabels)
        outdim = len(lbl) - indim
        for i in range(1, 1+outdim):
            # load the first one to get the size of X
            X, t = self._slice_loadtxt(B, sj=0, handle=sol.fslabels)
            X = np.hstack((np.full([X.shape[0],1], t), X))
            for sj in range(B.sj + 1):
                Xs, ts = self._slice_loadtxt(B, sj=sj, handle=sol.fslabels)
                Xs = np.hstack((np.full([Xs.shape[0],1],ts), Xs))
                X = np.vstack((X, Xs))
            lb = lbl[i]
            X_black = None
            linewidth = None # use default
            if lb in sol.features:
                features = sol.features[lb]
                if 'streamlines' in features:
                    method = features['streamlines']
                    X_black = method(B.problem).numpy()
                    linewidth = 1.0 # thin
            filename = self.cog.filename(
                action=self,
                handle=sol.fslabels,
                phasei=B.phasei,
                stem="fig",
                ending="png",
                ti=B.ti,
                tag=lbl[i] if outdim > 1 else None,
            )
            title = self.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                tag = f"{lb}(t, {lbl[0]})",
            )
            self.fig.heatmap(
                filename = filename,
                X = X,
                in1 = 0,
                in2 = 1,
                out1 = 2,
                lbl = ["t"] + lbl,
                title = title,
                value_range = B.problem.p.range(lbl[i]),
                plot_xray = True if i == 1 else False,
                # method = "linear",
                method = "nearest",
                X_black=X_black,
                X_black_linewidth=linewidth,
            )
        self._make_a_movie(B, sol.fslabels)

        #
        #
        #

        # todo - after_phase_1d
        #   generate series plot of error, animate




    def _after_phase_2d(self, B, sol):
        """
        Produce animation from heatmaps artifacts of time slices.

        :param B:
        :param sol:
        :return:
        """
        self._make_a_movie(B, sol.fslabels)



    def _after_phase_3d(self):
        raise NotImplementedError

















