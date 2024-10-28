


import numpy as np
from numpy import loadtxt as numpy_loadtxt

from ..types import \
    get_reference, \
    get_time_from_header


from mv1fw.visutil import Animation
from mv1fw import(
    get_fslabels,
    parse_fslabels,
    sortdown,
)


class ResultPlotter:
    """

    Handles the generation of plots and animations
    for the :any:`Result` action.
    This separation of responsibilities is mainly
    so that :any:`Result` is more surveyable,
    and because this way :any:`Result` has no
    depend on Numpy, while :any:`ResultPlotter`
    has no dependence on any ML framework.

    Parameters:

        result (:any:`Result`):
        skip_animations (boolean):

    """

    # todo binary format optional
    #  (cf. numpy_loadtxt)

    def __init__(
            self,
            result,
            skip_animations,
    ):
        # reference to result so that ResultPlotter impersonates Result
        self.result = result
        self.skip_animations = skip_animations
        # frame duration for animation artifacts (ms)
        self.frame_duration = 1000


    def after_subslice(
            self,
            B,
            sol,
            Xreg,
            sk,
            ti_,
            vres,
    ):
        """
        Plotting operation after a slice or subslice.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):
            Xreg (:any:`XFormat`):
            sk (optional integer):
                todo doc substepping
            ti_ (todo):
                todo - there may be a minor snafu here
            vres (integer):
                visualization resolution

        """
        fslabels = sol.fslabels
        _, indim, _ = parse_fslabels(fslabels)
        if indim == 0:
            self.after_subslice_0dt(B, sol, Xreg, sk, ti_, vres)
        elif indim == 1:
            self.after_subslice_1dt(B, sol, Xreg, sk, ti_, vres)
        elif indim == 2:
            self.after_subslice_2dt(B, sol, Xreg, sk, ti_, vres)
        elif indim == 3:
            self.after_subslice_3dt(B, sol, Xreg, sk, ti_, vres)
        else:
            raise NotImplementedError



    def after_phase(
            self,
            B,
            sol,
    ):
        """
        Plotting operation after the conclusion of a training phase.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        _, indim, _ = parse_fslabels(fslabels)
        if indim == 0:
            self.after_phase_0dt(B, sol)
        elif indim == 1:
            self.after_phase_1dt(B, sol)
        elif indim == 2:
            self.after_phase_2dt(B, sol)
        elif indim == 3:
            self.after_phase_3dt(B, sol)
        else:
            raise NotImplementedError



    #########################################################
    # 0-dt


    def after_subslice_0dt(self, B, sol, Xreg, sk, ti_, vres):
        """
        Nothing to do; plot after phase finishes.

        """
        pass



    def after_phase_0dt(self, B, sol):
        """
        Load stored time-wise data (in the 0-d case,
        the problem must be time-dependent)
        and plot a time series.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        ref = get_reference(fslabels=fslabels, references=B.problem.references)
        if ref is not None:
            lbl += [x+"ref" for x in lbl[indim:]]
            fslabels = get_fslabels(lbl, indim, with_t)
        X, _ = self.load(B, handle=fslabels)
        filename = self.result.cog.filename(
            action=self.result,
            handle=fslabels,
            phasei=B.phasei,
            stem="fig",
            ti=B.ti,
            ending="png",
        )
        # todo this looks like it can be improved
        outlbl = lbl[indim:]
        outlabels = ', '.join(outlbl)
        if len(lbl) == 1:
            title = f"{outlabels}(t)"
            yrange = B.problem.p.range(lbl[0]) \
                if lbl[0] in B.problem.p.ranges else None
        else:
            title = f"[{outlabels}](t)"
            yrange = None
        self.result.fig.series(
            filename = filename,
            X = X,
            inlabel = "t",
            inidx = 0,
            outlabels = outlbl,
            outidxs = range(1,X.shape[1]),
            title = title,
            text = None,
            xlim = B.phase.th.range(),
            ylim = yrange,
            t = None,
        )


    # 0-dt
    #########################################################
    # 1-dt


    def after_subslice_1dt(self, B, sol, Xreg, sk, ti_, vres):
        """
        Generate a line plot for the 1d outputs and
        references, if any, for the current timeslice.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):
            Xreg (:any:`XFormat`):
            sk (optional integer):
            ti_ (optional integer):
                todo docs
            vres (integer):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        outdim = len(lbl) - indim
        ref = get_reference(fslabels=fslabels, references=B.problem.references)
        X0 = Xreg.X()
        t = Xreg.t()
        if with_t:
            ttag = f" t {t:.4f}"
            ti = B.ti
            sj = B.sj
        else:
            ttag = ""
            ti = None
            sj = None
        for i in range(indim, indim+outdim):
            outidxs = [i]
            outlb = lbl[i]
            if ref is not None:
                outidxs.append(i+outdim)
            filename = self.result.cog.filename(
                action=self.result,
                handle=get_fslabels(lbl[:indim]+[outlb], indim, with_t),
                driveri=B.driveri,
                phasei=B.phasei,
                stem="fig",
                ending="png",
                ti=ti,
                sj=sj,
                sk=sk,
                # interpolation resolution (v for visualization)
                tag=f"v{vres}",
            )
            title = self.result.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=ti,
                sj=sj,
                sk=sk,
                tag = ' '.join([f"{outlb}({', '.join(lbl[:indim])})", ttag]),
            )
            # todo check necessary in regular grid case
            X0 = sortdown(X0, 0)
            # filename_tag = "sol"
            # title_tag = ""
            # filename = self.result.cog.tag_filename(filename, filename_tag)
            # title = self.result.cog.tag_title(title, title_tag)
            self.result.fig.series(
                filename = filename,
                X = X0,
                inlabel = lbl[0],
                inidx = 0,
                outlabels = [outlb, outlb+"ref"] if ref is not None else [outlb],
                outidxs = outidxs,
                title = title,
                text = None,
                xlim = B.problem.p.range(lbl[0]),
                ylim = B.problem.p.range(lbl[outidxs[0]]),
                half_linestyle = "dashed" if ref is not None else None,
            )
        if ref is not None:
            for i in range(indim, indim+outdim):
                Qerr = np.absolute(X0[:,i:i+1] - X0[:,i+outdim:i+outdim+1])
                XQerr = np.hstack((X0[:,:indim], Qerr))
                outlb = lbl[i]
                filename = self.result.cog.filename(
                    action=self.result,
                    handle=get_fslabels(lbl[:indim]+[outlb], indim, with_t),
                    phasei=B.phasei,
                    stem=self.result.error_stem,
                    ending="png",
                    ti=ti,
                    sj=sj,
                    sk=sk,
                    # interpolation resolution (v for visualization)
                    tag=f"v{vres}.err",
                )
                title = self.result.cog.title(
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=ti,
                    sj=sj,
                    sk=sk,
                    level=B.L,
                    tag = ' '.join([f"|{outlb} - ref|({', '.join(lbl[:indim])})", ttag]),
                )
                # SET THE VALUE RANGE FOR ERROR PLOTS.
                # There's no perfect way.
                # We make two plots,
                # one in the problem's range
                # and one in the error's own range.
                # we don't give the user control...it's too much hassle
                # when this is probably (?) sufficient.
                vr0, vr1 = B.problem.p.range(outlb)
                value_range = (0.0, vr1 - vr0)
                # > the "alt" plot allows the error's range to be selected automatically
                # todo better tag than 'alt'?
                value_range_alt = None
                filename_alt = self.result.cog.tag_filename(filename, "alt")
                self.result.fig.series(
                    filename = filename,
                    X = XQerr, # ncols is always 2
                    inlabel = lbl[0],
                    inidx = 0,
                    outlabels = [outlb+"err"],
                    outidxs = [1],
                    title = title,
                    text = None,
                    xlim = B.problem.p.range(lbl[0]),
                    ylim = value_range,
                )
                self.result.fig.series(
                    filename = filename_alt,
                    X = XQerr,
                    inlabel = lbl[0],
                    inidx = 0,
                    outlabels = [outlb+"err"],
                    outidxs = [1],
                    title = title,
                    text = None,
                    xlim = B.problem.p.range(lbl[0]),
                    ylim = value_range_alt,
                )


    def after_phase_1dt(self, B, sol):
        """
        Create heatmap plots as time varies
        and create animations from the series plots
        for a more detailed, less colorful picture of
        evolution over time.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        if not with_t:
            # > for time-independent problem there is nothing else to do
            pass
        else:
            outdim = len(lbl) - indim
            ref = get_reference(fslabels=fslabels, references=B.problem.references)
            if ref:
                handle = get_fslabels(lbl + [x+'ref' for x in lbl[indim:]], indim, with_t)
            else:
                handle = sol.fslabels
            for i in range(indim, indim+outdim):
                sk = None if sol.substep == 1 else 0
                # load the first slice to get the size of X
                # and handle the exceptional substep
                X, t = self.load(B, sj=0, sk=sk, handle=handle)
                X = np.hstack((np.full([X.shape[0],i], t), X))
                for sj in range(sol.every, B.sj + 1, sol.every):
                    for sk_ in range(sol.substep):
                        sk = None if sol.substep == 1 else sk_
                        Xs, ts = self.load(B, sj=sj, sk=sk, handle=handle)
                        Xs = np.hstack((np.full([Xs.shape[0],i],ts), Xs))
                        X = np.vstack((X, Xs))
                lb = lbl[i]
                # features
                # todo a feature is a plotting feature, it should be
                #  wrapped as a data structure and a list of these
                #  should be passable into fig methods (e.g. heatmap),
                #  where they can be uniformly processed in a loop.
                X_black = None
                X_black_linewidth = None # use default typ. 2.0
                if lb in sol.features:
                    features = sol.features[lb]
                    if 'streamlines' in features:
                        method = features['streamlines']
                        X_black = method(B.problem).numpy()
                        X_black_linewidth = 1.0 # thin
                    # more features can be added here
                filename = self.result.cog.filename(
                    action=self.result,
                    handle=sol.fslabels,
                    phasei=B.phasei,
                    stem="fig",
                    ending="png",
                    ti=B.ti,
                    tag=lbl[i] if outdim > 1 else None,
                )
                title = self.result.cog.title(
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=B.ti,
                    tag = f"{lb}(t, {lbl[0]})",
                )
                colormaps = B.problem.p.colormaps
                colormap = colormaps[lbl[i]] if lbl[i] in colormaps else None
                self.result.fig.heatmap(
                    filename = filename,
                    X = X,
                    in1 = 0,
                    in2 = 1,
                    out1 = 2,
                    lbl = ["t", lbl[0], lbl[i]],
                    title = title,
                    value_range = B.problem.p.range(lbl[i]),
                    plot_xray = True if i == 1 else False,
                    # method = "linear",
                    method = "nearest",
                    X_black=X_black,
                    X_black_linewidth=X_black_linewidth,
                    colormap=colormap,
                )
                # todo this is a little bit of a kludge, improve?
                #  e.g. implicit assumption is that rng is a value range.
                if "alternate_ranges" in dir(B.problem.p):
                    if lbl[i] in B.problem.p.alternate_ranges:
                        alternate_ranges = B.problem.p.alternate_ranges[lbl[i]]
                        for rngi, rng in enumerate(alternate_ranges):
                            srngi = str(rngi).zfill(2) if len(alternate_ranges) > 10 else str(rngi)
                            filename_ = self.result.cog.tag_filename(filename, "alt" + srngi)
                            self.result.fig.heatmap(
                                filename = filename_,
                                X = X,
                                in1 = 0,
                                in2 = 1,
                                out1 = 2,
                                lbl = ["t", lbl[0], lbl[i]],
                                title = title,
                                value_range = rng,
                                plot_xray = False,
                                # method = "linear",
                                method = "nearest",
                                X_black=X_black,
                                X_black_linewidth=X_black_linewidth,
                                colormap=colormap,
                            )
                if ref is not None:
                    filename = self.result.cog.filename(
                        action=self.result,
                        handle=get_fslabels(lbl[:indim]+[lb], indim, with_t),
                        phasei=B.phasei,
                        stem=self.result.error_stem,
                        ending="png",
                        ti=B.ti,
                        sj=B.sj,
                        sk=sk,
                        # interpolation resolution (v for visualization)
                        tag=f"err",
                    )
                    Q = np.absolute(X[:,2:3] - X[:,3:4])
                    # > replace the out column with |Q - Qref|, just do this in place
                    X[:,2:3] = Q
                    self.result.fig.heatmap(
                        filename = filename,
                        X = X,
                        in1 = 0,
                        in2 = 1,
                        out1 = 2,
                        lbl = ["t", lbl[0], lbl[i]], # todo I'm 90% sure that lbl[i] should be lbl[i]+'err'
                        title = title + " err", # todo review
                        value_range = B.problem.p.range(lbl[i]),
                        plot_xray = False,
                        # method = "linear",
                        method = "nearest",
                        # > keep streamlines, if any
                        X_black=X_black,
                        X_black_linewidth=X_black_linewidth,
                        colormap=colormap,
                    )
                    # error in error range plot, in "Reds" colormap
                    filename_ = self.result.cog.tag_filename(filename, 'reds')
                    self.result.fig.heatmap(
                        filename = filename_,
                        X = X,
                        in1 = 0,
                        in2 = 1,
                        out1 = 2,
                        lbl = ["t", lbl[0], lbl[i]], # todo I'm 90% sure that lbl[i] should be lbl[i]+'err'
                        title = title + " err", # todo review
                        value_range = None,
                        plot_xray = False,
                        # method = "linear",
                        method = "nearest",
                        # > keep streamlines, if any
                        X_black=X_black,
                        X_black_linewidth=X_black_linewidth,
                        colormap='Reds',
                    )
                    if "alternate_ranges" in dir(B.problem.p):
                        if lbl[i] in B.problem.p.alternate_ranges:
                            alternate_ranges = B.problem.p.alternate_ranges[lbl[i]]
                            for rngi, rng in enumerate(alternate_ranges):
                                srngi = str(rngi).zfill(2) if len(alternate_ranges) > 10 else str(rngi)
                                filename_ = self.result.cog.tag_filename(filename, "alt" + srngi)
                                self.result.fig.heatmap(
                                    filename = filename_,
                                    X = X,
                                    in1 = 0,
                                    in2 = 1,
                                    out1 = 2,
                                    lbl = ["t", lbl[0], lbl[i]],
                                    title = title + " err", # todo review
                                    value_range = rng,
                                    plot_xray = False,
                                    # method = "linear",
                                    method = "nearest",
                                    X_black=X_black,
                                    X_black_linewidth=X_black_linewidth,
                                    colormap=colormap,
                                )
            # > make animation for the subslice 1d plots
            self.make_animation(B, sol)
            # todo after_phase_1d generate series plot of error?




    # 1-dt
    #########################################################
    # 2-dt



    def after_subslice_2dt(self, B, sol, Xreg, sk, ti_, vres):
        """
        Generate heatmap visualizations of input vs. output at time t.
        In order to produce deliverable outputs, we must first interpolate
        onto a regular grid, then pass this to the plotting routine,
        which interpolates a second time. This may be necessary due to
        the backend interpolation algorithm working better on a
        regular input, or at worst inputs with regularly-shaped Voronoi
        cells (this may be the case), but we just aren't sure.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):
            Xreg (:any:`XFormat`):
            sk (optional integer):
            ti_ (optional integer):
            vres (integer):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        outdim = len(lbl) - indim
        ref = get_reference(fslabels=fslabels, references=B.problem.references)
        colormaps = B.problem.p.colormaps
        X0 = Xreg.X()
        t = Xreg.t()
        if with_t:
            ttag = f" t {t:.4f}"
            ti = B.ti
            sj = B.sj
        else:
            ttag = ""
            ti = None
            sj = None
        for i in range(indim, indim+outdim):
            outidxs = [i]
            outlb = lbl[i]
            if ref is not None:
                outidxs.append(i+outdim)
            filename = self.result.cog.filename(
                action=self.result,
                handle=get_fslabels(lbl[:indim]+[outlb], indim, with_t),
                driveri=B.driveri,
                phasei=B.phasei,
                stem="fig",
                ending="png",
                ti=ti,
                sj=sj,
                sk=sk,
                # interpolation resolution (v for visualization)
                tag=f"v{vres}",
            )
            title = self.result.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=ti,
                sj=sj,
                sk=sk,
                tag = ' '.join([f"{outlb}({', '.join(lbl[:indim])})", ttag]),
            )
            is_ref = False
            for outidx in outidxs:
                filename_tag = "ref" if is_ref else "sol"
                title_tag = "ref" if is_ref else ""
                filename_ = self.result.cog.tag_filename(filename, filename_tag)
                title = self.result.cog.tag_title(title, title_tag)
                colormap = colormaps[outlb] if outlb in colormaps else None
                self.result.fig.heatmap(
                    filename = filename_,
                    X = X0,
                    in1 = 0,
                    in2 = 1,
                    out1 = outidx,
                    lbl = lbl,
                    title = title,
                    value_range = B.problem.p.range(outlb),
                    color_label = outlb,
                    plot_xray = True if i == 0 and B.ti == 0 and B.sj == 0 and ti_ == 0 else False,
                    # cubic, linear, ...
                    method = "linear",
                    xlim=B.problem.p.range(lbl[0]),
                    ylim=B.problem.p.range(lbl[1]),
                    colormap=colormap,
                )
                is_ref = True
        if ref is not None:
            for i in range(indim, indim+outdim):
                Qerr = np.absolute(X0[:,i:i+1] - X0[:,i+outdim:i+outdim+1])
                XQerr = np.hstack((X0[:,:indim], Qerr))
                outlb = lbl[i]
                colormap = colormaps[outlb] if outlb in colormaps else None
                filename = self.result.cog.filename(
                    action=self.result,
                    handle=get_fslabels(lbl[:indim]+[outlb], indim, with_t),
                    phasei=B.phasei,
                    stem=self.result.error_stem,
                    ending="png",
                    ti=ti,
                    sj=sj,
                    sk=sk,
                    # interpolation resolution (v for visualization)
                    tag=f"v{vres}.err",
                )
                title = self.result.cog.title(
                    driveri=B.driveri,
                    phasei=B.phasei,
                    ti=ti,
                    sj=sj,
                    sk=sk,
                    level=B.L,
                    tag = ' '.join([f"|{outlb} - ref|({', '.join(lbl[:indim])})", ttag]),
                )
                # SET THE VALUE RANGE FOR ERROR PLOTS.
                # There's no perfect way.
                # We make two plots,
                # one in the problem's range
                # and one in the error's own range.
                # we don't give the user control...it's too much hassle
                # when this is probably (?) sufficient.
                vr0, vr1 = B.problem.p.range(outlb)
                value_range = (0.0, vr1 - vr0)
                filename_reds = self.result.cog.tag_filename(filename, 'reds')
                self.result.fig.heatmap(
                    filename = filename,
                    X = XQerr,
                    in1 = 0,
                    in2 = 1,
                    out1 = i,
                    lbl = lbl,
                    title = title,
                    value_range = value_range,
                    color_label = outlb,
                    plot_xray = False,
                    # cubic, linear, ...
                    method = "linear",
                    xlim=B.problem.p.range(lbl[0]),
                    ylim=B.problem.p.range(lbl[1]),
                    colormap=colormap,
                )
                self.result.fig.heatmap(
                    filename = filename_reds,
                    X = XQerr,
                    in1 = 0,
                    in2 = 1,
                    out1 = i,
                    lbl = lbl,
                    title = title,
                    value_range = None,
                    color_label = outlb,
                    plot_xray = False,
                    # cubic, linear, ...
                    method = "linear",
                    xlim=B.problem.p.range(lbl[0]),
                    ylim=B.problem.p.range(lbl[1]),
                    # white: no error black: error
                    # colormap="binary",
                    # green: no error yellow: error
                    # colormap="summer",
                    # white: no error blue: error
                    # colormap="Blues",
                    # white: no error red: error
                    colormap="Reds",
                )



    def after_phase_2dt(self, B, sol):
        """
        Produce animation from heatmaps artifacts of time slices,
        or do nothing in the time-independent case.

        Arguments:
            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        self.make_animation(B, sol)


    # 2-dt
    #########################################################
    # 3-dt


    def after_phase_3dt(self, B, sol):
        """
        Produce animation from artifacts of time slices.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        raise NotImplementedError



    def after_subslice_3dt(self, B, sol, Xreg, sk, ti, vres):
        """
        Not implemented yet.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):
            Xreg (:any:`XFormat`):
            sk (optional integer):
            ti (optional integer):
            vres (integer):

        """
        raise NotImplementedError


    # 3-dt
    ########################################################
    # Helpers (generic for any dimension)



    def load(self, B, handle, sj = None, sk = None):
        """
        Dimension-independent code for loading the base sample set.
        Used to make collective artifacts (as time varies).

        todo doc for sj, sk

        Arguments:

            B (:any:`Bundle`):
            handle (string):
            sj (optional integer):
            sk (optional integer):

        """
        filename = self.result.cog.filename(
            action=self.result,
            handle=handle,
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
            sj=sj,
            sk=sk,
        )
        if sj is not None:
            t = get_time_from_header(filename)
        else:
            t = None
        X = numpy_loadtxt(filename)
        return X, t



    def make_animation(self, B, sol):
        """
        Create an animation from a set of time-dependent
        generated plots.

        Arguments:

            B (:any:`Bundle`):
            sol (:any:`Solution`):

        """
        fslabels = sol.fslabels
        lbl, indim, with_t = parse_fslabels(fslabels)
        if not with_t:
            pass
        else:
            outdim = len(lbl) - indim
            anim = Animation()
            if indim == 1:
                # 1d case:
                tags = ["", "err"]
                fc_limit = 55
            elif indim == 2:
                # 2d case
                tags = ["sol", "ref", "err"]
                fc_limit = 35
            else:
                raise NotImplementedError
            for i in range(indim, indim+outdim):
                for tag in tags:
                    stem = self.result.error_stem if tag == "err" else "fig"
                    vtags = [f"v{x}." + tag for x in self.result.vresolution]
                    for vtag in vtags:
                        frame_glob = self.result.cog.filename(
                            action=self.result,
                            handle=fslabels,
                            phasei=B.phasei,
                            stem=stem,
                            ending="png",
                            ti=B.ti,
                            sj="*",
                            sk="*" if sol.substep > 1 else None,
                            tag=vtag,
                        )
                        filename = self.result.cog.filename(
                            action=self.result,
                            handle=fslabels,
                            stem=stem,
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



