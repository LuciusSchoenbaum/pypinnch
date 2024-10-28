




import numpy as np

from .. import Probe



class SampleMonitor(Probe):
    """

    Sample set inspection

    Collect the batches in the time cylinder, and emit them
    to allow monitoring of inputs to training.

    Parameters:

        show_in_view (boolean):
            Whether to pause training at start of train in order to
            show (via a GUI launched in a viewing window)
            a plot of the sample. STIUYKB.
            May work only for 2d case ITCINOOD.

    """

    def __init__(
            self,
            show_in_view=False,
    ):
        super().__init__()
        self.show = show_in_view


    def on_train(self, B):
        if B.problem.indim == 0:
            self.on_train_0d(B)
        elif B.problem.indim == 1:
            self.on_train_1d(B)
        elif B.problem.indim == 2:
            self.on_train_2d(B)
        elif B.problem.indim == 3:
            self.on_train_3d(B)
        else:
            self.log("[SampleMonitor] Unsupported input dimension.")


    def gate_iterloop(self, B, BB):
        # print a visualization of the sample set as it is received
        # (it has been contracted/expanded prior to training.)
        # todo
        pass


    #######
    # 0d


    def on_train_0d(self, B):
        # base.X has the form [u]
        # X = B.phase.samplesets.icbase.X
        t = B.phase.samplesets.icbase.t
        Xs = []
        xs = []
        for lb in B.phase.samplesets.active_csss:
            css = B.phase.samplesets.csss[lb]
            Xi = css.cyl.sampleset.cpu().numpy()
            Xs.append(Xi)
            xs.append(None)
        filename = self.cog.filename(
            action=self,
            handle="sample",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag="sample"
        )
        self.fig.scatterplot(
            filename=filename,
            X=None,
            x=None,
            title=title,
            xlim=(B.phase.th.tinit - B.phase.shelf, B.phase.th.tfinal + B.phase.shelf),
            ylim=(-1.0, 1.0),
            vlines=(t, t + B.phase.th.stepsize()),
            vlineslim=(-1.0, 0.0),
            Xs=Xs,
            xs=xs,
        )

    #######
    # 1d


    def on_train_1d(self, B):
        if B.problem.with_t:
            ti = B.ti
            # base.X [x,u1,u2,...,un]
            X = B.phase.samplesets.icbase.X[:,0:1]
            X = X.cpu().numpy()
            t = B.phase.samplesets.icbase.t
        else:
            ti = None
            X = None
            t = None
        Xs = []
        xs = []
        Xslabels = []
        for lb in B.phase.samplesets.active_csss:
            css = B.phase.samplesets.csss[lb]
            Xi = css.cyl.X().cpu().numpy()
            # we have to reverse the order bc time comes last :/
            Xi = np.hstack((Xi[:,1:2], Xi[:,0:1]))
            Xs.append(Xi)
            xs.append(None)
            Xslabels.append(lb)
        filename = self.cog.filename(
            action=self,
            handle="sample",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=ti,
            tr=B.tr,
            level=B.L,
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=ti,
            tr=B.tr,
            level=B.L,
            tag="samples"
        )
        bb = B.problem.bounding_box()
        if X is not None:
            self.fig.scatterplot(
                filename=filename,
                X=X,
                x=t,
                title=title,
                xlim=(B.phase.th.tinit - B.phase.shelf, B.phase.th.tfinal + B.phase.shelf),
                ylim=(bb.mins[0], bb.maxs[0]),
                vlines=(t, t + B.phase.th.stepsize()),
                vlineslim=(bb.mins[0], bb.maxs[0]),
                Xs=Xs,
                xs=xs,
                Xslabels=Xslabels,
            )
        else:
            # > 1d time independent SampleMonitor plot
            raise NotImplementedError



    #######
    # 2d


    def on_train_2d(self, B):
        if B.problem.with_t:
            ti = B.ti
            # todo pitstop etc.
            # base.X [x,y,u1, u2, ..., un]
            X = B.phase.samplesets.icbase.X[:,0:2]
            X = X.cpu().numpy()
            t = B.phase.samplesets.icbase.t
        else:
            ti = None
            X = None
            t = None
        # todo this code assumes models is a list of size 1.
        lbl = B.models[0].lbl
        if X is not None:
            # > plot icbase
            filename = self.cog.filename(
                action=self,
                handle="sample_base",
                stem="png",
                driveri=B.driveri,
                phasei=B.phasei,
                ti=ti,
                tr=B.tr,
                level=B.L,
            )
            title = self.cog.title(
                driveri=B.driveri,
                phasei=B.phasei,
                ti=ti,
                tr=B.tr,
                tag="base sample"
            )
            bb = B.problem.bounding_box()
            self.fig.scatterplot(
                filename=filename,
                X=X,
                title=title,
                xlim=(bb.mins[0], bb.maxs[0]),
                ylim=(bb.mins[1], bb.maxs[1]),
            )
        Xs = []
        xs = []
        Xslabels = []
        for lb in B.phase.samplesets.active_csss:
            css = B.phase.samplesets.csss[lb]
            Xi = css.cyl.X().cpu().numpy()
            Xs.append(Xi)
            xs.append(None)
            Xslabels.append(lb)
        filename = self.cog.filename(
            action=self,
            handle="sample",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=ti,
            tr=B.tr,
            level=B.L,
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=ti,
            tr=B.tr,
            tag="sample"
        )
        if X is not None:
            # > add third dimension for time axis
            self.fig.scatterplot3d(
                filename=filename,
                Xs=Xs,
                in1=0,
                in2=1,
                in3=2,
                lbl=lbl[0:2] + ["t"],
                title=title,
                show=self.show,
            )
        else:
            # > 2d scatterplot of "base" view
            bb = B.problem.bounding_box()
            self.fig.scatterplot(
                filename=filename,
                Xs=Xs,
                # awk
                xs=len(Xs)*[None],
                Xslabels=Xslabels,
                title=title,
                xlim=(bb.mins[0], bb.maxs[0]),
                ylim=(bb.mins[1], bb.maxs[1]),
            )


    #######
    # 3d


    def on_train_3d(self, B):
        self.log("[SampleMonitor] Routine not been implemented yet.")



