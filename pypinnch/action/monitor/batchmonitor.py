






from .. import Probe

from ..._impl.residual import Periodic

import numpy as np


class BatchMonitor(Probe):
    """
    Probe to monitor batches as they pass through the training.
    Useful for sanity checking, debugging, and improving user intuition.

    """

    # todo known issues with time-independent cases.

    # todo review, improve

    # todo a graph with sawtooth shapes, plotting
    #  the individual age counters as they count up and fall back to 0.
    #  Such a plot would immediately reveal any outliers or anomalies.

    def __init__(self):
        super().__init__()
        self.batchinfo = "<><>batch information sheet<><>\n\n"
        self.batch_plot_counter = None


    def on_phase(self, B):
        info = self.batchinfo
        # > initialize plot counter
        self.batch_plot_counter = 0
        # > print information sheet
        bs = B.phase.batchsize
        SPL = B.phase.samplesets.SPL
        info += f"batchsize: {bs}\n"
        info += f"SPL: {SPL}"
        if B.problem.with_t:
            info += f"\nICs:\n"
            sz = B.phase.samplesets.icbase.size()
            info += f"1 age = {sz} samples or {int(sz/bs)} batches\n"
            info += "\n-\n\n"
        for lb in B.phase.samplesets.active_csss:
            css = B.phase.samplesets.csss[lb]
            # c is a BatchCylinder instance.
            # c.c is a Constraint instance. # todo awk - but how to improve?
            info += f"constraint {lb}:\n"
            sz = css.cyl.size()
            base_sz = css.cyl.base.shape[0] if css.cyl.base is not None else 1
            info += f"1 age = {sz} = {css.cyl.nsamples_1d} x {base_sz} samples or {int(sz/bs)} batches\n"
            if isinstance(css.constraint.residual, Periodic):
                info += f"(Periodic)\n"
            info += "\t((((\n"
            if B.problem.indim > 0:
                # Here, we calculate what the total sample size "should" be
                # based on geometry and the value of SPL.
                m = css.measure() # cylinder measure, includes time
                idim = css.constraint.source.internal_dimension()
                SPM = SPL**idim
                info += f"\t\tm {m}    (source measure)\n"
                info += f"\t\tidim {idim}    (internal dimension)\n"
                info += f"\t\tspm {SPM}    (SPM)\n"
                info += f"\t\tm x spm = {m*SPM}\n"
                info += f"\t\tm x spm x spm1d = {m*SPM*SPL}\n"
            else:
                # todo what calculate here?
                pass
                # deprecated:
                # b = B.phase.samplesets.icbase.X.shape[0]
                # info += f"\t\tb {b}      (base sample set size)\n"
                # info += f"\t\tb x spl = {b*SPL}\n"
            info += "\t))))\n"
            if B.problem.indim > 0:
                info += "\t((((\n"
                tss = css.cyl.sampleset.shape[0]
                info += f"\t\tsample set size {tss}\n"
                tbs = css.cyl.batchsize
                info += f"\t\tbatchsize {tbs}\n"
                info += "\t))))\n"
                info += "\n"
            info += "\n-\n\n"
        info += "\n<><><><><>end<><><><><>\n"
        filename = self.cog.filename(
            action=self,
            handle="batchinfo",
            stem="",
            ending="txt",
            driveri=B.driveri,
            phasei=B.phasei,
        )
        with open(filename, 'w') as f:
            f.write(info)


    def on_train(self, B):
        self.batch_plot_counter = 0


    def after_batch(self, B, BB):
        if B.problem.with_t:
            if self.batch_plot_counter < 10:
                if B.problem.indim == 0:
                    self.after_batch_0dt(B, BB)
                elif B.problem.indim == 1:
                    self.after_batch_1dt(B, BB)
                elif B.problem.indim == 2:
                    self.after_batch_2dt(B, BB)
                elif B.problem.indim == 3:
                    self.after_batch_3dt(B, BB)
                else:
                    self.log("[SampleMonitor] Unsupported input dimension.")
                self.batch_plot_counter += 1
            else:
                pass
        else:
            # > time independent routines (separated in case it is useful)
            if self.batch_plot_counter < 10:
                if B.problem.indim == 1:
                    self.after_batch_1d(B, BB)
                elif B.problem.indim == 2:
                    self.after_batch_2d(B, BB)
                elif B.problem.indim == 3:
                    self.after_batch_3d(B, BB)
                else:
                    self.log("[SampleMonitor] Unsupported input dimension.")
                self.batch_plot_counter += 1


    def after_batch_0dt(self, B, BB):
        t = B.phase.samplesets.icbase.t
        Xs = [BB.XX.clone().detach().cpu().numpy()]
        xs = [None]
        for Xi_ in BB.XXs:
            Xi = Xi_.clone().detach().cpu().numpy()
            Xs.append(Xi)
            xs.append(None)
        filename = self.cog.filename(
            action=self,
            handle="batch",
            stem="",
            ending="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag=str(self.batch_plot_counter),
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag="batch"
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


    def after_batch_1dt(self, B, BB):
        t = B.phase.samplesets.icbase.t
        Xbase = BB.XX.clone().detach().cpu().numpy()
        Xbase = np.hstack((Xbase[:,1:2], Xbase[:,0:1]))
        Xs = [Xbase]
        xs = [None]
        Xslabels = ["base"]
        for i, Xi_ in enumerate(BB.XXs):
            Xi = Xi_.clone().detach().cpu().numpy()
            # we have to reverse the order because time comes last :/
            Xi = np.hstack((Xi[:,1:2], Xi[:,0:1]))
            Xs.append(Xi)
            xs.append(None)
            # todo use the constraint label instead
            Xslabels.append(f"c{i}")
        filename = self.cog.filename(
            action=self,
            handle="batch",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag=str(self.batch_plot_counter),
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag="batch"
        )
        bb = B.problem.bounding_box()
        self.fig.scatterplot(
            filename=filename,
            title=title,
            xlim=(B.phase.th.tinit - B.phase.shelf, B.phase.th.tfinal + B.phase.shelf),
            ylim=(bb.mins[0], bb.maxs[0]),
            vlines=(t, t + B.phase.th.stepsize()),
            vlineslim=(bb.mins[0], bb.maxs[0]),
            Xs=Xs,
            xs=xs,
            Xslabels=Xslabels,
        )





    def after_batch_2dt(self, B, BB):
        t = B.phase.samplesets.icbase.t
        lbl = B.models[0].lbl
        Xs = [BB.XX.clone().detach().cpu().numpy()]
        # Xslabels = ["base"]
        for i, Xi_ in enumerate(BB.XXs):
            Xi = Xi_.clone().detach().cpu().numpy()
            Xs.append(Xi)
            # Xslabels.append(f"c{i}")
        filename = self.cog.filename(
            action=self,
            handle="batch",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag=str(self.batch_plot_counter),
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            tag=f"batch t {t}",
        )
        self.fig.scatterplot3d(
            filename=filename,
            Xs=Xs,
            in1=0,
            in2=1,
            in3=2,
            lbl=lbl[0:2] + ["t"],
            title=title,
            show=False,
        )



    def after_batch_3dt(self, B, BB):
        self.log("[BatchMonitor] Routine has not been implemented yet.")


    def after_batch_1d(self, B, BB):
        # todo work in progress
        Xs = []
        xs = []
        for Xi_ in BB.XXs:
            Xi = Xi_.clone().detach().cpu().numpy()
            Xs.append(Xi)
            xs.append(None)
        filename = self.cog.filename(
            action=self,
            handle="batch",
            stem="",
            ending="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag=str(self.batch_plot_counter),
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag="batch"
        )
        bb = B.problem.bounding_box()
        self.fig.scatterplot(
            filename=filename,
            X=None,
            x=None,
            title=title,
            xlim=(bb.mins[0], bb.maxs[0]),
            ylim=(-1.0, 1.0),
            Xs=Xs,
            xs=xs,
        )





    def after_batch_2d(self, B, BB):
        # todo work in progress
        Xs = []
        xs = [None]
        Xslabels = []
        for i, Xi_ in enumerate(BB.XXs):
            Xi = Xi_.clone().detach().cpu().numpy()
            Xs.append(Xi)
            xs.append(None)
            # todo use the constraint label instead
            Xslabels.append(f"c{i}")
        filename = self.cog.filename(
            action=self,
            handle="batch",
            stem="png",
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag=str(self.batch_plot_counter),
        )
        title = self.cog.title(
            driveri=B.driveri,
            phasei=B.phasei,
            ti=B.ti,
            tr=B.tr,
            level=B.L,
            tag="batch"
        )
        bb = B.problem.bounding_box()
        self.fig.scatterplot(
            filename=filename,
            title=title,
            xlim=(bb.mins[0], bb.maxs[0]),
            ylim=(bb.mins[1], bb.maxs[1]),
            Xs=Xs,
            xs=xs,
            Xslabels=Xslabels,
        )


    def after_batch_3d(self, B, BB):
        self.log("[BatchMonitor] Routine has not been implemented yet.")








