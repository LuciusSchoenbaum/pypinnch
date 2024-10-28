





import numpy as np
from os import remove

from .action_impl import Probe


from mv1fw.fw import get_dtype


class LossCurves(Probe):
    """
    Tracking of loss over training iteration
    (so-called learning curves or loss curves)
    as well as the learning rate.

    This note describes the data formatting::

        X:

        Each column is a loss. If there are M ic constraints and N constraints,

        ic[0] ic[1]... ic[M-1] cs[0]   cs[1]...cs[N-1]        total
        L       L           L           L         L         L               ∑L         (iter #1)
        L       L           L           L         L         L               ∑L         (iter #2)
        ...   ...    ...
        L       L            L           L         L        L               ∑L         (the last iter of training)

        The total # of columns is M+N+1.
        The total # of rows is the minimum of max_iterations, and the iteration
        when the tolerance was reached.


        Y:

        This array stores averages taken over epochs.
        Additional columns for the "kit" used during training, where
        lr is the learning rate applied during the epoch by lr_sched,
        and iter is the iteration when the epoch was reached.

        ic[0] ic[1]... ic[M-1]  cs[0]   cs[1]   cs[2]   ...     cs[N-1]     total      kit         iter
        L       L           L            L       L           L                   L            ∑L   [4cols]       .        (epoch #1)
        L       L           L            L       L           L                   L            ∑L     . . . .         .        (epoch #2)
        ...   ...    ...
        L       L            L           L       L           L                   L            ∑L     . . . .         .       (last epoch:  maybe a "short" epoch)

        The total # of columns is M+N+1+len(kit)+1.
        The losses are averages over the epoch.
        todo add stddev?

        Plots:
        The left vertical axis is the scale of the loss.
        The right vertical axis is the scale of the learning rate.
        The x axis extent is max_iter, to give a visual signal when the "budget" was not consumed.


    Parameters:

        mltest: (optional :any:`MLTest`)
            Incorporate test performance in the loss curves
            (in the usual way, giving the typical "train" and "test"
            loss curves).

            .. warning::

                ``mltest`` not been implemented ITCINOOD.

    """

    # todo optional .npz format saving for larger jobs.
    #  Consider an automated decision algorithm for saving format.
    #  Perhaps, npz when maxiter is large enough.

    #  todo: after implementing MLtest, implement this.

    def __init__(self, every=1, mltest=None):
        super().__init__()

        self.epoch = 0
        self.every = every
        self.iter = 0

        # The per-iteration losses
        self.X = None
        # The per-epoch losses (average) and kit
        self.Y = None
        # The minimum total loss during last call to train(), during each step
        self.Z = None
        # The sum of the # of constraint losses and the # of ic losses (phase dependent)
        self.MN = None

        # the iteration of the previous checkpoint (0, 10, 30, 60, 90)
        self.checkpoint_iteration = 0
        self.checkpoint_i = 0

        self.mltest = mltest
        if mltest is not None:
            self.log(f"[Info] MLTest incorporation in LossCurves is not implemented.")


    def gate_strideloop(self, B):
        self.Z = np.full(
            shape=[
                B.problem.th.Nstep(),
                2,
            ],
            fill_value=-1.0
        ).astype(get_dtype(B.driver.config.fw_type))


    def gate_iterloop(self, B, BB):
        self.iter = 0
        self.epoch = 0
        self.checkpoint_i = 0
        # MN = M+N
        self.MN = len(BB.ic_losses) + len(BB.losses)
        maxit = B.phase.strategies.optimizer.kit.max_iterations
        # allocate memory
        self.X = np.full(
            shape=[maxit,
               len(BB.ic_losses) + len(BB.losses)
               + 1
            ],
            fill_value=-1.0
        ).astype(get_dtype(B.driver.config.fw_type))
        # This is an over-allocation, but we're dry-running it
        # and don't wish to make a fuss until we're more experienced.
        # todo obtain minimum # of iters per epoch, and modify allocation.
        self.Y = np.full(
            shape=[
                # ceiling(maxit/min_iter_per_epoch),
                maxit,
                len(BB.ic_losses) + len(BB.losses)
                + 1
                + len(B.phase.strategies.optimizer.kit)
                + 1
            ],
            fill_value=-1.0
        ).astype(get_dtype(B.driver.config.fw_type))


    def after_iter(self, B, BB):
        self.iter += 1
        if self.iter == self.every:
            # todo still needs some coding to support self.every > 1.
            # write in line of data BB.losses
            beg = 0
            end = len(BB.ic_losses)
            self.X[BB.iteration,beg:end] = BB.ic_losses
            beg = end
            end += len(BB.losses)
            self.X[BB.iteration,beg:end] = BB.losses
            self.iter = 0


    def on_end_of_epoch(self, B, BB):
        beg = len(BB.ic_losses) + len(BB.losses) + 1
        end = beg + len(B.phase.strategies.optimizer.kit)
        self.Y[self.epoch:self.epoch+1,beg:end] = B.phase.strategies.optimizer.kit.as_np(
            dtype=B.problem.background.dtype(),
        )
        beg = end
        end += 1
        # record iteration
        self.Y[self.epoch:self.epoch+1, beg:end] = BB.iteration
        # push counter
        self.epoch += 1


    def on_checkpoint(self, B, BB):
        begX = self.checkpoint_iteration
        endX = BB.iteration
        MN = self.MN
        # calculate X data
        self.X[begX:endX, MN] = np.sum(self.X[begX:endX, :MN], axis=1)

        self._store_X_data(B, BB, begX=begX, endX=endX)
        self._plot_data(B, BB, endX, endY=None)

        self.checkpoint_iteration = BB.iteration
        self.checkpoint_i += 1

    def after_iterloop(self, B, BB):
        begX = self.checkpoint_iteration
        endX = BB.iteration
        MN = self.MN
        # previous epoch's final iteration
        # iter_prev_epoch = int(self.Y[self.epoch-1, MN+1+len(B.phase.strategies.optimizer.kit)])
        # write incomplete epoch to Y
        self.on_end_of_epoch(B, BB)
        endY = self.epoch
        # the epoch counter is off by one
        # (probably not used again but just to be safe)
        self.epoch -= 1
        # calculate X data
        self.X[begX:endX, MN] = np.sum(self.X[begX:endX, :MN], axis=1)
        # calculate Y data
        iti = MN+1+len(B.phase.strategies.optimizer.kit)
        end = int(self.Y[0, iti])
        if end >= 0:
            # i = 0:
            self.Y[0,:MN+1] = np.sum(self.X[:end,:MN+1], axis=0)
            for i in range(1,endY):
                beg = int(self.Y[i-1, iti])
                end = int(self.Y[i, iti])
                # sum rows and store a line in Y
                self.Y[i,:MN+1] = np.sum(self.X[beg:end,:MN+1], axis=0)
            # finish computing averages
            # i = 0:
            self.Y[0,:MN+1] /= self.Y[0,iti]
            for i in range(1,endY):
                self.Y[i, :MN+1] /= self.Y[i,iti] - self.Y[i-1,iti]
        else:
            self.log("[Warn] No epoch during training")

        self._store_X_data(B, BB, begX = None, endX = endX)
        self._store_Y_data(B, BB, endY)
        self._plot_data(B, BB, endX, endY)


    def after_step(self, B):
        MN = self.MN
        # >> record the minimum observed total loss during this step.
        # Only the last call to train() during the step is considered.
        # > extract the minimum observed loss, averaged over epoch
        endY = self.epoch
        if endY > 0:
            # This minimum is over averages, so there should not
            # be a "noisy" minimum present.
            # Still, it is not advertised as a perfect statistic.
            L = np.min(self.Y[:endY,MN])
        else:
            L = -10.0
        tti = B.ti + B.tj
        self.Z[tti,:] = [tti,L]
        self.checkpoint_iteration = 0


    def after_strideloop(self, B):
        filename = self.cog.filename(
            action=self,
            handle="loss_vs_step",
            stem="dat",
            phasei=B.phasei,
        )
        np.savetxt(filename, X=self.Z)
        filename = self.cog.filename(
            action=self,
            handle="loss_vs_step",
            stem="fig",
            ending="png",
            phasei=B.phasei,
        )
        Lmax = np.max(self.Z[:,1])
        self.fig.series(
            filename=filename,
            title="average minimum loss vs. step",
            text=None,
            X=self.Z,
            inlabel="step",
            inidx=0,
            outlabels=["loss"],
            outidxs=[1],
            xlim=(self.Z[0,0], self.Z[-1,0]),
            ylim=(0.0, Lmax)
        )


    def _store_X_data(self, B, BB, begX, endX):
        handle0 = "loss_iter"
        if begX is not None:
            handle = B.checkpoint_tags[self.checkpoint_i] + handle0
        else:
            handle = handle0
        filename = self.cog.filename(
            action=self,
            handle=handle,
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
            tj=B.tj,
            level=B.L,
        )
        header = ""
        for label in B.problem.ic_constraints:
            header += f"ic:{label}, "
        for i, lb in enumerate(B.problem.constraints):
            if B.phase.active_constraint[lb]:
                header += f"c{i}, "
        header += "total"
        # todo use begX to append begX:endX to end of file, instead of writing new file.
        #  Can't use numpy convenience function, unfortunately.
        np.savetxt(filename, X=self.X[:endX,:], header=header)
        if self.checkpoint_i > 0:
            handle = B.checkpoint_tags[self.checkpoint_i-1] + handle0
            filename = self.cog.filename(
                action=self,
                handle=handle,
                phasei=B.phasei,
                stem="dat",
                ti=B.ti,
                tj=B.tj,
                level=B.L,
            )
            try:
                remove(filename)
            except:
                print(f"[LossCurves] File not found: {filename}")


    def _store_Y_data(self, B, BB, endY):
        filename = self.cog.filename(
            action=self,
            handle="loss_epoch",
            phasei=B.phasei,
            stem="dat",
            ti=B.ti,
            tj=B.tj,
            level=B.L,
        )
        header = ""
        for label in B.problem.ic_constraints:
            header += f"ic:{label}, "
        for i, lb in enumerate(B.problem.constraints):
            if B.phase.active_constraint[lb]:
                header += f"c{i}, "
        header += f"total, {B.phase.strategies.optimizer.kit.header()}, iter"
        np.savetxt(fname=filename, X=self.Y[:endY,:], header=header)


    def _plot_data(self, B, BB, endX, endY):
        handle0 = "loss"
        handle = handle0
        if B.problem.with_t:
            title = f"loss: ti {B.ti} tj {B.tj} ph {B.phasei} L {B.L} (t = {B.phase.samplesets.icbase.t:.4f})"
            ti = B.ti
            tj = B.tj
        else:
            title = f"loss: ph {B.phasei}"
            ti = None
            tj = None
        if endY is None:
            handle = B.checkpoint_tags[self.checkpoint_i] + handle
            title = f"{B.checkpoint_tags[self.checkpoint_i]}" + title
        filename = self.cog.filename(
            action=self,
            handle=handle,
            phasei=B.phasei,
            stem="fig",
            ending="png",
            ti=ti,
            tj=tj,
            level=B.L,
        )
        constraints = B.problem.constraints
        caption = ""
        for i, lb in enumerate(constraints):
            if B.phase.active_constraint[lb]:
                caption += f"c{i+1}:{lb} "
        self.fig.lrlosscurves_rev1(
            filename=filename,
            title=title,
            caption=caption,
            X=self.X,
            Y=self.Y,
            endX=endX,
            endY=endY,
            tolerance=B.phase.strategies.optimizer.kit.tolerance,
            ic_constraints=B.problem.ic_constraints,
        )
        if self.checkpoint_i > 0:
            handle = B.checkpoint_tags[self.checkpoint_i-1] + handle0
            filename = self.cog.filename(
                action=self,
                handle=handle,
                phasei=B.phasei,
                stem="fig",
                ending="png",
                ti=ti,
                tj=tj,
                level=B.L,
            )
            try:
                remove(filename)
            except:
                print(f"[LossCurves] File not found: {filename}")

