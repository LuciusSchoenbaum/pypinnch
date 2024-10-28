






from .phase_impl import (
    Phase,
    tolerance_finished,
)
from .._impl.types import timed

from mv1fw.fw import (
    fw_scalar
)




class StandardPINN(Phase):
    """
    Training parameters and networks to train,
    for a phase in a multi-phase training procedure.
    This uses a "standard" training procedure for PINNs.

    Parameters:

        batchsize (integer):
            batch size during training.
            The batch size set here is propagated to all sources.
            A batch size should be chosen depending on hardware.
            Bigger is usually better up to what hardware permits.
            Choice can influence the balance between
            performance and accuracy.

        SPL (integer):
            Samples per unit length, used to obtain SPM
            (samples per unit measure), e.g. for 3d problem
            SPM = SPL*SPL*SPL.

        constraints (optional list of string):
            List of constraints to train, if only a selection
            should be trained on. If left unspecified,
            all the constraints are considered.

        constraints_skip (optional list of string):
            List of constraints to pass over during training.
            It is possible to utilize both `constraints` and `constraints_skip`
            simultaneously, but it is more common to use one or the other to
            either make a selection or omit one or more constraints.

        step_multiple (integer):
            time info for multiphase training over a stride:
            the stepsize for a step in terms of the fundamental stepsize.
            Remind: the engine sets the
            stride and allocates the time horizon
            for the stride.

        shelf (scalar):
            Extent of sample in time past the next step.
            E.g., if current time is 4.0, stepsize is 0.5,
            if shelf is 0.1, then the time interval the model
            is trained on, during this step, is [4.0, 4.6].

        weights (dict, string -> float or special value):
            dict of weights. Special values can be exponential weight function
            or some other function of the current epoch.
            If float, then the weight is constant during training.

        strategies (:any:`Strategy` or list of :any:`Strategy`):
            Training strategies, list of strategies to apply during
            the phase.

        actions (:any:`Action` or list of :any:`Action`):
            Actions, list of actions or probes to perform
            during the phase.

    """

    def __init__(
            self,
            batchsize,
            SPL,
            constraints = None,
            constraints_skip = None,
            step_multiple = 1,
            shelf = 0.0,
            weights = None,
            strategies = None,
            actions = None,
    ):
        super().__init__(
            strategies=strategies,
            actions=actions,
            batchsize=batchsize,
            SPL=SPL,
            step_multiple=step_multiple,
            shelf=shelf,
            weights=weights,
            constraints=constraints,
            constraints_skip=constraints_skip,
        )
        self.tolerance_finished = False


    @timed("train")
    def train(
            self,
            level,
    ):
        """
        Main call to train neural network(s).

        .. note::
            We use a closure to call the optimizer,
            because this form is allowed for some optimizers,
            such as the Pytorch Adam optimizer,
            and required for other optimizers, like the Pytorch LBFGS optimizer.

        Arguments:

            level (integer):
                The current level, managed by the caller

        """
        # > set the return value, boolean pass/fail comparison to tolerance
        passed = False
        self.tolerance_finished = False
        # > get the module(s)
        if len(self.hub.modules) > 1:
            raise ValueError(f"{__class__.__name__}: multiple models not supported by the training procedure!")
        module = self.hub.modules[0]
        # > get the strategies and initialize them
        strats = self.strategies
        self.L = level
        if strats.using('grading'):
            strats.optimizer.kit = strats.grading.kits[self.L]
            if strats.optimizer.kit is None:
                raise ValueError(f"Require to set kit (max_iterations, tolerance, ...) via Grading object.")
        else:
            if strats.optimizer.kit is None:
                raise ValueError(f"Require to set kit (max_iterations, tolerance, ...) via Optimizer object.")
        for strat in strats:
            strat.init(phase=self)
        optimizer = strats.optimizer.get(
            level=self.L,
            module=module,
        )
        lr_sched = strats.lr_sched.get(
            level=self.L,
            optimizer=optimizer,
            kit=strats.optimizer.kit,
        )

        # > update moments
        self.momentsets.update(
            iteration=0,
            problem=self.problem,
        )

        self.out.gate_iterloop(
            kit=strats.optimizer.kit,
            optimizer=optimizer,
            lr_sched=lr_sched,
            active_csss=self.samplesets.active_csss,
        )

        iteration = 0
        while True:

            self.out.on_iter()

            self.batch()

            self.out.after_batch(hub=self.hub)

            def closure():
                optimizer.zero_grad()
                loss = fw_scalar(dtype=self.hub.fw_type, a=0.0).to(device=self.config.device)
                losses_ic, Lic = self.ic_loss()
                if strats.using('causalweighting'):
                    # todo
                    # losses_c, Lc = self.constraint_loss_causal()
                    raise NotImplementedError
                else:
                    losses_c, Lc = self.constraint_loss()
                # > check loss rel. tolerance
                self.tolerance_finished = tolerance_finished(
                    losses_ic=losses_ic,
                    losses_c=losses_c,
                    tolerance=strats.optimizer.kit.tolerance,
                )
                # > get loss-aware weights
                if strats.using('laweighting'):
                    lambdas_ic, lambdas_c = strats.laweighting.get(
                        losses_ic=losses_ic,
                        losses_c=losses_c,
                        epoch=self.samplesets.epoch(),
                    )
                else:
                    lambdas_ic, lambdas_c = len(losses_ic)*[1.0], len(losses_c)*[1.0]
                # > get tuning weights set directly/explicitly
                weights_ic, weights_c = self._get_weights(
                    iteration=iteration,
                )
                for i, L_ in enumerate(losses_c):
                    loss += lambdas_c[i]*weights_c[i]*L_
                for i, L_ in enumerate(losses_ic):
                    loss += lambdas_ic[i]*weights_ic[i]*L_
                if strats.using('taweighting'):
                    strats.taweighting.set_loss(Lic + Lc)
                loss.backward()
                return loss

            optimizer.step(closure)

            self.out.after_iter()

            # Break-checks and Epoch-checks
            # Impl note: always perform epoch-check *after* break-checks.
            taw_finished = strats.taweighting.finished() if strats.using('taweighting') else True
            if self.tolerance_finished and taw_finished:
                self.out.on_tolerance_break()
                passed = True
                break
            if iteration+1 == strats.optimizer.kit.max_iterations:
                self.out.on_maxiter_break()
                break
            if self.out.action_triggered_break():
                self.out.on_action_triggered_break()
                break
            if self.samplesets.end_of_epoch():
                # > callbacks triggered at end of epoch
                if strats.using('taweighting'):
                    # todo review
                    strats.taweighting.on_end_of_epoch()
                self.out.on_end_of_epoch()
                strats.lr_sched.step(
                    optimizer=optimizer,
                    phase=self,
                    iteration=None,
                )
                self.out.after_lr_sched_step()
            if strats.using('taweighting'):
                taw = strats.taweighting
                if taw.end_of_stage():
                    taw.step()
                    self.out.after_taweighting_step()
                    if not taw.gradual_mode():
                        # taw is not in gradual mode.
                        # We want to reinitialize the optimizer.
                        # Just create a new optimizer, it is not an expensive operation.
                        # The same applies to lr scheduler.
                        # todo review
                        optimizer = strats.optimizer.get(
                            level=self.L,
                            module=module,
                        )
                        # todo review - ?
                        lr_sched = strats.lr_sched.get(
                            level=self.L,
                            optimizer=optimizer,
                            kit=strats.optimizer.kit,
                        )
                    else:
                        # taw is in gradual mode,
                        # and the end_of_stage() signal is set.
                        pass
                else:
                    # Either TAW is already finished,
                    # or else TAW is proceeding and not at end of stage.
                    pass
            strats.lr_sched.step(
                optimizer=optimizer,
                phase=self,
                iteration=iteration,
            )
            iteration += 1
            self.hub.iteration += 1
            # > update moments
            self.momentsets.update(
                iteration=iteration,
                problem=self.problem,
            )
        #} // iter

        self.out.after_iterloop()

        return passed





