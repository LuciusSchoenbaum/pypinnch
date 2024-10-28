# File manager.py Created by Lucius Schoenbaum June 23, 2023




import numpy as np
from os.path import (
    realpath as os_path_realpath,
)
from ...action import ActionBundle, ProbeBundle

from mv1fw import (
    CogManager,
    Logger,
    create_dir,
)
from mv1fw.fw import get_dtype
from mv1fw.visutil import Figure



class ActionManager:
    """
    The :any:`ActionManager` manages the output
    for the choice of :any:`Engine`, the main class of the
    training and simulation. All output passes through
    an :any:`Action`.

    More precisely, it hosts the :any:`Action` s and :any:`Probe` s
    (referring to "actions" generically means both actions and probes,
    as all probes are indeed actions),
    that are assigned by the user to :any:`Engine` s, :any:`Driver` s,
    and :any:`Phase` s. It provides these actions with the
    :any:`ActionBundle` and :any:`ProbeBundle` and keeps
    these "bundles" up-to-date as the simulation progresses, so that
    the actions have valid information.
    It also provides the :any:`Logger`, and the :any:`CogManager`
    that an action can use to perform its function.
    All of this is done in order to facilitate the use of actions or probes.

    Arguments:

        engine (:any:`Engine`):
            The main class of the training and simulation.
        output_abs_path_root (optional string):
            The absolute root path for building the directory tree
            for outputs. If None, a simple local directory "output" is used.

    """

    # todo review whether it is the engine who should invoke the manager,
    #  or whether it should be the driver. Indeed, do the engine
    #  and the drivers - separate processes in general - have separate
    #  logs, separate cogs, separate bundles? they should.
    #  So when a bundle is created (cf. __init__),
    #  it should perhaps receive a *driver*,
    #  not the engine. etc. etc.

    def __init__(
        self,
        engine,
        output_abs_path_root = None,
    ):
        if len(engine.drivers) > 1:
            # > raise error pending parallel driver update
            raise ValueError(f"Manager class does not support multiple drivers yet.")
        self.cog = CogManager(
            output_abs_path_root=output_abs_path_root,
        )
        # > logging
        self.log = Logger()
        self.fig = Figure()
        self.B = ActionBundle(
            engine=engine,
        )
        self.BB = ProbeBundle()


    def set_output_absolute_directory(self, abs_path):
        """
        Set the engine's output directory to
        an absolute path.

        Arguments:

            abs_path (string): absolute path

        """
        # > just re__init__
        self.cog = CogManager(
            output_abs_path_root = abs_path,
        )


    def set_output_relative_directory(self, rel_path):
        """
        Set the engine's output directory to
        an absolute path.
        The resulting path for outputs will be
        the "join" of abs_path and output_stem.

        Arguments:

            rel_path (string): absolute path

        """
        abs_path = os_path_realpath(rel_path)
        self.set_output_absolute_directory(
            abs_path = abs_path,
        )


    #<><><><><><><><><><><><><><><><>
    # Outer calls


    def on_start(self):
        """
        Initialize manager after engine's start(), "on start".
        It should be fine to call before or after engine.init(),
        as it only sets counters and references ITCINOOD.
        However, the call to default_timer suggests to call
        before init, i.e., at the very top of start(), because engine.init()
        is timed separately and this way subroutine timers are all
        nested within the overall timer using base_elapsed.

        """
        # > initialize bundle
        self.B.init()


    def after_init(self):
        """
        Initialize output artifacts after the initialization
        of program artifacts.
        """
        # > initialize cog, this step cleans the tree
        self.cog.init(
            engine=self.B.engine,
        )
        # > initialize all actions and probes
        for a in self.B.engine.actions:
            a.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=0)
        for p in self.B.engine.probes:
            p.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=0)
        for driver in self.B.engine.drivers:
            for a in driver.actions:
                a.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=1)
            for p in driver.probes:
                p.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=1)
            for plb in driver.phases:
                phase = driver.phases[plb]
                for a in phase.actions:
                    a.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=2)
                for p in phase.probes:
                    p.init(cog=self.cog, fig=self.fig, log=self.log, cog_id=2)


    #<><><><><><><><><><><><><><><><>
    # Output calls that trigger Action callbacks

    def gate_strideloop(self, driver):
        self.B.init_strideloop(
            driver=driver,
        )
        for a in self.B.engine.actions:
            a.gate_strideloop(self.B)
        for p in self.B.engine.probes:
            p.gate_strideloop(self.B)
        # todo this is incorrect code for multiple drivers -
        #  gate_strideloop should be called once separately for each driver -
        #  but review before changing - it's ok in one driver case.
        for a in driver.actions:
            a.gate_strideloop(self.B)
        for p in driver.probes:
            p.gate_strideloop(self.B)
        for plb in driver.phases:
            phase = driver.phases[plb]
            for a in phase.actions:
                a.gate_strideloop(self.B)
            for p in phase.probes:
                p.gate_strideloop(self.B)
        self.log(f"\n~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~\n")


    def on_stride(self, driver):
        self.log("\n--------------------------")
        self.log("--------------------------")
        self.log("--------------------------")
        self.log(f"STRIDE {self.B.stride}: ti = {self.B.ti}\n")
        self.B.init_stride(
            driver=driver,
        )
        self.cog.init_stride(
            driver=driver,
            ti=self.B.ti,
        )
        for a in self.B.engine.actions:
            a.on_stride(self.B)
        for p in self.B.engine.probes:
            p.on_stride(self.B)
        for driver in self.B.engine.drivers:
            for a in driver.actions:
                a.on_stride(self.B)
            for p in driver.probes:
                p.on_stride(self.B)
            for plb in driver.phases:
                phase = driver.phases[plb]
                for a in phase.actions:
                    a.on_stride(self.B)
                for p in phase.probes:
                    p.on_stride(self.B)


    def after_strideloop(self):
        for a in self.B.engine.actions:
            a.after_strideloop(self.B)
        for p in self.B.engine.probes:
            p.after_strideloop(self.B)
        for driver in self.B.engine.drivers:
            for a in driver.actions:
                a.after_strideloop(self.B)
            for p in driver.probes:
                p.after_strideloop(self.B)
            for plb in driver.phases:
                phase = driver.phases[plb]
                for a in phase.actions:
                    a.after_strideloop(self.B)
                for p in phase.probes:
                    p.after_strideloop(self.B)


    def on_end(self):
        self.log("\nEnd of run.\n")
        self.log("~-~-~-~-~-~\n")
        for a in self.B.engine.actions:
            a.on_end(self.B)
        for p in self.B.engine.probes:
            p.on_end(self.B)
        for driver in self.B.engine.drivers:
            for a in driver.actions:
                a.on_end(self.B)
            for p in driver.probes:
                p.on_end(self.B)
            for plb in driver.phases:
                phase = driver.phases[plb]
                for a in phase.actions:
                    a.on_end(self.B)
                for p in phase.probes:
                    p.on_end(self.B)


    def after_critical_section(self):
        for a in self.B.engine.actions:
            a.after_critical_section(self.B)
        for p in self.B.engine.probes:
            p.after_critical_section(self.B)
        for a in self.B.driver.actions:
            a.after_critical_section(self.B)
        for p in self.B.driver.probes:
            p.after_critical_section(self.B)
        for plb in self.B.driver.phases:
            phase = self.B.driver.phases[plb]
            for a in phase.actions:
                a.after_critical_section(self.B)
            for p in phase.probes:
                p.after_critical_section(self.B)


    def after_communication(self):
        for a in self.B.engine.actions:
            a.after_communication(self.B)
        for p in self.B.engine.probes:
            p.after_communication(self.B)
        for a in self.B.driver.actions:
            a.after_communication(self.B)
        for p in self.B.driver.probes:
            p.after_communication(self.B)
        for plb in self.B.driver.phases:
            phase = self.B.driver.phases[plb]
            for a in phase.actions:
                a.after_communication(self.B)
            for p in phase.probes:
                p.after_communication(self.B)


    def after_stride(self, ti):
        self.B.ti = ti
        for a in self.B.engine.actions:
            a.after_stride(self.B)
        for p in self.B.engine.probes:
            p.after_stride(self.B)
        for a in self.B.driver.actions:
            a.after_stride(self.B)
        for p in self.B.driver.probes:
            p.after_stride(self.B)
        for plb in self.B.driver.phases:
            phase = self.B.driver.phases[plb]
            for a in phase.actions:
                a.after_stride(self.B)
            for p in phase.probes:
                p.after_stride(self.B)
        self.log("End of stride.\n--------------\n\n")


    def gate_phaseloop(self):
        for a in self.B.engine.actions:
            a.gate_phaseloop(self.B)
        for p in self.B.engine.probes:
            p.gate_phaseloop(self.B)
        for a in self.B.driver.actions:
            a.gate_phaseloop(self.B)
        for p in self.B.driver.probes:
            p.gate_phaseloop(self.B)
        for plb in self.B.driver.phases:
            phase = self.B.driver.phases[plb]
            for a in phase.actions:
                a.gate_phaseloop(self.B)
            for p in phase.probes:
                p.gate_phaseloop(self.B)


    def on_phase(self, plb, phase):
        self.B.init_phase(
            phase=phase,
        )
        self.cog.init_phase(
            phase=phase,
            phi=self.B.phasei,
        )
        # > log
        handle = plb
        nstep = self.B.phase.th.Nstep()
        phi = self.B.phasei
        s = "s" if nstep > 1 else ""
        self.log(f"\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        nphase = len(self.B.driver.phases)
        if nphase > 1:
            self.log(f"{handle} Phase ({phi}/{nphase}, {nstep} step{s})\n")
        else:
            self.log(f"{handle} Phase ({nstep} step{s})\n")
        for a in self.B.engine.actions:
            a.on_phase(self.B)
        for p in self.B.engine.probes:
            p.on_phase(self.B)
        for a in self.B.driver.actions:
            a.on_phase(self.B)
        for p in self.B.driver.probes:
            p.on_phase(self.B)
        for a in self.B.phase.actions:
            a.on_phase(self.B)
        for p in self.B.phase.probes:
            p.on_phase(self.B)
        # after slice: the 0th slice (sj = 0), i.e. IC
        for a in self.B.engine.actions:
            a.after_slice(self.B)
        for p in self.B.engine.probes:
            p.after_slice(self.B)
        for a in self.B.driver.actions:
            a.after_slice(self.B)
        for p in self.B.driver.probes:
            p.after_slice(self.B)
        for a in self.B.phase.actions:
            a.after_slice(self.B)
        for p in self.B.phase.probes:
            p.after_slice(self.B)



    def gate_steploop(self):
        for a in self.B.engine.actions:
            a.gate_steploop(self.B)
        for p in self.B.engine.probes:
            p.gate_steploop(self.B)
        for a in self.B.driver.actions:
            a.gate_steploop(self.B)
        for p in self.B.driver.probes:
            p.gate_steploop(self.B)
        for a in self.B.phase.actions:
            a.gate_steploop(self.B)
        for p in self.B.phase.probes:
            p.gate_steploop(self.B)



    def on_step(self):
        # todo for nstep > ?? don't log each step?
        self.log(f"step ti {self.B.ti}, tj {self.B.tj}...")
        self.B.sj += 1
        for a in self.B.engine.actions:
            a.on_step(self.B)
        for p in self.B.engine.probes:
            p.on_step(self.B)
        for a in self.B.driver.actions:
            a.on_step(self.B)
        for p in self.B.driver.probes:
            p.on_step(self.B)
        for a in self.B.phase.actions:
            a.on_step(self.B)
        for p in self.B.phase.probes:
            p.on_step(self.B)


    def on_train(self):
        for a in self.B.engine.actions:
            a.on_train(self.B)
        for p in self.B.engine.probes:
            p.on_train(self.B)
        for a in self.B.driver.actions:
            a.on_train(self.B)
        for p in self.B.driver.probes:
            p.on_train(self.B)
        for a in self.B.phase.actions:
            a.on_train(self.B)
        for p in self.B.phase.probes:
            p.on_train(self.B)


    def after_train(self):
        for a in self.B.engine.actions:
            a.after_train(self.B)
        for p in self.B.engine.probes:
            p.after_train(self.B)
        for a in self.B.driver.actions:
            a.after_train(self.B)
        for p in self.B.driver.probes:
            p.after_train(self.B)
        for a in self.B.phase.actions:
            a.after_train(self.B)
        for p in self.B.phase.probes:
            p.after_train(self.B)
        self.B.tr += 1


    def after_step(self, passed):
        # todo instead of passing `passed` in after_step,
        #  write passed into the driver
        # expose the pass/fail status for this step to Actions
        self.B.passed = passed
        for a in self.B.engine.actions:
            a.after_step(self.B)
        for p in self.B.engine.probes:
            p.after_step(self.B)
        for a in self.B.driver.actions:
            a.after_step(self.B)
        for p in self.B.driver.probes:
            p.after_step(self.B)
        for a in self.B.phase.actions:
            a.after_step(self.B)
        for p in self.B.phase.probes:
            p.after_step(self.B)
        self.B.tj += 1
        for a in self.B.engine.actions:
            a.after_slice(self.B)
        for p in self.B.engine.probes:
            p.after_slice(self.B)
        for a in self.B.driver.actions:
            a.after_slice(self.B)
        for p in self.B.driver.probes:
            p.after_slice(self.B)
        for a in self.B.phase.actions:
            a.after_slice(self.B)
        for p in self.B.phase.probes:
            p.after_slice(self.B)


    def after_phase(self):
        for a in self.B.engine.actions:
            a.after_phase(self.B)
        for p in self.B.engine.probes:
            p.after_phase(self.B)
        for a in self.B.driver.actions:
            a.after_phase(self.B)
        for p in self.B.driver.probes:
            p.after_phase(self.B)
        for a in self.B.phase.actions:
            a.after_phase(self.B)
        for p in self.B.phase.probes:
            p.after_phase(self.B)
        self.log("End of phase.\n-=-=-=-=-=-=-\n\n")
        self.B.phase = None


    def after_phaseloop(self):
        for a in self.B.engine.actions:
            a.after_phaseloop(self.B)
        for p in self.B.engine.probes:
            p.after_phaseloop(self.B)
        for a in self.B.driver.actions:
            a.after_phaseloop(self.B)
        for p in self.B.driver.probes:
            p.after_phaseloop(self.B)
        for plb in self.B.driver.phases:
            phase = self.B.driver.phases[plb]
            for a in phase.actions:
                a.after_phaseloop(self.B)
            for p in phase.probes:
                p.after_phaseloop(self.B)


    # Expand/Contract/Advance


    def on_expand(self):
        self.B.L += 1
        for a in self.B.engine.actions:
            a.on_expand(self.B)
        for p in self.B.engine.probes:
            p.on_expand(self.B)
        for a in self.B.driver.actions:
            a.on_expand(self.B)
        for p in self.B.driver.probes:
            p.on_expand(self.B)
        for a in self.B.phase.actions:
            a.on_expand(self.B)
        for p in self.B.phase.probes:
            p.on_expand(self.B)


    def on_contract(self):
        self.B.L -= 1
        for a in self.B.engine.actions:
            a.on_contract(self.B)
        for p in self.B.engine.probes:
            p.on_contract(self.B)
        for a in self.B.driver.actions:
            a.on_contract(self.B)
        for p in self.B.driver.probes:
            p.on_contract(self.B)
        for a in self.B.phase.actions:
            a.on_contract(self.B)
        for p in self.B.phase.probes:
            p.on_contract(self.B)


    def on_advance(self):
        for a in self.B.engine.actions:
            a.on_advance(self.B)
        for p in self.B.engine.probes:
            p.on_advance(self.B)
        for a in self.B.driver.actions:
            a.on_advance(self.B)
        for p in self.B.driver.probes:
            p.on_advance(self.B)
        for a in self.B.phase.actions:
            a.on_advance(self.B)
        for p in self.B.phase.probes:
            p.on_advance(self.B)


    #<><><><><><><><><><><><><><><><>
    # Output calls that trigger Probe callbacks


    # Main Training Callbacks


    def gate_iterloop(self, kit, optimizer, lr_sched, active_csss):
        # > populate BB bundle
        dtype = get_dtype(self.B.driver.config.fw_type)
        self.BB.iteration = 0
        self.BB.kit = kit
        self.BB.optimizer = optimizer
        self.BB.lr_sched = lr_sched
        # todo should this be work of manager? should this be initialized elsewhere perhaps?
        self.BB.ic_losses = np.zeros([len(self.B.problem.ic_constraints),]).astype(dtype)
        self.BB.losses = np.zeros([len(active_csss),]).astype(dtype)
        # > call probes
        for p in self.B.engine.probes:
            p.gate_iterloop(self.B, self.BB)
        for p in self.B.driver.probes:
            p.gate_iterloop(self.B, self.BB)
        for p in self.B.phase.probes:
            p.gate_iterloop(self.B, self.BB)


    def on_iter(self):
        for p in self.B.engine.probes:
            p.on_iter(self.B, self.BB)
        for p in self.B.driver.probes:
            p.on_iter(self.B, self.BB)
        for p in self.B.phase.probes:
            p.on_iter(self.B, self.BB)


    def after_batch(self, hub):
        self.BB.XX = hub.XX
        self.BB.QQref = hub.QQref
        self.BB.XXs = hub.XXs
        self.BB.QQrefs = hub.QQrefs
        for p in self.B.engine.probes:
            p.after_batch(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_batch(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_batch(self.B, self.BB)


    def after_ic_loss(self, icci, loss):
        self.BB.ic_losses[icci] = loss
        for p in self.B.engine.probes:
            p.after_ic_loss(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_ic_loss(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_ic_loss(self.B, self.BB)


    def after_constraint_loss(self, ci, loss):
        self.BB.losses[ci] = loss
        for p in self.B.engine.probes:
            p.after_constraint_loss(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_constraint_loss(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_constraint_loss(self.B, self.BB)


    def after_residual(self, R, T, W):
        self.BB.R = R
        self.BB.T = T
        self.BB.W = W
        for p in self.B.engine.probes:
            p.after_residual(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_residual(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_residual(self.B, self.BB)
        # free/garbage collection
        self.BB.R = None
        self.BB.W = None
        self.BB.T = None


    def after_iter(self):
        it = self.BB.iteration
        # > get counters from info probe
        info = self.B.engine.probes[0]
        it_checkpoint_c = info.it_checkpoint_c
        it_checkpoint = info.it_checkpoint
        if it > 0 and it_checkpoint_c < len(it_checkpoint):
            if it_checkpoint[it_checkpoint_c] == it:
                # > call checkpoint callbacks
                self.on_checkpoint()
        for p in self.B.engine.probes:
            p.after_iter(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_iter(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_iter(self.B, self.BB)
        # free/garbage collection
        self.BB.XX = None
        self.BB.QQref = None
        self.BB.XXs = None
        self.BB.QQrefs = None
        self.BB.iteration += 1


    def on_end_of_epoch(self):
        for p in self.B.engine.probes:
            p.on_end_of_epoch(self.B, self.BB)
        for p in self.B.driver.probes:
            p.on_end_of_epoch(self.B, self.BB)
        for p in self.B.phase.probes:
            p.on_end_of_epoch(self.B, self.BB)


    def on_checkpoint(self):
        for p in self.B.engine.probes:
            p.on_checkpoint(self.B, self.BB)
        for p in self.B.driver.probes:
            p.on_checkpoint(self.B, self.BB)
        for p in self.B.phase.probes:
            p.on_checkpoint(self.B, self.BB)



    def after_iterloop(self):
        for p in self.B.engine.probes:
            p.after_iterloop(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_iterloop(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_iterloop(self.B, self.BB)
        self.log(f"Done with train at level {self.B.L}.")


    # End-of-Iteration Conditional Branches


    def after_lr_sched_step(self):
        for p in self.B.engine.probes:
            p.after_lr_sched_step(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_lr_sched_step(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_lr_sched_step(self.B, self.BB)


    def on_tolerance_break(self):
        self.log(f"train at level {self.B.L}: "
             f"break at iteration {self.BB.iteration} after reaching target "
             f"tolerance {self.B.phase.strategies.optimizer.kit.tolerance}. "
             f"({(self.BB.iteration/self.B.phase.strategies.optimizer.kit.max_iterations)*100:04f}% "
             f"of allocation)")
        for p in self.B.engine.probes:
            p.on_tolerance_break(self.B, self.BB)
        for p in self.B.driver.probes:
            p.on_tolerance_break(self.B, self.BB)
        for p in self.B.phase.probes:
            p.on_tolerance_break(self.B, self.BB)


    def on_maxiter_break(self):
        self.log(f"train at level {self.B.L}: time out after "
                 f"{self.B.phase.strategies.optimizer.kit.max_iterations} iterations.")
        for p in self.B.engine.probes:
            p.on_maxiter_break(self.B, self.BB)
        for p in self.B.driver.probes:
            p.on_maxiter_break(self.B, self.BB)
        for p in self.B.phase.probes:
            p.on_maxiter_break(self.B, self.BB)


    def action_triggered_break(self):
        return self.B.action_triggered_break


    def on_action_triggered_break(self):
        self.log(f"Quitting training early due to action trigger.")


    def after_taweighting_step(self):
        # self.log(f"TAWeighting step.")
        for p in self.B.engine.probes:
            p.after_taweighting_step(self.B, self.BB)
        for p in self.B.driver.probes:
            p.after_taweighting_step(self.B, self.BB)
        for p in self.B.phase.probes:
            p.after_taweighting_step(self.B, self.BB)


    # Other


    def after_problem_get(self, valuesin, varlist, values):
        if self.B.driver.during_phase:
            self.BB.valuesin = valuesin
            self.BB.varlist = varlist
            self.BB.values = values
            for p in self.B.engine.probes:
                p.after_problem_get(self.B, self.BB)
            for p in self.B.driver.probes:
                p.after_problem_get(self.B, self.BB)
            for p in self.B.phase.probes:
                p.after_problem_get(self.B, self.BB)
            self.BB.valuesin = None
            self.BB.varlist = None
            self.BB.values = None



