



from . import Probe

from sys import \
    stderr as sys_stderr

from os.path import \
    join as os_path_join, \
    basename as os_path_basename
from shutil import \
    copy as shutil_copy

from timeit import default_timer

from .._impl.types import \
    timingstore, \
    timingstore_filename_handle, \
    approximately

from numpy import \
    zeros as np_zeros, \
    savetxt as np_savetxt





class Info(Probe):
    """
    The default action, it cannot be disabled.
    Store basic information about the run.

    """

    # todo Info manages the timing store

    def __init__(self):
        super().__init__()
        # duration of algorithm/run
        self.total_time = None
        self.base_elapsed = None
        # duration of stride
        self.total_time_stride = 0.0
        # duration of phase
        self.total_time_phase = 0.0
        # > start the timer for measuring total time
        self.total_time = 0.0
        self.base_elapsed = default_timer()
        # checkpoints
        self.it_checkpoint = None
        self.it_checkpoint_c = 0
        self.it_base_time = None


    def gate_strideloop(self, B):
        path = self.cog.path(
            action=self,
        )
        # > store a copy of the main module
        if B.engine.background.file is not None:
            shutil_copy(
                B.engine.background.file,
                os_path_join(
                    path,
                    os_path_basename(B.engine.background.file)
                ),
            )
        # > store a copy of the engine card, if in a separate file
        if B.engine.file is not None:
            shutil_copy(
                B.engine.file,
                os_path_join(
                    path,
                    os_path_basename(B.engine.file)
                ),
            )
        # > store a copy of the problem card, if in a separate file
        if B.engine.problem.file is not None:
            shutil_copy(
                B.engine.problem.file,
                os_path_join(
                    path,
                    os_path_basename(B.engine.problem.file)
                ),
            )
        # > store a copy of the model card, if in a separate file
        if B.engine.models.file is not None:
            shutil_copy(
                B.engine.models.file,
                os_path_join(
                    path,
                    os_path_basename(B.engine.models.file)
                ),
            )
        # > emit information file
        information = "\n\n<><>information sheet<><>\n\n"
        information += f"Engine: {B.engine.__str__()}\n\n"
        information += "\n\n<><><><>end<><><><>\n\n"
        filename = self.cog.filename(
            action=self,
            handle="information",
            stem="txt",
        )
        with open(filename, 'w') as f:
            f.write(information)
        # > also add the information to the log,
        # it is in two places but that's fine.
        self.log(information)
        # > store a list of timesteps correlated with times
        # todo this is invalid/impossible if the timestep is adaptive, but
        #  in that you can add code to Info class.
        nstep = B.problem.th.Nstep()
        if nstep >= 1000:
            self.log(f"[Warning] nstep {nstep}, an unusually large number of steps.")
        if B.problem.with_t:
            # > timesteps file
            # todo review after .mv1 toml-based format
            filename = self.cog.filename(
                action=self,
                handle="timesteps",
                stem="dat",
            )
            timesteps = np_zeros((nstep+1, 2))
            t = B.problem.th.tinit
            for ti in range(nstep+1):
                timesteps[(ti,0)] = ti
                timesteps[(ti,1)] = t
                t = t + B.problem.th.stepsize()
            # noinspection PyTypeChecker
            np_savetxt(fname=filename, X=timesteps, header="ti, t")


    def after_init(self, B):
        # todo should this check be somewhere else - Backend?
        if not B.engine.background.backend:
            print(f"No backend found for requested backend "
                  f"{B.engine.background.backend_requested}. "
                  f"Exiting early.",
                  file=sys_stderr,
            )
            self.store_log()
            exit(0)


    def gate_iterloop(self, B, BB):
        self.cycle_total_time()
        self.it_base_time = self.total_time
        self.log(f"iter 0: 0% of iterations complete.\n  | total time: {self.total_time:.8f}, {approximately(self.total_time)}.")
        self.it_checkpoint = []
        self.it_checkpoint_c = 0
        for pct in B.checkpoints:
            self.it_checkpoint.append(int(BB.kit.max_iterations*(pct/100.0)))


    def on_checkpoint(self, B, BB):
        it = BB.iteration
        maxit = BB.kit.max_iterations
        self.cycle_total_time()
        el_time = self.total_time - self.it_base_time
        rem_time = (maxit - it)*el_time/it
        # total_time = self.total_time + rem_time
        self.log(f"iter {it}: {B.checkpoints[self.it_checkpoint_c]}% of iterations complete.")
        self.log(f"  | elapsed time: {el_time:.8f}, {approximately(el_time)}.")
        self.log(f"  | remaining time: {rem_time:.8f}, {approximately(rem_time)}.")
        self.log(f"  | total time: {self.total_time:.8f}, {approximately(self.total_time)}.")
        # > save the log, in case there is a crash
        # each time just overwrite the last saved log
        self.store_log()
        self.it_checkpoint_c += 1


    def on_end(self, B):
        # > collect the full start-to-finish clock measurement
        self.cycle_total_time()
        timingstore.rec["ttx"] = self.total_time
        # emit timing information
        # filename = used_to_be_(self.pasta.ttt())
        filename = self.cog.filename(
            action=self,
            handle=timingstore_filename_handle,
            stem="dat",
        )
        with open(filename, 'w') as f:
            f.write(str(timingstore))
        self.log(f"ttx {self.total_time:.8f}, {approximately(self.total_time)}\n\n\n")
        self.store_log()


    def store_log(self):
        filename = self.cog.filename(
            action=self,
            handle="log",
            stem="txt",
        )
        with open(filename, 'w') as f:
            f.write(str(self.log))


    def cycle_total_time(self):
        base_elapsed = default_timer()
        elapsed = base_elapsed - self.base_elapsed
        self.base_elapsed = base_elapsed
        self.total_time += elapsed
        return elapsed



