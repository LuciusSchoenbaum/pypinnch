





import os
import socket
from sys import platform
import random
from numpy import (
    float32,
    float64,
    random as np_random,
)



from mv1fw import (
    Logger,
)
from mv1fw.fw import (
    framework,
    fw_type,
    fw_cuda_is_available,
    fw_cuda_device_count,
    fw_cuda_get_device_name,
    fw_cuda_manual_seed_all,
    fw_set_default_dtype,
    fw_manual_seed,
    fw_backends_mps_is_available,
    fw_backends_mps_is_built,
)



class Background:
    """
    Background is a record class
    for background information about
    the system where the process is running,
    and other basic inputs to initialize (like random seeds).

    Parameters:

         system (string):
            The system (a name, such as hostname or a nickname)
            If none, hostname is used.
         clean (boolean):
            Whether to clean the output after each run.
            If false, each new run will generate a unique
            directory where the data is stored.
         backend (optional string):
            A user-requested backend such as "cuda", "cpu", "mps".
            If None, autodetect.
         precision (integer):
            precision, 32 or 64.
         seed (integer):
            a random seed in case you wish to have a
            reproducible run.
         file (string):
            Always `__file__`.

    """
    # todo experiment with precision levels

    def __init__(self,
        system,
        clean,
        backend,
        precision,
        seed,
        file,
    ):
        # initialize os's RNG (not used?)
        self.seed = seed
        if not (precision == 32 or precision == 64):
            raise ValueError(f"Precision must be either 32 or 64.")
        else:
            self.precision = precision
        if system is None or system == "":
            self.system = socket.gethostname()
        else:
            self.system = system
        # > file where the Topline and Background are defined, typ.`main.py`
        self.file = file
        self.clean = clean
        self.backend_requested = backend
        self.backend = None
        # > path for references, if any, set by `Engine`
        self.reference_absolute_root_directory = None


    def init(self, out = None):
        """
        Set up device(s), initialize framework.

        :meta private:
        """
        out = out if out is not None else Logger(level='quiet')
        # any platform-dependent operations
        if platform == "linux":
            pass
        elif platform == "darwin":
            pass
        elif platform == "win32" or platform == "cygwin":
            pass
        # > initialize random seed(s)
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)
        random.seed(self.seed)
        np_random.seed(self.seed)
        # > set up gpu and framework
        av = fw_cuda_is_available()
        dc = fw_cuda_device_count()
        # out.log(f"cuda is available: {av}")
        # out.log(f"cuda device count: {dc}")
        out.log(f"[Backend] {self.backend_requested} was requested.")
        if self.backend_requested == "cpu":
            self.backend = "cpu"
            out.log("[Backend] Using cpu...")
        elif self.backend_requested == "cuda":
            if av and dc > 0:
                # This can return an empty string, no need to log.
                # out.log("current device: ", torch.cuda.current_device())
                for i in range(dc):
                    out.log(f"[Backend] device {i} name: {fw_cuda_get_device_name(0)}")
                self.backend = "cuda"
                fw_cuda_manual_seed_all(self.seed)
                out.log("[Backend] Using cuda...")
            else:
                if not av:
                    out.log("[Backend] cuda is not available.")
                else:
                    out.log("[Backend] cuda is not available because no cuda devices were recognized.")
        elif self.backend_requested == "mps":
            # mps differs from cuda in our (somewhat limited) experience:
            # (1) it only supports float32,
            # (2) it does not support all of the same AD as cuda, ITCINOOD
            # Reason #2 prevents the use of mps for PINNs but it may be fixed
            # as frameworks are updated.
            if not fw_backends_mps_is_available():
                out.log("[Backend] Using mps backend...")
                self.backend = "mps"
                if self.precision == 64:
                    out.log("[Backend] The MPS framework doesn't support float64.")
                    self.precision= 32
                out.log("[Backend] Using mps...")
            else:
                if not fw_backends_mps_is_built():
                    out.log(f"MPS not available because the framework {framework} was not built with MPS enabled.")
                else:
                    out.log("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
        else:
            out.log(f"Unrecognized backend {self.backend_requested}")
        out.log(f"Using precision {self.precision}...")
        fw_set_default_dtype(fw_type(float64))
        fw_manual_seed(self.seed)
        out.log("\n\n")


    def dtype(self):
        out = float32 if self.precision == 32 else float64 if self.precision == 64 else None
        if out is None:
            raise ValueError(f"Unrecognized dtype for precision {self.precision}")
        return out


    def fw_type(self):
        """
        Return the framework data type, or ``fw_type``, corresponding
        to the dtype corresponding to the precision.

         Returns:
             framework data type

        :meta private:
        """
        out = fw_type(self.dtype())
        return out


    def __str__(self):
        out = ""
        out += f"system: {self.system}\n"
        out += f"backend: {self.backend_requested}\n"
        out += f"precision: {self.precision}\n"
        out += f"seed: {self.seed}\n"
        return out




