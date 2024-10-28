




from torch import (
    device as torch_device,
)


class DriverConfig:
    """
    Modifications to :any:`Background`
    when creating a :any:`Driver` instance,
    who might work on a single device (GPU) in parallel
    with other :any:`Driver` instances.

    .. note::
        At this point, you may wish to
        configure each driver to run on
        its respective node/environment.
        You can adjust the random seed,
        or adjust the device, by modifying
        the background of each driver here.

    Arguments:

        device_int (integer): a device id

    """

    def __init__(
            self,
            device_int = 0,
    ):
        self.device_int = device_int
        self.device = None
        self.fw_type = None

    def init(self, background):
        # read background, write into config
        if background.backend == "cpu":
            self.device = torch_device("cpu")
        elif background.backend == "cuda":
            self.device = torch_device(f"cuda:{self.device_int}")
        elif background.backend == "mps":
            self.device = torch_device("mps")
        else:
            self.device = None
            print(f"[Warn] Driver: backend {background.backend} not recognized.")
        self.fw_type = background.fw_type()

    def __str__(self):
        out = ""
        out += f"device: {self.device_int} ({self.device})\n"
        return out


