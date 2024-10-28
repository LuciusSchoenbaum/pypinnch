


import torch

from .action_impl import Probe


class ModelCheckpoint(Probe):
    """
    This action saves the model (as PyTorch module)
    at a checkpoint (*), using a convenient
    `PyTorch format <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_.

    (*) Recall checkpoints are defined
    by the user, when the engine is created.

    .. note::

        Does not save the optimizer state, ITCINOOD.

    Parameters:

        lower_bound (integer):
            Lowest checkpoint, as percentage of training,
            to save the model. (Default: 50)
    """

    # todo more testing

    def __init__(self, lower_bound = 50):
        super().__init__()
        self.checkpoint_i = 0
        self.lower_bound = lower_bound


    def on_checkpoint(self, B, BB):
        if self.checkpoint_i < len(B.checkpoints) \
        and B.checkpoints[self.checkpoint_i] >= self.lower_bound:
            handle = B.checkpoint_tags[self.checkpoint_i]
            filename = self.cog.filename(
                action=self,
                handle=handle,
                phasei=B.phasei,
                stem="pth",
                ti=B.ti,
                tj=B.tj,
                level=B.L,
            )
            # _, fname = split_path(filename)
            # self.log( f"Saving model to {fname}")
            checkpoint = {
                "state_dict": B.modules[0].state_dict(),
                # todo? ...not of great concern yet.
                # "optimizer_state_dict": B.phase.strategies.optimizer.state_dict(),
                "iter_when_saved": BB.iteration,
            }
            torch.save(checkpoint, filename)
        self.checkpoint_i += 1




