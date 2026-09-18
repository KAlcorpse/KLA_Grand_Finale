"""MiniRestormer — the submitted architecture and its checkpoint.

`best.pth` stores the architecture in its own weight shapes, so
`config_from_state_dict` rebuilds the right model from the checkpoint alone and
a checkpoint can never be silently loaded into the wrong network.
"""
from .minirestormer import (  # noqa: F401
    MiniRestormer,
    config_from_state_dict,
    load_state_dict_compat,
)
