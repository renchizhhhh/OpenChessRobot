import numpy as np
import torch
import typing
import functools
from collections.abc import Iterable
from pathlib import Path
import os
from utili.recap.path_manager import register_translator

_DATA_DIR = Path(os.getenv("DATA_DIR",
                           Path(__file__).parent.parent.parent / "data"))
_CONFIG_DIR = Path(os.getenv("CONFIG_DIR",
                             Path(__file__).parent.parent / "config"))
_RUNS_DIR = Path(os.getenv("RUNS_DIR",
                           Path(__file__).parent.parent / "runs"))
_RESULTS_DIR = Path(os.getenv("RESULTS_DIR",
                              Path(__file__).parent.parent.parent / "results"))
_MODELS_DIR = Path(os.getenv("MODELS_DIR",
                             Path(__file__).parent.parent.parent / "models"))
_REPORT_DIR = Path(os.getenv("REPORT_DIR",
                             Path(__file__).parent.parent.parent.parent / "chess-recognition-report"))

register_translator("data", _DATA_DIR)
register_translator("config", _CONFIG_DIR)
register_translator("runs", _RUNS_DIR)
register_translator("results", _RESULTS_DIR)
register_translator("models", _MODELS_DIR)
register_translator("report", _REPORT_DIR)

#: Device to be used for computation (GPU if available, else CPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]

def device(x: T, dev: str = DEVICE) -> T:
    """Convenience method to move a tensor/module/other structure containing tensors to the device.

    Args:
        x (T): the tensor (or strucure containing tensors)
        dev (str, optional): the device to move the tensor to. Defaults to DEVICE.

    Raises:
        TypeError: if the type was not a compatible tensor

    Returns:
        T: the input tensor moved to the device
    """

    to = functools.partial(device, dev=dev)
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return map(to, x)
    else:
        raise TypeError


def listify(func: typing.Callable[..., typing.Iterable]) -> typing.Callable[..., typing.List]:
    """Decorator to convert the output of a generator function to a list.

    Args:
        func (typing.Callable[..., typing.Iterable]): the function to be decorated

    Returns:
        typing.Callable[..., typing.List]: the decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper
