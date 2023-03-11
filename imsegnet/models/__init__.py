from typing import Type

from .interface import Interface, Option
from .simvp import SimVPModel, SimVPOption

models: dict[str, Type[Interface]] = {
    "simvp": SimVPModel,
}

options: dict[str, Type[Option]] = {
    "simvp": SimVPOption,
}
