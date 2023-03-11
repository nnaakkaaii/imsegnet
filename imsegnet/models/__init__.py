from typing import Dict, Type

from .interface import Interface, Option
from .simvp import SimVPModel, SimVPOption

models: Dict[str, Type[Interface]] = {
    "simvp": SimVPModel,
}

options: Dict[str, Type[Option]] = {
    "simvp": SimVPOption,
}
