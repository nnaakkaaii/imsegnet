from typing import List

from PIL.Image import Image, fromarray
from torch import Tensor

from ..interface import Interface


class ToPil(Interface):
    def forward(self, x: List[Tensor]) -> List[Image]:
        return [fromarray(x_.numpy(), mode="L") for x_ in x]
