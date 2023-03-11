from PIL.Image import Image, fromarray
from torch import Tensor

from ..interface import Interface


class ToPil(Interface):
    def forward(self, x: list[Tensor]) -> list[Image]:
        return [fromarray(x_.numpy(), mode="L") for x_ in x]
