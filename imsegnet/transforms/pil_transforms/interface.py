from abc import abstractmethod
from typing import List

from PIL.Image import Image

from ..interface import Interface as TransformInterface


class Interface(TransformInterface):
    @abstractmethod
    def forward(self, x: List[Image]) -> List[Image]:
        pass

    def __call__(self, x: List[Image]) -> List[Image]:
        return self.forward(x)
