from abc import abstractmethod
from typing import List

from torch import Tensor

from ..interface import Interface as TransformInterface


class Interface(TransformInterface):
    @abstractmethod
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        pass

    def __call__(self, x: List[Tensor]) -> List[Tensor]:
        return self.forward(x)
