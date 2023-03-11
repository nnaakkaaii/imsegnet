from abc import abstractmethod

from torch import Tensor

from ..interface import Interface as TransformInterface


class Interface(TransformInterface):
    @abstractmethod
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        pass

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        return self.forward(x)
