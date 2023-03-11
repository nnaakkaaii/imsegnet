from abc import abstractmethod

from PIL.Image import Image

from ..interface import Interface as TransformInterface


class Interface(TransformInterface):
    @abstractmethod
    def forward(self, x: list[Image]) -> list[Image]:
        pass

    def __call__(self, x: list[Image]) -> list[Image]:
        return self.forward(x)
