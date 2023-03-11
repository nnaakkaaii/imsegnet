from abc import ABCMeta, abstractmethod
from typing import Any


class Interface(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: list[Any]) -> list[Any]:
        pass

    def __call__(self, x: list[Any]) -> list[Any]:
        return self.forward(x)
