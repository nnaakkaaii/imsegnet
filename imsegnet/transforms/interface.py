from abc import ABCMeta, abstractmethod
from typing import Any, List


class Interface(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: List[Any]) -> List[Any]:
        pass

    def __call__(self, x: List[Any]) -> List[Any]:
        return self.forward(x)
