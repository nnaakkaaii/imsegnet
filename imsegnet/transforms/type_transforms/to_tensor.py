from typing import List

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from ..interface import Interface


class ToTensor(Interface):
    def forward(self, x: List[Image]) -> List[Tensor]:
        return [to_tensor(x_) for x_ in x]
