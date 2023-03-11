import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from ..interface import Interface


class ToTensor(Interface):
    def forward(self, x: list[Image]) -> list[Tensor]:
        ret: list[Tensor] = []
        for x_ in x:
            y = to_tensor(x_)
            if y.dim() == 2:
                ret.append(torch.unsqueeze(y, dim=0))
            elif y.dim() == 3:
                ret.append(y)
            else:
                raise RuntimeError

        return ret
