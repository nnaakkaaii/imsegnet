import torch
from torch import Tensor

from .interface import Interface


class Differentiate(Interface):
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        ys: list[Tensor] = []
        for x_ in x:
            y0 = torch.zeros_like(x_[0:1])
            y = x_[1:] - x_[:-1]
            ys.append(torch.cat([y0, y], dim=0))
        return ys


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.differentiate
    from ...datasets.moving_mnist import MovingMNIST
    s = MovingMNIST(root_dir="./data", tensor_transforms=[Differentiate()])
    d, _ = next(iter(s))
    print(d.shape)
