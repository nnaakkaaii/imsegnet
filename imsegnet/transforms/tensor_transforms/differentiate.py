from typing import List

import torch
from torch import Tensor

from .interface import Interface


class Differentiate(Interface):
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        y = torch.zeros_like(x[0])
        ys: List[Tensor] = []
        for x_ in x:
            ys.append(x_ - y)
            y = x_
        return ys

    def backward(self, x: List[Tensor]) -> List[Tensor]:
        y = torch.zeros_like(x[0])
        ys: List[Tensor] = []
        for x_ in x:
            ys.append(x_ + y)
            y = x_ + y
        return ys


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.differentiate
    from ...datasets.moving_mnist import MovingMNIST
    s = MovingMNIST(root_dir="./data", tensor_transforms=[Differentiate()])
    d = next(iter(s))
    print(d["x"].shape)

    x1 = [torch.unsqueeze(x, dim=0) for x in d["x"]]
    t1 = [torch.unsqueeze(t, dim=0) for t in d["t"]]

    s = MovingMNIST(root_dir="./data")
    d = next(iter(s))

    x2 = [torch.unsqueeze(x, dim=0) for x in d["x"]]
    t2 = [torch.unsqueeze(t, dim=0) for t in d["t"]]

    xt1 = x2 + t2
    xt2 = Differentiate().forward(xt1)
    xt3 = Differentiate().backward(xt2)

    print((torch.cat(xt2, dim=0) - torch.cat(x1 + t1, dim=0)).sum())
    print((torch.cat(xt3, dim=0) - torch.cat(xt1, dim=0)).sum())
