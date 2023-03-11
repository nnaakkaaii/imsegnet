from typing import List

from torch import Tensor
from torchvision import transforms

from .interface import Interface


class Normalize(Interface):
    MEAN = (0.5,)
    STD = (0.5,)

    def __init__(self):
        self.normalize = transforms.Normalize(self.MEAN, self.STD)
        self.unnormalize = transforms.Compose([
            transforms.Normalize((0,), tuple(1/i for i in self.STD)),
            transforms.Normalize(tuple(-i for i in self.MEAN), (1,)),
            ])

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        return [self.normalize(_x) for _x in x]

    def backward(self, x: List[Tensor]) -> List[Tensor]:
        return [self.unnormalize(_x) for _x in x]


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.normalize
    from ...datasets.moving_mnist import MovingMNIST

    s = MovingMNIST(root_dir="./data",
                    pil_transforms=[],
                    tensor_transforms=[],
                    )
    d = next(iter(s))
    print(d["x"].shape)
    print(d["x"].min(), d["x"].max())
    s = MovingMNIST(root_dir="./data",
                    pil_transforms=[],
                    tensor_transforms=[Normalize()],
                    )
    d = next(iter(s))
    print(d["x"].shape)
    print(d["x"].min(), d["x"].max())
