import random
from typing import List

from torch import Tensor
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomFlip(Interface):
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        if random.random() > 0.5:
            return [tf.hflip(x_) for x_ in x]
        return x


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.random_flip
    from ...datasets.moving_mnist import MovingMNIST
    s = MovingMNIST(root_dir="./data", tensor_transforms=[RandomFlip()])
    d = next(iter(s))
    print(d["x"].shape)
