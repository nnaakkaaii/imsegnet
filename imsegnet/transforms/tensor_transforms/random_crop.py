from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomCrop(Interface):
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        _, w, h = x[0].size()
        assert w == h
        p = transforms.RandomCrop.get_params(x[0], output_size=(w, h))
        return [tf.crop(x_, *p) for x_ in x]


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.random_crop
    from ...datasets.moving_mnist import MovingMNIST
    s = MovingMNIST(root_dir="./data", tensor_transforms=[RandomCrop()])
    d = next(iter(s))
    print(d["x"].shape)
