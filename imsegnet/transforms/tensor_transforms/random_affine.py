from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomAffine(Interface):
    DEGREE = 15
    TRANSLATE_X = 0.1
    TRANSLATE_Y = 0.1
    SCALE_MIN = 0.8
    SCALE_MAX = 1.2

    def __init__(
            self,
            degree: int | None = None,
            translate_x: float | None = None,
            translate_y: float | None = None,
            scale_min: float | None = None,
            scale_max: float | None = None,
    ) -> None:
        self.degree = [-self.DEGREE, self.DEGREE]
        self.translate = [self.TRANSLATE_X, self.TRANSLATE_Y]
        self.scale = [self.SCALE_MIN, self.SCALE_MAX]

        if degree is not None:
            self.degree = [-degree, degree]
        if translate_x is not None:
            self.translate[0] = translate_x
        if translate_y is not None:
            self.translate[1] = translate_y
        if scale_min is not None:
            self.scale[0] = scale_min
        if scale_max is not None:
            self.scale[1] = scale_max

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        _, w, h = x[0].size()
        p = transforms.RandomAffine.get_params(
            degrees=self.degree,
            translate=self.translate,
            scale_ranges=self.scale,
            shears=None,
            img_size=[w, h],
            )
        return [tf.affine(x_, *p) for x_ in x]


if __name__ == "__main__":
    # python3 -m imsegnet.transforms.tensor_transforms.random_affine
    from ...datasets.moving_mnist import MovingMNIST
    s = MovingMNIST(root_dir="./data",
                    tensor_transforms=[RandomAffine()],
                    )
    d, _ = next(iter(s))
    print(d.shape)
