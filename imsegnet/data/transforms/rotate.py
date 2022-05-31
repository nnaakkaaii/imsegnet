import random

from PIL.ImageFile import ImageFile
from PIL.Image import Resampling

from .base import BaseTransform


class Rotate(BaseTransform):
    def __init__(self,
                 degree: int) -> None:
        super().__init__()
        self.degree = degree

    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        degree = random.uniform(-self.degree, self.degree)
        x = x.rotate(degree, Resampling.BILINEAR)
        t = t.rotate(degree, Resampling.NEAREST)
        return x, t
