import random

from PIL.ImageFile import ImageFile
from PIL.Image import Transpose

from .base import BaseTransform


class HorizontalFlip(BaseTransform):
    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        if random.random() < .5:
            x = x.transpose(Transpose.FLIP_LEFT_RIGHT)
            t = t.transpose(Transpose.FLIP_LEFT_RIGHT)
        return x, t
