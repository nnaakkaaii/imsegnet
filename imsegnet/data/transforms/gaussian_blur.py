import random

from PIL import ImageFile, ImageFilter

from .base import BaseTransform


class GaussianBlur(BaseTransform):
    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        if random.random() < .5:
            x = x.filter(ImageFilter.GaussianBlur(random.random()))
        return x, t
