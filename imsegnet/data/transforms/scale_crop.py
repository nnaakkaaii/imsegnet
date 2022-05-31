import random

from PIL import ImageOps
from PIL.ImageFile import ImageFile
from PIL.Image import Resampling

from .base import BaseTransform


class ScaleCrop(BaseTransform):
    def __init__(self,
                 base_size: int,
                 crop_size: int,
                 fill: int = 0,
                 fixed: bool = False) -> None:
        super().__init__()
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.fixed = fixed

    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        if self.fixed:
            short_size = self.crop_size
        else:
            short_size = random.randint(
                self.base_size // 2,
                self.base_size * 2,
            )
        w, h = x.size
        if h > w:
            ow = short_size
            oh = h * ow // w
        else:
            oh = short_size
            ow = w * oh // h
        x = x.resize((ow, oh), Resampling.BILINEAR)
        t = t.resize((ow, oh), Resampling.NEAREST)
        if short_size < self.crop_size:
            pad_h = self.crop_size - oh if oh < self.crop_size else 0
            pad_w = self.crop_size - ow if ow < self.crop_size else 0
            x = ImageOps.expand(x, border=(0, 0, pad_w, pad_h), fill=0)
            t = ImageOps.expand(t, border=(0, 0, pad_w, pad_h), fill=self.fill)
        w, h = x.size
        if self.fixed:
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
        else:
            x1 = (w - self.crop_size) // 2
            y1 = (h - self.crop_size) // 2
        x = x.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        t = t.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return x, t
