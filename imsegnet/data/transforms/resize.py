from PIL.Image import Resampling
from PIL.ImageFile import ImageFile

from .base import BaseTransform


class FixedResize(BaseTransform):
    def __init__(self,
                 size: int) -> None:
        super().__init__()
        self.size = (size, size)

    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        assert x.size == t.size
        x = x.resize(self.size, Resampling.BILINEAR)
        t = t.resize(self.size, Resampling.NEAREST)
        return x, t
