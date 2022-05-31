import abc

import torch
import numpy as np
from PIL.ImageFile import ImageFile


class BaseTransform(metaclass=abc.ABCMeta):
    def pil_transform(
            self,
            x: ImageFile,
            t: ImageFile,
    ) -> tuple[ImageFile, ImageFile]:
        raise NotImplementedError

    def __call__(
            self,
            sample: dict[str, torch.Tensor | np.ndarray | ImageFile],
    ) -> dict[str, torch.Tensor | np.ndarray | ImageFile]:
        x = sample['x']
        t = sample['t']
        if isinstance(x, ImageFile) and isinstance(t, ImageFile):
            sample['x'], sample['t'] = self.pil_transform(x, t)
            return sample
        raise NotImplementedError
