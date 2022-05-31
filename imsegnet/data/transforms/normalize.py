import torch

import numpy as np
from PIL.ImageFile import ImageFile

from .base import BaseTransform


class Normalize(BaseTransform):
    def __init__(
            self,
            mean: tuple[float, float, float],
            std: tuple[float, float, float],
    ) -> None:
        self.mean = mean
        self.std = std

    def np_transform(
            self,
            x: np.ndarray,
            t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x.astype(np.float32)
        t = t.astype(np.float32)
        x /= 255.
        x -= self.mean
        x /= self.std
        return x, t

    def __call__(
            self,
            sample: dict[str, torch.Tensor | np.ndarray | ImageFile],
    ) -> dict[str, torch.Tensor | np.ndarray | ImageFile]:
        sample['x'], sample['t'] = self.np_transform(sample['x'], sample['t'])
        return sample
