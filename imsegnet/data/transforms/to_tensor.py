import torch
import numpy as np
from PIL.ImageFile import ImageFile

from .base import BaseTransform


class ToTensor(BaseTransform):
    def numpy2tensor(
            self,
            x: np.ndarray,
            t: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x = x.astype(np.float32).transpose((2, 0, 1))
        t = t.astype(np.float32)
        x_tensor = torch.from_numpy(x).float()
        t_tensor = torch.from_numpy(t).float()
        return x_tensor, t_tensor

    def __call__(
            self,
            sample: dict[str, torch.Tensor | np.ndarray | ImageFile],
    ) -> dict[str, torch.Tensor | np.ndarray | ImageFile]:
        sample['x'], sample['t'] = self.numpy2tensor(sample['x'], sample['t'])
        return sample
