from .gaussian_blur import GaussianBlur
from .horizontal_flip import HorizontalFlip
from .normalize import Normalize
from .resize import FixedResize
from .rotate import Rotate
from .scale_crop import ScaleCrop
from .to_tensor import ToTensor


transforms = {
    'gaussian_blur': GaussianBlur,
    'horizontal_flip': HorizontalFlip,
    'normalize': Normalize,
    'fixed_resize': FixedResize,
    'rotate': Rotate,
    'scale_crop': ScaleCrop,
    'to_tensor': ToTensor,
}


__all__ = [
    'GaussianBlur',
    'HorizontalFlip',
    'Normalize',
    'FixedResize',
    'Rotate',
    'ScaleCrop',
    'ToTensor',
]
