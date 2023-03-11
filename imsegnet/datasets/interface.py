from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ..transforms.interface import Interface as TransformInterface
from ..transforms.pil_transforms.interface import \
    Interface as PilTransformInterface
from ..transforms.tensor_transforms.interface import \
    Interface as TensorTransformInterface
from ..transforms.type_transforms.to_pil import ToPil
from ..transforms.type_transforms.to_tensor import ToTensor


class Interface(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def transforms(
            pil_transforms: Optional[List[PilTransformInterface]] = None,
            tensor_transforms: Optional[List[TensorTransformInterface]] = None,
            ):
        to_pil: List[TransformInterface] = [ToPil()]

        _pil_transforms: List[PilTransformInterface] = []
        if pil_transforms is not None:
            _pil_transforms = pil_transforms

        to_tensor: List[TransformInterface] = [ToTensor()]

        _tensor_transforms: List[TensorTransformInterface] = []
        if tensor_transforms is not None:
            _tensor_transforms = tensor_transforms

        return Compose(to_pil + _pil_transforms + to_tensor + _tensor_transforms)

    @staticmethod
    def vanilla_transforms():
        return Compose([ToPil(), ToTensor()])
