from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ..transforms.interface import Interface as TransformInterface
from ..transforms.tensor_transforms.interface import Interface as TensorTransformInterface
from ..transforms.pil_transforms.interface import Interface as PilTransformInterface
from ..transforms.type_transforms.to_tensor import ToTensor
from ..transforms.type_transforms.to_pil import ToPil


class Interface(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def transforms(
            pil_transforms: list[PilTransformInterface] | None = None,
            tensor_transforms: list[TensorTransformInterface] | None = None,
            ):
        to_pil: list[TransformInterface] = [ToPil()]

        _pil_transforms: list[PilTransformInterface] = []
        if pil_transforms is not None:
            _pil_transforms = pil_transforms

        to_tensor: list[TransformInterface] = [ToTensor()]

        _tensor_transforms: list[TensorTransformInterface] = []
        if tensor_transforms is not None:
            _tensor_transforms = tensor_transforms

        return Compose(to_pil + _pil_transforms + to_tensor + _tensor_transforms)

    @staticmethod
    def vanilla_transforms():
        return Compose([ToPil(), ToTensor()])
