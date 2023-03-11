import errno
import os

import numpy as np
import torch
from torch import Tensor

from ..transforms.pil_transforms.interface import Interface as PilTransformInterface
from ..transforms.tensor_transforms.interface import Interface as TensorTransformInterface
from .interface import Interface as DatasetInterface


class MovingMNIST(DatasetInterface):
    PHASES = {"train", "test"}
    URL = "https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz"
    RAW_FOLDER = "raw"
    PROCESSED_FOLDER = "processed"
    TRAIN_FILE = "moving_mnist_train.pt"
    TEST_FILE = "moving_mnist_test.pt"

    def __init__(self,
                 root_dir: str,
                 split: int = 1000,
                 pil_transforms: list[PilTransformInterface] | None = None,
                 tensor_transforms: list[TensorTransformInterface] | None = None,
                 phase: str = "train",
                 download: bool = False,
                 limit: int | None = None,
                 ) -> None:
        assert phase in self.PHASES
        super().__init__()

        self.transform = self.transforms(pil_transforms, tensor_transforms)
        self.vanilla_transform = self.vanilla_transforms()
        self.limit = limit

        if download:
            self.__download(root_dir, split)

        if not self.__check_existence(root_dir):
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if phase == "train":
            file = self.TRAIN_FILE
        elif phase == "test":
            file = self.TEST_FILE
        else:
            raise RuntimeError("phase not supported")

        self.data = torch.load(os.path.join(root_dir, self.PROCESSED_FOLDER, file))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        def _transform_time(data: Tensor) -> Tensor:
            new_data = self.transform(list(data))
            return torch.cat(new_data, dim=0)

        ret = _transform_time(self.data[index])
        return {
            "x": ret[:10],
            "t": ret[10:],
        }

    def __len__(self) -> int:
        if self.limit is None:
            return len(self.data)
        return min(self.limit, len(self.data))

    def __check_existence(self, root_dir: str) -> bool:
        return (
            os.path.exists(os.path.join(root_dir, self.PROCESSED_FOLDER, self.TRAIN_FILE))
            and os.path.exists(os.path.join(root_dir, self.PROCESSED_FOLDER, self.TEST_FILE))
        )

    def __download(self, root_dir: str, split: int):
        import gzip
        from urllib.request import urlopen

        if self.__check_existence(root_dir):
            return

        try:
            os.makedirs(os.path.join(root_dir, self.RAW_FOLDER))
            os.makedirs(os.path.join(root_dir, self.PROCESSED_FOLDER))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        print("Downloading " + self.URL)
        data = urlopen(self.URL)
        filename = self.URL.rpartition("/")[2]
        file_path = os.path.join(root_dir, self.RAW_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(data.read())
        with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

        print("Processing...")

        train_set = torch.from_numpy(np.load(os.path.join(root_dir, self.RAW_FOLDER, "mnist_test_seq.npy")).swapaxes(0, 1)[:-split])
        test_set = torch.from_numpy(np.load(os.path.join(root_dir, self.RAW_FOLDER, "mnist_test_seq.npy")).swapaxes(0, 1)[-split:])

        with open(os.path.join(root_dir, self.PROCESSED_FOLDER, self.TRAIN_FILE), "wb") as f:
            torch.save(train_set, f)
        with open(os.path.join(root_dir, self.PROCESSED_FOLDER, self.TEST_FILE), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


if __name__ == "__main__":
    # python3 -m imsegnet.datasets.moving_mnist
    dataset = MovingMNIST(root_dir="./data",
                          phase="train",
                          download=True)
    print("Dataset length: %d" % len(dataset))
    dt = dataset[0]
    print("Dataset size: ", dt["x"].size(), dt["t"].size())
