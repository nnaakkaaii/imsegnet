import os
from pathlib import Path, PosixPath

from PIL import Image
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ..transforms.base import BaseTransform


class PascalDataset(Dataset):
    n_classes = 21

    def __init__(self,
                 base_dir: PosixPath = Path('inputs/raw/Pascal/'),
                 transforms: tuple[BaseTransform] = (),
                 split: str = 'train') -> None:
        super().__init__()
        self.transform = Compose(transforms)

        image_dir = os.path.join(base_dir, 'JPEGImages')
        cat_dir = os.path.join(base_dir, 'SegmentationClass')

        _splits_dir = os.path.join(base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(image_dir, line + ".jpg")
            _cat = os.path.join(cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __getitem__(self,
                    index: int) -> dict[str, ImageFile]:
        x, t = self._make_img_gt_point_pair(index)
        sample = {'x': x, 't': t}
        return self.transform(sample)

    def _make_img_gt_point_pair(self,
                                index: int) -> tuple[ImageFile, ImageFile]:
        x = Image.open(self.images[index]).convert('RGB')
        t = Image.open(self.categories[index])
        return x, t

    def __len__(self) -> int:
        return len(self.images)
