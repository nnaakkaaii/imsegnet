from pathlib import Path, PosixPath

import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools import mask
from pycocotools.coco import COCO
from torchvision.transforms import Compose
from tqdm import trange
from PIL import Image, ImageFile

from ..transforms.base import BaseTransform

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCODataset(Dataset):
    n_classes = 21
    cat_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21,
                67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    def __init__(self,
                 base_dir: PosixPath = Path('inputs/raw/COCO/'),
                 transforms: tuple[BaseTransform] = (),
                 split: str = 'train',
                 year: str = '2017') -> None:
        super().__init__()
        ann_file = base_dir / f'annotations/instances_{split}{year}.json'
        ids_file = base_dir / f'annotations/{split}_ids_{year}.pth'
        self.img_dir = base_dir / f'images/{split}{year}'
        self.transform = Compose(transforms)
        self.split = split
        self.coco = COCO(ann_file)
        if ids_file.is_file():
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def _preprocess(self,
                    ids: list[str],
                    ids_file: PosixPath) -> list[str]:
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            coco_target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            msk = self._gen_seg_mask(coco_target,
                                     img_metadata['height'],
                                     img_metadata['width'])
            if (msk > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description(
                f'Doing: {i}/{len(ids)}, got {len(new_ids)} qualified images')
        print(f'Found number of qualified images: {len(new_ids)}')
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self,
                      target: list,
                      h: int,
                      w: int) -> np.ndarray:
        msk = np.zeros((h, w), dtype=np.uint8)
        for instance in target:
            rle = mask.frPyObjects(instance['segmentation'], h, w)
            m = mask.decode(rle)
            cat = instance['category_id']
            if cat not in self.cat_list:
                continue
            c = self.cat_list.index(cat)
            if len(m.shape) < 3:
                msk[:, :] += (msk == 0) * (m * c)
            else:
                msk[:, :] += (msk == 0) * ((np.sum(m, axis=2) > 0) * c).astype(np.uint8)
        return msk

    def __getitem__(self,
                    index: int) -> dict[str, ImageFile]:
        x, t = self._make_img_gt_point_pair(index)
        sample = {'x': x, 't': t}
        return self.transform(sample)

    def _make_img_gt_point_pair(
            self,
            index: int) -> tuple[ImageFile, ImageFile]:
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(self.img_dir / path).convert('RGB')
        coco_target = coco.loadAnns(coco.getAnnIds(img_id))
        target = Image.fromarray(self._gen_seg_mask(
            coco_target,
            img_metadata['height'],
            img_metadata['width']))
        return img, target

    def __len__(self) -> int:
        return len(self.ids)
