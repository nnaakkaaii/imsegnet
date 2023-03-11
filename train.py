import csv
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from imsegnet.datasets.moving_mnist import MovingMNIST
from imsegnet.models import models, options
from imsegnet.transforms.tensor_transforms.differentiate import Differentiate
from imsegnet.transforms.tensor_transforms.interface import \
    Interface as TransformInterface
from imsegnet.transforms.tensor_transforms.normalize import Normalize
from imsegnet.transforms.tensor_transforms.random_affine import RandomAffine
from imsegnet.transforms.tensor_transforms.random_crop import RandomCrop
from imsegnet.transforms.tensor_transforms.random_flip import RandomFlip


def run(opts: dict[str, Any],
        step_size: int,
        gamma: float,
        batch_size: int,
        num_epochs: int,
        save_dir: Path,
        model_name: str,
        use_differentiate: bool,
        use_random_crop: bool,
        use_random_flip: bool,
        use_random_affine: bool,
        debug: bool,
        **kwargs,
        ) -> float:
    id_ = str(uuid.uuid4())
    save_dir = save_dir / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + id_)

    os.makedirs(save_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert model_name in models
    opt = options[model_name].with_default(**kwargs)
    model = models[model_name](opt)
    model.load_state_dict(save_dir)
    model.to(device)

    train_tensor_transforms: list[TransformInterface] = []
    test_tensor_transforms: list[TransformInterface] = []
    if use_differentiate:
        train_tensor_transforms.append(Differentiate())
        test_tensor_transforms.append(Differentiate())
    if use_random_flip:
        train_tensor_transforms.append(RandomFlip())
    if use_random_crop:
        train_tensor_transforms.append(RandomCrop())
    if use_random_affine:
        train_tensor_transforms.append(RandomAffine())
    train_tensor_transforms.append(Normalize())
    test_tensor_transforms.append(Normalize())

    train_set = MovingMNIST(
        root_dir="./data",
        tensor_transforms=train_tensor_transforms,
        phase="train",
        limit=20 if debug else None,
        )
    test_set = MovingMNIST(
        root_dir="./data",
        tensor_transforms=test_tensor_transforms,
        phase="test",
        limit=20 if debug else None,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        step_size=step_size,
        gamma=gamma,
        )

    with open(save_dir / "option.json", "w") as f:
        json.dump(opts, f, indent=4)

    with open(save_dir / "result.csv", "w") as f:
        csv.writer(f).writerow([
            "dt",
            "epoch",
            "train_loss",
            "test_loss",
            "best_test_loss",
            ])

    best_test_loss = 1e8

    for epoch in range(1, num_epochs + 1):
        model.train()

        train_losses = []
        for d in tqdm(train_loader):
            model.forward(d["x"])
            loss = model.backward(d["t"])
            train_losses.append(loss)

        scheduler.step()

        model.eval()

        test_losses = []
        with torch.no_grad():
            for d in tqdm(test_loader):
                model.forward(d["x"])
                loss = model.loss(d["t"]).item()
                test_losses.append(loss)

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))

        best_test_loss = min(test_loss, best_test_loss)

        print(f"[Epoch {epoch:04}: "
              f"train loss {train_loss:.3f} / "
              f"test loss {test_loss:.3f} / "
              f"lr {scheduler.get_last_lr()[0]:.5f}")

        with open(save_dir / "result.csv", "a") as f:
            csv.writer(f).writerow([
                datetime.now(),
                epoch,
                train_loss,
                test_loss,
                best_test_loss,
                ])

        if best_test_loss == test_loss:
            model.save_state_dict(save_dir)

    return best_test_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size",
                        type=int,
                        default=10,
                        )
    parser.add_argument("--gamma",
                        type=float,
                        default=0.5,
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        )
    parser.add_argument("--num_epochs",
                        type=int,
                        default=50,
                        )
    parser.add_argument("--save_dir",
                        type=str,
                        default="./results",
                        )
    parser.add_argument("--model_name",
                        type=str,
                        default="simvp",
                        )
    parser.add_argument("--use_differentiate",
                        action="store_true",
                        )
    parser.add_argument("--use_random_crop",
                        action="store_true",
                        )
    parser.add_argument("--use_random_flip",
                        action="store_true",
                        )
    parser.add_argument("--use_random_affine",
                        action="store_true",
                        )
    parser.add_argument("--debug",
                        action="store_true",
                        )
    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        )
    parser.add_argument("--in_shape",
                        type=int,
                        nargs="+",
                        default=[10, 64, 64],
                        )
    parser.add_argument("--kernels",
                        type=int,
                        nargs="+",
                        default=[3, 5, 7, 11],
                        )
    args = parser.parse_args()

    run(vars(args),
        args.step_size,
        args.gamma,
        args.batch_size,
        args.num_epochs,
        Path(args.save_dir),
        args.model_name,
        args.use_differentiate,
        args.use_random_crop,
        args.use_random_flip,
        args.use_random_affine,
        args.debug,
        lr=args.lr,
        in_shape=tuple(args.in_shape),
        kernels=list(args.kernels),
        )
