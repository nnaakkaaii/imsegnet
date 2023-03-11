import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from imsegnet.datasets.moving_mnist import MovingMNIST
from imsegnet.models import models, options
from imsegnet.transforms.tensor_transforms.differentiate import Differentiate
from imsegnet.transforms.tensor_transforms.normalize import Normalize


def run(save_dir: Path,
        model_name: str,
        num_samples: int,
        ) -> None:
    os.makedirs(save_dir / "inference", exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert model_name in models
    assert os.path.isfile(save_dir / "option.json")

    with open(save_dir / "option.json", "r") as f:
        opt = json.load(f)
    option = options[model_name].with_default(**opt)
    model = models[model_name](option)

    model.load_state_dict(save_dir)
    model.to(device)

    differentiate = Differentiate()
    normalize = Normalize()

    test_set = MovingMNIST(
        root_dir="./data",
        tensor_transforms=[differentiate, normalize],
        phase="test",
        )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        )

    model.eval()
    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader)):
            if i % (len(test_loader) / num_samples) != 0:
                continue

            y = model.forward(d["x"]).cpu().detach().clone()

            x = [torch.unsqueeze(x_, dim=0) for x_ in d["x"][0]]
            t = [torch.unsqueeze(t_, dim=0) for t_ in d["t"][0]]
            y = [torch.unsqueeze(y_, dim=0) for y_ in y[0]]

            ys = x + y
            ys = normalize.backward(ys)
            ys = differentiate.backward(ys)

            ts = x + t
            ts = normalize.backward(ts)
            ts = differentiate.backward(ts)

            rs = torch.unsqueeze(torch.cat(ts + ys, dim=0), dim=1)
            img = make_grid(rs, nrow=20)
            to_pil_image(img).save(save_dir / "inference" / f"{i:03}.jpg")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        type=str,
                        default="results",
                        )
    parser.add_argument("--model_name",
                        type=str,
                        default="simvp",
                        )
    parser.add_argument("--num_samples",
                        type=int,
                        default=10,
                        )
    args = parser.parse_args()

    run(Path(args.save_dir),
        args.model_name,
        args.num_samples,
        )
