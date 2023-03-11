import os
from pathlib import Path

import torch
from torch import nn, Tensor, optim
from torch.nn import DataParallel

from ..networks.simvp import SimVP
from .interface import Interface, Option


class SimVPOption(Option):
    in_shape: tuple[int, ...]
    in_channels: int
    inner_channels: int
    num_strides: int
    num_layers: int
    kernels: list[int]
    groups: int
    lr: float

    def set_default(self) -> None:
        self.in_channels = 16
        self.inner_channels = 256
        self.num_strides = 4
        self.num_layers = 8
        self.kernels = [3, 5, 7, 11]
        self.groups = 8

    @classmethod
    def with_default(cls, **kwargs) -> 'SimVPOption':
        o = cls()
        o.set_default()
        if "in_shape" in kwargs:
            o.in_shape = kwargs["in_shape"]
        else:
            raise ValueError("in_shape is required")
        if "in_channels" in kwargs:
            o.in_channels = kwargs["in_channels"]
        if "inner_channels" in kwargs:
            o.inner_channels = kwargs["inner_channels"]
        if "num_strides" in kwargs:
            o.num_strides = kwargs["num_strides"]
        if "num_layers" in kwargs:
            o.num_layers = kwargs["num_layers"]
        if "kernels" in kwargs:
            o.kernels = kwargs["kernels"]
        if "groups" in kwargs:
            o.groups = kwargs["groups"]
        if "lr" in kwargs:
            o.lr = kwargs["lr"]
        else:
            raise ValueError("lr is required")
        return o


class SimVPModel(Interface):
    filename = "net_simvp.pth"

    def __init__(self, opt: SimVPOption) -> None:
        super().__init__(opt)
        self.x = torch.zeros(0)
        self.y = torch.zeros(0)
        self.net = SimVP(
            in_shape=opt.in_shape,
            in_channels=opt.in_channels,
            inner_channels=opt.inner_channels,
            num_strides=opt.num_strides,
            num_layers=opt.num_layers,
            kernels=opt.kernels,
            groups=opt.groups,
            )
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr)
        self.criterion = nn.MSELoss()

    def to(self, device: str) -> None:
        if device == "cuda:0":
            self.net = DataParallel(self.net)
            torch.backends.cudnn.benchmark = True
        self.net.to(device)
        self.device = device

    def load_state_dict(self, save_dir: Path) -> None:
        net_path = save_dir / self.filename
        if os.path.isfile(net_path):
            self.net.load_state_dict(torch.load(net_path))

    def save_state_dict(self, save_dir: Path) -> None:
        net_path = save_dir / self.filename
        if isinstance(self.net, DataParallel):
            torch.save(self.net.module.state_dict(), net_path)
        else:
            torch.save(self.net.state_dict(), net_path)

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        self.x = x
        self.y = y
        return y

    def loss(self, t: Tensor) -> Tensor:
        return self.criterion(self.y, t.to(self.device))

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()
