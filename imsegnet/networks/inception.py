from torch import nn, Tensor

from .conv2d import GroupConv2d


class Inception(nn.Module):
    def __init__(self, *args: nn.Module) -> None:
        super().__init__()
        self.layers = nn.Sequential(*args)

    def forward(self, x: Tensor) -> Tensor:
        return sum(layer(x) for layer in self.layers)


class InceptionSC(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inner_channels: int,
            out_channels: int,
            kernels: list[int],
            groups: int = 8,
    ) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                ),
            Inception(
                *[GroupConv2d(
                    inner_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    groups=groups,
                    act_norm=True,
                ) for kernel in kernels]
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
