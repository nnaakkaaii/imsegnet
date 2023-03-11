import torch
from torch import nn, Tensor

from .conv2d import ConvSC
from .inception import InceptionSC


def stride_generator(n: int, reverse: bool = False) -> list[int]:
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:n]))
    return strides[:n]


class SimVPEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inner_channels: int,
            num_strides: int,
    ) -> None:
        super().__init__()
        strides = stride_generator(num_strides)
        self.layer = ConvSC(
            in_channels,
            inner_channels,
            stride=strides[0],
            )
        self.layers = nn.ModuleList([
            ConvSC(
                inner_channels,
                inner_channels,
                stride=stride,
                )
            for stride in strides[1:]
            ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        enc = self.layer(x)
        latent = enc
        for layer in self.layers:
            latent = layer(latent)
        return latent, enc


class SimVPDecoder(nn.Module):
    def __init__(
            self,
            inner_channels: int,
            out_channels: int,
            num_strides: int,
    ) -> None:
        super().__init__()
        strides = stride_generator(num_strides, reverse=True)
        self.layers = nn.ModuleList([
            ConvSC(
                inner_channels,
                inner_channels,
                stride=stride,
                transpose=True,
                )
            for stride in strides[:-1]
            ])
        self.layer = ConvSC(
            2 * inner_channels,
            inner_channels,
            stride=strides[-1],
            transpose=True,
            )
        self.readout = nn.Conv2d(
            inner_channels,
            out_channels,
            1,
            )

    def forward(
            self,
            latent: Tensor,
            enc: Tensor | None = None,
    ) -> None:
        for layer in self.layers:
            latent = layer(latent)
        c = [latent]
        if enc is not None:
            c.append(enc)
        y = self.layer(torch.cat(c, dim=1))
        return self.readout(y)


class SimVPXnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inner_channels: int,
            num_layers: int,
            kernels: list[int],
            groups: int = 8,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            InceptionSC(
                in_channels,
                inner_channels // 2,
                inner_channels,
                kernels=kernels,
                groups=groups,
                ),
            *[
                InceptionSC(
                    inner_channels,
                    inner_channels // 2,
                    inner_channels,
                    kernels=kernels,
                    groups=groups,
                    )
                for _ in range(num_layers - 1)
                ],
            )
        self.decoder = nn.Sequential(
            InceptionSC(
                inner_channels,
                inner_channels // 2,
                inner_channels,
                kernels=kernels,
                groups=groups,
                ),
            *[
                InceptionSC(
                    2 * inner_channels,
                    inner_channels // 2,
                    inner_channels,
                    kernels=kernels,
                    groups=groups,
                    ),
                ],
            InceptionSC(
                2 * inner_channels,
                inner_channels // 2,
                in_channels,
                kernels=kernels,
                groups=groups,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b, t * c, h, w)

        # encoder
        skips: list[Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        skips.pop()
        skips.reverse()

        # decoder
        x = self.decoder[0](x)
        for s, layer in zip(skips, self.decoder[1:]):
            x = layer(torch.cat([x, s], dim=1))

        return x.reshape(b, t, c, h, w)


class SimVP(nn.Module):
    def __init__(
            self,
            in_shape: tuple[int, ...],
            in_channels: int,
            inner_channels: int,
            num_strides: int,
            num_layers: int,
            kernels: list[int],
            groups: int,
    ) -> None:
        super().__init__()

        t, _, _ = in_shape
        self.encoder = SimVPEncoder(
            1,
            in_channels,
            num_strides,
            )
        self.xnet = SimVPXnet(
            t * in_channels,
            inner_channels,
            num_layers,
            kernels,
            groups,
            )
        self.decoder = SimVPDecoder(
            in_channels,
            1,
            num_strides,
            )

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w = x.shape
        x = x.view(b * t, 1, h, w)

        latent, enc = self.encoder(x)

        l_shape = latent.shape[1:]
        latent = latent.view(b, t, *l_shape)
        latent = self.xnet(latent)
        latent = latent.reshape(b * t, *l_shape)

        y = self.decoder(latent, enc)
        return y.reshape(b, t, h, w)


if __name__ == "__main__":
    # python3 -m imsegnet.networks.simvp
    net = SimVP((10, 64, 64),
                in_channels=16,
                inner_channels=256,
                num_strides=4,
                num_layers=8,
                kernels=[3, 5],
                groups=8,
                )
    out = net(torch.randn(16, 10, 64, 64))
    print(out.shape)
