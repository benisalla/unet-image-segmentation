"""The model family: one configurable encoder-decoder and the fully connected baseline.

Shapes for a 96x96 RGB input, `base=16`, `depth=4`:

    stem        3 x 96 x 96  -> 16 x 96 x 96
    down 1..4   16x96x96 -> 32x48x48 -> 64x24x24 -> 128x12x12 -> 256x6x6   (bottleneck)
    up 4..1     256x6x6 -> 128x12x12 -> 64x24x24 -> 32x48x48 -> 16x96x96
    head        16 x 96 x 96 -> 1 x 96 x 96   (one logit per pixel)

With skip connections each up step concatenates the encoder feature map of the same
resolution before its convolutions (U-Net); without them the decoder only sees the
bottleneck (SegNet-like encoder-decoder, minus SegNet's pooling indices).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unetseg.config import ModelConfig


def conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvBlock(nn.Module):
    """One (`double=False`) or two (`double=True`) 3x3 conv + BN + ReLU.

    Two stacked 3x3 convs see a 5x5 window with fewer weights than one 5x5 conv and
    with an extra non-linearity in between. With `residual=True` the input is added
    back (through a 1x1 conv when the channel count changes): y = relu(F(x) + Wx).
    """

    def __init__(self, in_ch: int, out_ch: int, double: bool = True, residual: bool = False):
        super().__init__()
        last_in = in_ch
        layers: list[nn.Module] = []
        if double:
            layers.append(conv_bn_relu(in_ch, out_ch))
            last_in = out_ch
        # the last conv has no ReLU yet: it is applied after the (optional) residual add
        layers += [nn.Conv2d(last_in, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch)]
        self.body = nn.Sequential(*layers)
        self.residual = residual
        if residual:
            self.shortcut = (
                nn.Identity()
                if in_ch == out_ch
                else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.residual:
            y = y + self.shortcut(x)
        return F.relu(y, inplace=True)


class Down(nn.Module):
    """Halve the resolution, then a ConvBlock.

    `maxpool`: a fixed 2x2 max (no weights, keeps the strongest response).
    `strided`: a learned 3x3 conv with stride 2 (+BN+ReLU), more weights.
    """

    def __init__(self, in_ch: int, out_ch: int, down: str, double: bool, residual: bool):
        super().__init__()
        if down == "maxpool":
            self.reduce = nn.MaxPool2d(2)
            self.block = ConvBlock(in_ch, out_ch, double, residual)
        elif down == "strided":
            self.reduce = conv_bn_relu(in_ch, in_ch, stride=2)
            self.block = ConvBlock(in_ch, out_ch, double, residual)
        else:
            raise ValueError(f"down must be 'maxpool' or 'strided', got {down!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.reduce(x))


class Up(nn.Module):
    """Transposed conv x2, optionally concatenate the skip, then a ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, skip: bool, double: bool, residual: bool):
        super().__init__()
        self.skip = skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(out_ch * 2 if skip else out_ch, out_ch, double, residual)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if self.skip:
            if skip is None:
                raise ValueError("this Up block expects a skip tensor")
            dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
            if dh or dw:  # odd input sizes: pad so the two maps line up
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([skip, x], dim=1)
        return self.block(x)


class EncoderDecoder(nn.Module):
    """U-Net when `skip=True`, SegNet-like encoder-decoder when `skip=False`."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.base * 2**i for i in range(cfg.depth + 1)]
        self.stem = ConvBlock(cfg.in_ch, chs[0], cfg.double, cfg.residual)
        self.downs = nn.ModuleList(
            Down(chs[i], chs[i + 1], cfg.down, cfg.double, cfg.residual) for i in range(cfg.depth)
        )
        self.ups = nn.ModuleList(
            Up(chs[i + 1], chs[i], cfg.skip, cfg.double, cfg.residual) for i in reversed(range(cfg.depth))
        )
        self.head = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feats = [x]
        for down in self.downs:
            x = down(x)
            feats.append(x)
        feats.pop()  # the bottleneck itself is not a skip
        for up in self.ups:
            x = up(x, feats.pop() if self.cfg.skip else None)
        return self.head(x)  # logits, B x 1 x H x W


class FullyConnectedSeg(nn.Module):
    """The notebook's `FCN`: flatten the image, a stack of Linear layers, reshape.

    Every output pixel is connected to every input pixel, so the model has no notion
    of locality or translation invariance and its size grows with H*W. It is here as
    the "no convolutions, no hierarchy" baseline. (Not to be confused with the Fully
    *Convolutional* Network of Long et al. 2015.)

    The notebook ended with Linear -> BatchNorm -> ReLU, so its logits were >= 0: the
    probability of a pixel could never go below 0.5 (the BCE of every background pixel
    stayed >= ln 2) and the model could only say "background" by outputting exactly 0.
    The last layer here is a plain Linear.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        size = cfg.img_size
        h = cfg.fcn_hidden

        def fc(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(i, o, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(
            fc(cfg.in_ch * size * size, h * 2),  # "encoder": 3HW -> 2h -> h -> h/2
            fc(h * 2, h),
            fc(h, h // 2),
            fc(h // 2, h),  # "decoder": h/2 -> h -> 2h -> HW
            fc(h, h * 2),
            nn.Linear(h * 2, size * size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, hgt, wid = x.shape
        if hgt != self.cfg.img_size or wid != self.cfg.img_size:
            raise ValueError(f"fcn was built for {self.cfg.img_size}px inputs, got {hgt}x{wid}")
        return self.net(x.flatten(1)).view(b, 1, hgt, wid)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "fcn":
        return FullyConnectedSeg(cfg)
    return EncoderDecoder(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
