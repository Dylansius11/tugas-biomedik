"""U-Net (docs/PROTOCOL.md section 5). Outputs logits; apply sigmoid outside."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, filters=(32, 64, 128, 256), bottleneck: int = 512, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.encoders = nn.ModuleList()
        c = in_ch
        for f in filters:
            self.encoders.append(_block(c, f))
            c = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(c, bottleneck)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        c = bottleneck
        for f in reversed(filters):
            self.ups.append(nn.ConvTranspose2d(c, f, 2, stride=2))
            self.decoders.append(_block(2 * f, f))
            c = f
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.head(x)


def build_model(cfg: dict) -> UNet:
    t = cfg["train"]
    return UNet(tuple(t["filters"]), t["bottleneck"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
