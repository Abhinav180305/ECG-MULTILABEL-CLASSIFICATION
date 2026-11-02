"""1D ResNet + Transformer encoder for 12-lead ECG (multi-label), with a masked-reconstruction
head for optional self-supervised pretraining."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {
    #          widths                  blocks        heads  tf_layers
    "tiny":  ((16, 32, 64, 128),      (1, 1, 1, 1), 4,     1),     # smoke tests / CPU
    "small": ((32, 64, 128, 256),     (2, 2, 2, 2), 4,     1),
    "base":  ((64, 128, 256, 512),    (2, 2, 2, 2), 8,     2),
    "large": ((128, 256, 512, 1024),  (3, 4, 6, 3), 8,     4),     # for ~1M+ records
}


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, k: int = 7, p_drop: float = 0.1):
        super().__init__()
        pad = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride, pad, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, pad, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p_drop)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(self.drop(y)))
        return F.relu(y + (x if self.short is None else self.short(x)))


def sinusoidal_pe(length: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe


class ECGNet(nn.Module):
    def __init__(self, n_labels: int, size: str = "base", in_ch: int = 12, dropout: float = 0.3):
        super().__init__()
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}")
        widths, blocks, heads, tf_layers = SIZES[size]
        self.cfg = {"n_labels": n_labels, "size": size, "in_ch": in_ch, "dropout": dropout}
        self.stem = nn.Sequential(nn.Conv1d(in_ch, widths[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for si, (w, nb) in enumerate(zip(widths, blocks)):
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                layers.append(ResBlock(cin, w, stride))
                cin = w
        self.body = nn.Sequential(*layers)
        self.d_model = widths[-1]
        enc = nn.TransformerEncoderLayer(self.d_model, heads, dim_feedforward=2 * self.d_model,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, tf_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * self.d_model, n_labels))
        # SSL decoder: features -> upsample to input length -> 12 leads
        self.recon = nn.Sequential(nn.Conv1d(self.d_model, 64, 5, padding=2), nn.ReLU(inplace=True),
                                   nn.Conv1d(64, in_ch, 5, padding=2))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, T) -> (B, C, T') contextualized features."""
        h = self.body(self.stem(x))                      # (B, C, T')
        t = h.transpose(1, 2)                            # (B, T', C)
        t = t + sinusoidal_pe(t.shape[1], t.shape[2], t.device).unsqueeze(0)
        return self.transformer(t).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(pooled)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = F.interpolate(h, size=x.shape[-1], mode="linear", align_corners=False)
        return self.recon(h)


def build_model(n_labels: int, size: str, dropout: float = 0.3) -> ECGNet:
    return ECGNet(n_labels=n_labels, size=size, dropout=dropout)
