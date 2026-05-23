import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN Generator (convolutional).

    Maps a latent vector z ~ N(0, I) to a (3, image_size, image_size) image
    via a linear projection followed by transposed convolutions.

    Supports image_size in {32, 64, 128, 256}.
    Channel width is capped at 512 to keep memory reasonable.
    """

    def __init__(self, latent_dim: int = 128, base_channels: int = 64, image_size: int = 64):
        super().__init__()
        assert image_size in (32, 64, 128, 256), \
            "image_size must be 32, 64, 128 or 256"
        self.latent_dim = latent_dim

        # Number of upsampling steps: spatial goes 4 -> image_size
        n_ups = int(np.log2(image_size)) - 2
        init_ch = min(base_channels * (2 ** (n_ups - 1)), 512)
        self.init_ch = init_ch

        self.project = nn.Sequential(
            nn.Linear(latent_dim, init_ch * 4 * 4, bias=False),
            nn.BatchNorm1d(init_ch * 4 * 4),
            nn.ReLU(True),
        )

        layers = []
        in_ch = init_ch
        for _ in range(n_ups - 1):
            out_ch = min(in_ch // 2, 512) if in_ch > base_channels else base_channels
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch
        # Final upsample -> 3 channels, no BN
        layers += [
            nn.ConvTranspose2d(in_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> img: (B, 3, H, W)"""
        x = self.project(z)
        x = x.view(x.size(0), self.init_ch, 4, 4)
        return self.conv(x)

    @torch.no_grad()
    def generate(self, n: int, device) -> torch.Tensor:
        """Sample n images from the prior."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator (convolutional).

    Takes a (3, image_size, image_size) image and returns a raw logit (use with BCEWithLogitsLoss).

    Supports image_size in {32, 64, 128, 256}.
    No BatchNorm on the first layer (standard DCGAN practice).
    Channel width is capped at 512.
    """

    def __init__(self, base_channels: int = 64, image_size: int = 64):
        super().__init__()
        assert image_size in (32, 64, 128, 256), \
            "image_size must be 32, 64, 128 or 256"

        n_downs = int(np.log2(image_size)) - 2

        layers = []
        in_ch = 3
        out_ch = base_channels
        for i in range(n_downs):
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                # No BN on first layer
                *([] if i == 0 else [nn.BatchNorm2d(out_ch)]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        # 4x4 spatial -> scalar logit
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) -> logits: (B, 1, 1, 1)"""
        return self.net(img)
