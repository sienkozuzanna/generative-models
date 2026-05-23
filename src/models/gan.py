import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Vanilla GAN Generator (MLP).

    Maps a latent vector z ~ N(0, I) to a flattened image, then reshapes to (3, image_size, image_size).

    Architecture: Linear -> [hidden layers] -> Linear -> Tanh
    """

    def __init__(self, latent_dim: int = 128, image_size: int = 64,
        hidden_dims: list[int] = [256, 512, 1024]):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.out_dim = 3 * image_size * image_size

        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            in_dim = h
        layers += [
            nn.Linear(in_dim, self.out_dim),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> img: (B, 3, image_size, image_size)"""
        x = self.net(z)
        return x.view(x.size(0), 3, self.image_size, self.image_size)

    @torch.no_grad()
    def generate(self, n: int, device) -> torch.Tensor:
        """Sample n images from the prior."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z)


class Discriminator(nn.Module):
    """Vanilla GAN Discriminator (MLP).

    Takes a (3, image_size, image_size) image, flattens it, and returns a raw logit (use with BCEWithLogitsLoss).

    Architecture: Linear -> [hidden layers] -> Linear (logit)
    """

    def __init__(self, image_size: int = 64, hidden_dims: list[int] = [1024, 512, 256]):
        super().__init__()
        in_dim = 3 * image_size * image_size

        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) -> logits: (B, 1)"""
        x = img.view(img.size(0), -1)
        return self.net(x)
