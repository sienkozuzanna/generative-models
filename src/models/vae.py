"""
Variational Autoencoder (VAE) for cat image generation.

Architecture:
    Encoder: Conv layers -> flatten -> mu, log_var
    Decoder: linear -> reshape -> ConvTranspose layers -> tanh

Output range: [-1, 1] (matches CatDataset normalization)
"""

import torch
import torch.nn as nn


# Encoder

class Encoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=128, base_channels=32):
        """
        Encodes an image to a latent distribution (mu, log_var).

        Args:
            image_size: input image size (64 or 256)
            latent_dim: size of the latent vector z
            base_channels: number of channels in the first conv layer (doubles each layer: 32 -> 64 -> 128 -> 256)
        """
        super().__init__()

        self.net = nn.Sequential(
            # image_size -> image_size/2
            nn.Conv2d(3, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # image_size/2 -> image_size/4
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # image_size/4 -> image_size/8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # image_size/8 -> image_size/16
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # after 4 stride-2 convolutions: image_size / 16
        self.feature_map_size = image_size // 16
        self.flat_dim = base_channels * 8 * self.feature_map_size ** 2

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1) # flatten
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

# Decoder

class Decoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=128, base_channels=32):
        """
        Decodes a latent vector z back to an image.
        Output range: [-1, 1] via tanh.
        """
        super().__init__()

        self.feature_map_size = image_size // 16
        self.base_channels = base_channels
        self.flat_dim = base_channels * 8 * self.feature_map_size ** 2

        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.net = nn.Sequential(
            # image_size/16 -> image_size/8
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # image_size/8 -> image_size/4
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # image_size/4 -> image_size/2
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # image_size/2 -> image_size
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.base_channels * 8, self.feature_map_size, self.feature_map_size)
        return self.net(h)

# VAE

class VAE(nn.Module):
    def __init__(self, image_size=64, latent_dim=128, base_channels=32):
        """
        Variational Autoencoder.

        Args:
            image_size: 64 or 256
            latent_dim: size of the latent space (hyperparameter)
            base_channels: controls model capacity (hyperparameter)

        Example:
            model = VAE(image_size=64, latent_dim=128)
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var)
        """
        super().__init__()
        self.encoder = Encoder(image_size, latent_dim, base_channels)
        self.decoder = Decoder(image_size, latent_dim, base_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + eps * std
        Allows backprop through the sampling step.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # at inference time just use the mean
            return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, n, device):
        """Sample n random images from the prior N(0, I)."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x):
        """Encode images to latent vectors (returns mu)."""
        self.eval()
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def interpolate(self, z1, z2, steps=8):
        """
        Linearly interpolate between two latent vectors.
        Returns steps+2 images (including z1 and z2).
        """
        self.eval()
        ratios = torch.linspace(0, 1, steps + 2, device=z1.device)
        vectors = torch.stack([z1 + r * (z2 - z1) for r in ratios])
        return self.decoder(vectors)

# Loss 

def vae_loss(recon, target, mu, log_var, beta=1.0):
    """
    VAE loss = Reconstruction loss + beta * KL divergence

    Args:
        recon:reconstructed images  [B, 3, H, W], range [-1, 1]
        target: original images [B, 3, H, W], range [-1, 1]
        mu: latent mean [B, latent_dim]
        log_var: latent log variance [B, latent_dim]
        beta: weight of KL term (beta=1 -> standard VAE, beta>1 -> beta-VAE, better disentanglement)

    Returns:
        total_loss, recon_loss, kl_loss  (all scalars)
    """
    # reconstruction loss — MSE works well for images in [-1, 1]
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss