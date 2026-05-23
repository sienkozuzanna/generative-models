import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al. 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int64 -> emb: (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """ResNet block conditioned on timestep embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2-D feature maps."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    """U-Net noise predictor with sinusoidal time embeddings and self-attention.

    Matches the architecture from Ho et al. 2020 (DDPM):
    - GroupNorm throughout
    - SiLU activations
    - Self-attention at specified resolutions
    - Sinusoidal timestep conditioning injected via ResBlock

    Args:
        image_size: Spatial size of input images (H = W).
        in_channels: Number of input/output channels (3 for RGB).
        base_channels: Base channel width (multiplied per level).
        channel_mults: Channel multiplier per resolution level.
        num_res_blocks: Number of ResBlocks per level.
        attn_resolutions: Spatial resolutions where self-attention is applied.
    """

    def __init__(self, image_size: int = 64, in_channels: int = 3, base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2), num_res_blocks: int = 2, attn_resolutions: tuple = (16, 8)):
        super().__init__()
        self.image_size = image_size
        emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks  = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        res = image_size
        in_ch = base_channels
        self._skip_chs = [in_ch]

        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResBlock(in_ch, out_ch, emb_dim),
                    SelfAttention(out_ch) if res in attn_resolutions else nn.Identity(),
                ]))
                self._skip_chs.append(out_ch)
                in_ch = out_ch
            self.down_samples.append(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
            )
            self._skip_chs.append(in_ch)
            res //= 2

        self.mid_res1 = ResBlock(in_ch, in_ch, emb_dim)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_res2 = ResBlock(in_ch, in_ch, emb_dim)

        self.up_blocks   = nn.ModuleList()
        self.up_samples  = nn.ModuleList()
        skip_chs = list(self._skip_chs)

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_samples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
            ))
            res *= 2
            for i in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                self.up_blocks.append(nn.ModuleList([
                    ResBlock(in_ch + skip_ch, out_ch, emb_dim),
                    SelfAttention(out_ch) if res in attn_resolutions else nn.Identity(),
                ]))
                in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W), t: (B,) int64 -> noise_pred: (B,3,H,W)"""
        emb = self.time_mlp(t)
        x   = self.init_conv(x)

        skips = [x]
        ds_iter  = iter(self.down_samples)
        block_i  = 0
        for level_i in range(len(self.down_samples)):
            for _ in range(len([b for b in self.down_blocks]) // len(self.down_samples)):
                if block_i >= len(self.down_blocks):
                    break
                blk = self.down_blocks[block_i]
                x = blk[0](x, emb)
                x = blk[1](x)
                skips.append(x)
                block_i += 1
            x = next(ds_iter)(x)
            skips.append(x)

        x = self.mid_res1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, emb)

        us_iter = iter(self.up_samples)
        block_i = 0
        n_mults = len(self.down_samples)
        blocks_per_level = len(self.up_blocks) // n_mults
        for level_i in range(n_mults):
            x = next(us_iter)(x)
            for _ in range(blocks_per_level):
                if block_i >= len(self.up_blocks):
                    break
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                blk = self.up_blocks[block_i]
                x = blk[0](x, emb)
                x = blk[1](x)
                block_i += 1

        return self.out_conv(F.silu(self.out_norm(x)))


class DDPM(nn.Module):
    """DDPM (Ho et al. 2020) with linear noise schedule.

    Wraps UNet with:
    - forward diffusion q(x_t | x_0)  — used during training
    - reverse sampling p(x_{t-1} | x_t) — used during inference

    Training objective: predict the noise epsilon added at timestep t (simple MSE).

    Args:
        unet: UNet noise predictor.
        T: Total number of diffusion timesteps.
        beta_start: Start value of linear beta schedule.
        beta_end: End value of linear beta schedule.
    """

    def __init__(self, unet: UNet, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.unet = unet
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar",(1 - alpha_bar).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps"""
        s1 = self.sqrt_alpha_bar[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return s1 * x0 + s2 * eps

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Training step: sample random t, corrupt x0, predict noise, return MSE."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        return F.mse_loss(self.unet(xt, t), eps)

    @torch.no_grad()
    def generate(self, n: int, device, steps: int = None) -> torch.Tensor:
        """Ancestral DDPM sampling.

        Args:
            n: Number of images to generate.
            device: Target device.
            steps: Number of denoising steps (subset of T for speed). None = full T steps.

        Returns:
            Tensor of shape (n, 3, H, W) in range [-1, 1].
        """
        steps = steps or self.T
        img_size  = self.unet.image_size
        timesteps = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        x = torch.randn(n, 3, img_size, img_size, device=device)

        for i, t_val in enumerate(timesteps):
            t = t_val.expand(n)
            eps_pred = self.unet(x, t)

            beta_t = self.betas[t_val]
            alpha_t = self.alphas[t_val]
            ab_t = self.alpha_bar[t_val]

            # predict x0 from noise prediction
            x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            if t_val > 0:
                ab_prev = (
                    self.alpha_bar[timesteps[i + 1]]
                    if i + 1 < len(timesteps)
                    else torch.tensor(1.0, device=device)
                )
                mean = (
                    (ab_prev.sqrt() * beta_t  / (1 - ab_t)) * x0_pred
                  + (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
                )
                var  = (1 - ab_prev) / (1 - ab_t) * beta_t
                x    = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = x0_pred

        return x.clamp(-1, 1)