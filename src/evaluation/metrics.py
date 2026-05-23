import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import lpips
    _lpips_fn = lpips.LPIPS(net='alex')
except:
    _lpips_fn = None


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 truncated at the pool layer — returns 2048-dim features."""

    def __init__(self, device):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        self.net = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.resize = transforms.Resize((299, 299), antialias=True)
        self.to(device)
        self.eval()
        self.device = device

    @torch.no_grad()
    def get_features(self, images_tensor):
        """
        Args:
            images_tensor: [N, 3, H, W] in range [-1, 1]
        Returns:
            features: [N, 2048] numpy array
        """
        imgs = self.resize((images_tensor * 0.5 + 0.5).clamp(0, 1))
        features = []
        loader = DataLoader(TensorDataset(imgs), batch_size=32)
        for (batch,) in loader:
            feat = self.net(batch.to(self.device))
            features.append(feat.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(features, axis=0)


def compute_fid(real_features, fake_features):
    """
    Frechet Inception Distance — lower is better.

    Args:
        real_features: [N, 2048] numpy array
        fake_features: [M, 2048] numpy array
    Returns:
        fid score (float)
    """
    mu_r = real_features.mean(0)
    mu_f = fake_features.mean(0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def compute_inception_score(images_tensor, device, splits=10):
    """
    Inception Score — higher is better.

    Args:
        images_tensor: [N, 3, H, W] in range [-1, 1]
        splits:        number of splits for IS estimation
    Returns:
        (mean, std) of IS across splits
    """
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    resize = transforms.Resize((299, 299), antialias=True)
    imgs = resize((images_tensor * 0.5 + 0.5).clamp(0, 1))

    preds = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(imgs), batch_size=32):
            logits = inception(batch.to(device))
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    scores = []
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size: (i + 1) * split_size]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))

def evaluate_model(generate_fn, real_loader, device, n_samples=5000):
    """
    Compute FID and IS for any generative model.

    Args:
        generate_fn:  callable(n: int) -> tensor [n, 3, H, W] in [-1, 1]

                      VAE: lambda n: model.generate(n, device)
                      DCGAN: lambda n: model.generate(n, device)
                      GAN: lambda n: model.generate(n, device)
                      DDPM: lambda n: ddpm.sample(n, device)

        real_loader:  DataLoader yielding real images in [-1, 1]
        device: 'cuda' or 'cpu'
        n_samples: number of samples (>=2048 recommended for stable FID)

    Returns:
        dict with keys: 'fid', 'is_mean', 'is_std'
    """
    print(f"Evaluating with {n_samples} samples...")
    extractor = InceptionFeatureExtractor(device)

    # real image features
    print("  Extracting real image features...")
    real_imgs = []
    for batch in tqdm(real_loader, desc="  Real"):
        real_imgs.append(batch)
        if sum(x.shape[0] for x in real_imgs) >= n_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:n_samples]
    real_features = extractor.get_features(real_imgs)

    # generated image features
    print("  Generating fake images...")
    fake_imgs = []
    with torch.no_grad():
        while sum(x.shape[0] for x in fake_imgs) < n_samples:
            fake_imgs.append(generate_fn(64).cpu())
    fake_imgs    = torch.cat(fake_imgs, dim=0)[:n_samples]
    fake_features = extractor.get_features(fake_imgs)

    # metrics
    print("  Computing FID...")
    fid = compute_fid(real_features, fake_features)

    print("  Computing Inception Score...")
    is_mean, is_std = compute_inception_score(fake_imgs, device)

    results = {"fid": fid, "is_mean": is_mean, "is_std": is_std}

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"FID: {fid:.2f}  (lower is better)")
    print(f"Inception Score: {is_mean:.2f} +/- {is_std:.2f}  (higher is better)")
    print("=" * 40)

    return results


def latent_smoothness(zs: torch.Tensor):
    """
    zs: [T, D]
    """
    diffs = torch.norm(zs[1:] - zs[:-1], dim=1)
    return diffs.mean().item()


def pixel_smoothness(images: torch.Tensor):
    """
    images: [T, C, H, W]
    """
    diffs = torch.norm(images[1:] - images[:-1], dim=(1, 2, 3))
    return diffs.mean().item()


def lpips_smoothness(images: torch.Tensor, device="cpu"):
    """
    images: [T, C, H, W], expected range [-1, 1]
    """
    if _lpips_fn is None:
        raise ImportError("Install lpips: pip install lpips")

    _lpips_fn.to(device)

    diffs = []
    for t in range(len(images) - 1):
        x1 = images[t:t+1].to(device)
        x2 = images[t+1:t+2].to(device)
        diffs.append(_lpips_fn(x1, x2).item())

    return sum(diffs) / len(diffs)


def interpolation_metrics(images, zs, device="cpu"):
    """
    Returns all metrics in one dict
    """
    return {
        "latent_smoothness": latent_smoothness(zs),
        "pixel_smoothness": pixel_smoothness(images),
        "lpips_smoothness": lpips_smoothness(images, device=device),
    }
