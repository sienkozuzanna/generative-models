import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


PALETTE = {
    "vae":  "#2E86AB",
    "dcgan": "#E84855",
    "gan": "#F5A623",
    "ddpm": "#44BBA4",
    "accent": "#9B59B6",
    "bg": "#FFFFFF",
    "grid": "#E0E0E0",
}

MODEL_COLORS = {
    "VAE": PALETTE["vae"],
    "DCGAN": PALETTE["dcgan"],
    "GAN": PALETTE["gan"],
    "DDPM": PALETTE["ddpm"],
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": False,
})


def _savefig(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")


#VAE loss curves (single run) 
def plot_vae_loss_curves(history: dict, title="VAE Training Curves", save_path=None):
    """
    Plot total, reconstruction, and KL loss curves for a single VAE run.

    Args:
        history:  dict with keys 'total', 'recon', 'kl' — lists of per-epoch values
        title: plot title
        save_path: if given, save figure here
    """
    epochs = range(1, len(history["total"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for ax, key, color, label in zip(
        axes,
        ["total", "recon", "kl"],
        [PALETTE["vae"], PALETTE["dcgan"], PALETTE["gan"]],
        ["Total Loss", "Reconstruction Loss", "KL Divergence"],
    ):
        ax.plot(epochs, history[key], color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()

# Loss curves — multiseed (mean ± std)
def plot_loss_curves_multiseed(seed_results: dict, loss_keys=("total",), loss_labels=("Loss",), 
                               title="Training Curves (mean ± std)", save_path=None):
    """
    Plot mean ± std loss curves across multiple seeds.

    Args:
        seed_results: dict {seed: result_dict} where result_dict['history'] has keys matching loss_keys
        loss_keys: which loss keys to plot
        loss_labels: display names for each loss
        title: plot title
        save_path: optional save path
    """
    colors = [PALETTE["vae"], PALETTE["dcgan"], PALETTE["gan"]]
    n_epochs = len(next(iter(seed_results.values()))["history"][loss_keys[0]])
    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(1, len(loss_keys), figsize=(5 * len(loss_keys), 4))
    if len(loss_keys) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for ax, key, color, label in zip(axes, loss_keys, colors, loss_labels):
        values = np.array([r["history"][key] for r in seed_results.values()])
        mean, std = values.mean(axis=0), values.std(axis=0)

        ax.plot(epochs, mean, color=color, linewidth=2, label="mean")
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, label="±std")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# GAN loss curves (generator + discriminator)
def plot_gan_loss_curves(history: dict, title="GAN Training Curves", save_path=None):
    """
    Plot generator and discriminator loss for GAN/DCGAN.

    Args:
        history: dict with keys 'g_loss', 'd_loss' — per-epoch lists
    """
    epochs = range(1, len(history["g_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for ax, key, color, label in zip(
        axes,
        ["g_loss", "d_loss"],
        [PALETTE["dcgan"], PALETTE["vae"]],
        ["Generator Loss", "Discriminator Loss"],
    ):
        ax.plot(epochs, history[key], color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# Model comparison — FID and IS bar charts 
def plot_model_comparison(results_dict: dict, title="Model Comparison", save_path=None):
    """
    Bar chart comparing FID and IS across models.
    Shows mean ± std if multiple seeds are present.

    Args:
        results_dict: dict mapping model_name -> list of result dictseach result dict has 'fid', 'is_mean', 'is_std'
                      each result dict has 'fid', 'is_mean', 'is_std'
        title: plot title
        save_path: optional save path
    """
    models = list(results_dict.keys())
    colors = [MODEL_COLORS.get(m, PALETTE["accent"]) for m in models]

    fid_means = [np.mean([r["fid"] for r in results_dict[m]]) for m in models]
    fid_stds = [np.std([r["fid"]  for r in results_dict[m]]) for m in models]
    is_means = [np.mean([r["is_mean"] for r in results_dict[m]]) for m in models]
    is_stds = [np.std([r["is_mean"]  for r in results_dict[m]]) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    # FID — lower is better
    bars = ax1.bar(models, fid_means, yerr=fid_stds, capsize=5,
                   color=colors, edgecolor="black", linewidth=0.6, width=0.5)
    for bar, m, s in zip(bars, fid_means, fid_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.5,
                 f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("FID ↓ (lower is better)")
    ax1.set_ylabel("FID Score")

    # IS — higher is better
    bars = ax2.bar(models, is_means, yerr=is_stds, capsize=5,
                   color=colors, edgecolor="black", linewidth=0.6, width=0.5)
    for bar, m, s in zip(bars, is_means, is_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.02,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.set_title("Inception Score ↑ (higher is better)")
    ax2.set_ylabel("IS")

    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# Hyperparameter comparison
def plot_hyperparam_comparison(results_dict: dict, param_name: str,
    title=None, save_path=None):
    """
    Compare FID and IS across hyperparameter values (mean ± std across seeds).

    Args:
        results_dict: dict {param_value: [result_dict_per_seed]} each result dict has 'fid', 'is_mean'
        param_name:  display name, e.g. "latent_dim" or "learning_rate"
        title: plot title
        save_path: optional save path

    """
    params = list(results_dict.keys())
    colors = sns.color_palette("deep", n_colors=len(params))

    fid_means = [np.mean([r["fid"] for r in results_dict[p]]) for p in params]
    fid_stds = [np.std([r["fid"]  for r in results_dict[p]]) for p in params]
    is_means = [np.mean([r["is_mean"] for r in results_dict[p]]) for p in params]
    is_stds = [np.std([r["is_mean"]  for r in results_dict[p]]) for p in params]

    x = np.arange(len(params))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or f"Hyperparameter: {param_name}", fontsize=15, fontweight="bold", y=1.02)

    for ax, means, stds, ylabel, direction in [
        (ax1, fid_means, fid_stds,  "FID",  "↓ lower is better"),
        (ax2, is_means,  is_stds,   "IS",   "↑ higher is better"),
    ]:
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.6)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in params])
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} {direction}")

    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()

# Generated images grid

def plot_image_grid(images, title="Generated Images", nrow=8, save_path=None):
    """
    Display a grid of generated images.

    Args:
        images: numpy array [N, H, W, 3] in range [0, 1] or tensor [N, 3, H, W] in range [-1, 1]
        title: plot title
        nrow: images per row
        save_path: optional save path
    """
    import torch
    if isinstance(images, torch.Tensor):
        images = (images * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    n = len(images)
    ncol = nrow
    nrow_actual = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(ncol * 1.5, nrow_actual * 1.5))
    axes = np.array(axes).flatten()

    for ax, img in zip(axes, images):
        ax.imshow(img.clip(0, 1))
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()

# Interpolation grid 
def plot_interpolation(images, title="Latent Space Interpolation (z₁ → z₂)", save_path=None,):
    """
    Display 10 interpolated images in a single row.

    Args:
        images: numpy [10, H, W, 3] or tensor [10, 3, H, W] in [-1, 1]
        title: plot title
        save_path: optional save path
    """
    import torch
    if isinstance(images, torch.Tensor):
        images = (images * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    n = len(images)
    labels = ["z₁"] + [f"t={i/(n-1):.1f}" for i in range(1, n - 1)] + ["z₂"]

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img.clip(0, 1))
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()

# Summary table
def print_summary_table(results_dict: dict):
    """
    Print a clean summary table of FID and IS for all models.

    Args:
        results_dict: dict {model_name: [result_dict_per_seed]}
    """
    print(f"\n{'Model':<10} {'FID':>12} {'IS':>16}")
    print("-" * 42)
    for model, results in results_dict.items():
        fid_mean = np.mean([r["fid"] for r in results])
        fid_std = np.std([r["fid"]  for r in results])
        is_mean = np.mean([r["is_mean"] for r in results])
        is_std = np.std([r["is_mean"]  for r in results])
        print(f"{model:<10} {fid_mean:>6.2f} ± {fid_std:<4.2f}  {is_mean:>5.2f} ± {is_std:.2f}")
    print()
