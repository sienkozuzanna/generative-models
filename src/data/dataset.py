from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CatDataset(Dataset):
    def __init__(self, data_dir, image_size=64, augment=True):
        """
        Cat dataset — loads preprocessed images (after face crop).

        Args:
            data_dir: folder with images, e.g. data/processed/cats_64 or /kaggle/input/cats-processed-64/cats_64
            image_size: target image size (64 or 256)
            augment: True during training, False during evaluation/generation
        """
        self.paths = sorted(Path(data_dir).glob("*.jpg"))
        assert len(self.paths) > 0, f"No images found in {data_dir}"

        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]

        if augment:
            transform_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size, padding=int(image_size * 0.05)),
            ]

        transform_list += [
            transforms.ToTensor(),
            # normalize to [-1, 1] — required by GAN/DCGAN (tanh output)
            # inverse: x * 0.5 + 0.5 to get back to [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def get_dataloader(data_dir, image_size=64, batch_size=32,
                   augment=True, num_workers=2):
    """
    Shortcut — returns a ready DataLoader.

    Example:
        loader = get_dataloader("data/processed/cats_64", batch_size=64)
        batch = next(iter(loader))  # [64, 3, 64, 64], values in [-1, 1]
    """
    dataset = CatDataset(data_dir, image_size=image_size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # important for GAN — avoid incomplete batches
    )
    return loader


def denormalize(tensor):
    """
    Reverse normalization [-1, 1] -> [0, 1] for visualization.

    Example:
        imgs = denormalize(batch) # tensor [B, C, H, W]
        imgs = imgs.permute(0,2,3,1).numpy()
        plt.imshow(imgs[0])
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)