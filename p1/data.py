"""Dataset helpers. Synthetic data are strictly pipeline debugging, NOT scientific evidence."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PixelDataset(Dataset):
    def __init__(self, images, labels, augment=False, normalize=False):
        self.images = np.asarray(images, dtype=np.uint8)
        self.targets = np.asarray(labels, dtype=np.int64)
        self.transform = (transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) if augment else transforms.ToTensor())
        if normalize:
            self.transform = transforms.Compose([self.transform, transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.targets[i])


def make_synthetic(size, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(size, dtype=np.int64) % 10
    rng.shuffle(labels)
    images = rng.integers(0, 45, size=(size, 32, 32, 3), dtype=np.uint8)
    # Class-specific spatial/color patch for quick shape/gradient diagnostics.
    for i, label in enumerate(labels):
        y = 2 + (int(label) // 5) * 14
        x = 2 + (int(label) % 5) * 6
        images[i, y:y + 11, x:x + 5, :] = 180
        images[i, y:y + 11, x:x + 5, int(label) % 3] = 255
    return images, labels


def load_partition(config, *, train):
    """Training/validation never instantiate or load the official test dataset."""
    name = config["dataset"]
    if name == "synthetic-debug":
        size = config.get("synthetic_train_size", 96) if train else config.get("synthetic_test_size", 32)
        return make_synthetic(size, config["seed"] + (0 if train else 1))
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=config.get("data_root", "data_cache"), train=train,
                          download=config.get("download_data", False))
        return dataset.data, np.asarray(dataset.targets, dtype=np.int64)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def test_size(config):
    return 10000 if config["dataset"] == "cifar10" else config.get("synthetic_test_size", 32)
