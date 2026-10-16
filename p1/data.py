"""Dataset helpers. Synthetic data are strictly pipeline debugging, NOT scientific evidence."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PixelDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = np.asarray(images, dtype=np.uint8)
        self.targets = np.asarray(labels, dtype=np.int64)
        self.transform = (transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) if augment else transforms.ToTensor())

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


def load_images(config):
    name = config["dataset"]
    if name == "synthetic-debug":
        tr_x, tr_y = make_synthetic(config.get("synthetic_train_size", 96), config["seed"])
        te_x, te_y = make_synthetic(config.get("synthetic_test_size", 32), config["seed"] + 1)
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        d0 = CIFAR10(root="data_cache", train=True, download=True)
        d1 = CIFAR10(root="data_cache", train=False, download=True)
        tr_x, tr_y = d0.data, np.asarray(d0.targets, dtype=np.int64)
        te_x, te_y = d1.data, np.asarray(d1.targets, dtype=np.int64)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return tr_x, np.asarray(tr_y), te_x, np.asarray(te_y)
