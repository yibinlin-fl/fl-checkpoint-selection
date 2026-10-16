"""Simple debugging CNN and CIFAR-adapted ResNet18 (no pretrained weights)."""
import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 4 * 4, classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name, classes=10):
    if name == "small_cnn":
        return SmallCNN(classes)
    if name == "resnet18_cifar":
        model = models.resnet18(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model {name}")
