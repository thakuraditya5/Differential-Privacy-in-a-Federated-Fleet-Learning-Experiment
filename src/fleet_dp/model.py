"""Neural network model definitions.

Provides small CNN architectures suitable for MNIST and CIFAR-10
that can run on a normal laptop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """Small CNN for MNIST classification.

    Architecture:
        Conv2D(1, 16, 5) -> ReLU -> MaxPool(2)
        Conv2D(16, 32, 5) -> ReLU -> MaxPool(2)
        Flatten -> Linear(512, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cifar10CNN(nn.Module):
    """Small CNN for CIFAR-10 classification.

    Architecture:
        Conv2D(3, 32, 3) -> ReLU -> MaxPool(2)
        Conv2D(32, 64, 3) -> ReLU -> MaxPool(2)
        Conv2D(64, 64, 3) -> ReLU
        Flatten -> Linear(1024, 256) -> ReLU -> Linear(256, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(name: str = "mnist_cnn") -> nn.Module:
    """Factory function to create a model by name.

    Args:
        name: Model name ('mnist_cnn' or 'cifar10_cnn').

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model name is unsupported.
    """
    models = {
        "mnist_cnn": MnistCNN,
        "cifar10_cnn": Cifar10CNN,
    }
    if name not in models:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(models.keys())}"
        )
    return models[name]()
