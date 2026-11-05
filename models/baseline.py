"""
Baseline Architecture Definitions.
Person A / Co-developed: MLP and Small CNN for CIFAR-10 / Fashion-MNIST.
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Standard 4-layer dense neural network.
    Input: 3072 (CIFAR-10 3x32x32 flattened) or 784 (Fashion-MNIST)
    Architecture: Input -> 1024 -> 512 -> 256 -> 10
    """
    def __init__(self, in_features: int = 3072, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out


class SmallCNN(nn.Module):
    """
    Small convolutional feature extractor followed by wide dense projection layers.
    Ideal for examining SVD compression on large dense classification heads.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        self.flatten = nn.Flatten()
        # Flattened features: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        out = self.fc3(x)
        return out
