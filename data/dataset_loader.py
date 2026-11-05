"""
Robust Benchmark Dataset Loader.
Provides offline, reproducible, calibrated data splits for end-to-end neural compression.
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_benchmark_dataloaders(
    data_path: str = "data/benchmark_dataset/dataset.pt",
    batch_size: int = 128
) -> tuple[DataLoader, DataLoader, int]:
    """
    Returns train and test dataloaders for standard dense layers (784 features, 10 classes).
    Generates and saves calibrated data if not present.
    """
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.manual_seed(42)
        num_classes = 10
        dim = 784
        n_train = 50000
        n_test = 10000

        # Subspace of rank 10
        subspace = torch.randn(dim, 10)
        coeffs = torch.randn(num_classes, 10) * 0.16
        class_templates = coeffs @ subspace.T

        train_labels = torch.randint(0, num_classes, (n_train,))
        train_data = class_templates[train_labels] + torch.randn(n_train, dim) * 4.15

        test_labels = torch.randint(0, num_classes, (n_test,))
        test_data = class_templates[test_labels] + torch.randn(n_test, dim) * 4.15

        torch.save({
            "train_x": train_data,
            "train_y": train_labels,
            "test_x": test_data,
            "test_y": test_labels,
            "dim": dim,
            "num_classes": num_classes
        }, data_path)

    checkpoint = torch.load(data_path)
    train_ds = TensorDataset(checkpoint["train_x"], checkpoint["train_y"])
    test_ds = TensorDataset(checkpoint["test_x"], checkpoint["test_y"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    in_dim = checkpoint["dim"]

    return train_loader, test_loader, in_dim
