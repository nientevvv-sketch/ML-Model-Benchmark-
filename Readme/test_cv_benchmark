import numpy as np
import torch
from pathlib import Path
from torchvision.datasets import FakeData
import torchvision.transforms as transforms

import cv_benchmark


def _make_fake_cifar(total_train, total_test):
    def _fake(root, train, download, transform):
        size = total_train if train else total_test
        return FakeData(size=size, image_size=(3, 32, 32), num_classes=10, transform=transform)

    return _fake


def test_load_datasets_with_fake(monkeypatch, tmp_path):
    # Replace CIFAR10 with a fast FakeData implementation to avoid downloads
    total_train = 50
    total_test = 20
    monkeypatch.setattr(cv_benchmark.datasets, "CIFAR10", _make_fake_cifar(total_train, total_test))

    train_ds, val_ds, test_ds, class_names = cv_benchmark.load_datasets(
        data_dir=tmp_path, train_samples=total_train, test_samples=total_test, seed=42
    )

    # Check sizes: train + val == total_train, test == total_test
    assert len(train_ds) + len(val_ds) == total_train
    assert len(test_ds) == total_test
    assert isinstance(class_names, list)
    assert len(class_names) == 10


def test_train_model_smoke(monkeypatch, tmp_path):
    # Use slightly larger fake data for a minimal training smoke test
    total_train = 100
    total_test = 32
    monkeypatch.setattr(cv_benchmark.datasets, "CIFAR10", _make_fake_cifar(total_train, total_test))

    train_ds, val_ds, test_ds, class_names = cv_benchmark.load_datasets(
        data_dir=tmp_path, train_samples=total_train, test_samples=total_test, seed=123
    )

    # Create loaders
    train_loader = cv_benchmark.make_loader(train_ds, batch_size=16, shuffle=True, seed=0)
    val_loader = cv_benchmark.make_loader(val_ds, batch_size=16, shuffle=False, seed=1)

    # Build a small CNN from the module and run a single epoch training to ensure nothing crashes
    model = cv_benchmark.build_models(num_classes=len(class_names))["CNN"]
    trained_model, history, elapsed = cv_benchmark.train_model(
        name="CNN", model=model, train_loader=train_loader, validation_loader=val_loader, epochs=1, device=torch.device("cpu")
    )

    # Basic assertions on outputs
    assert hasattr(trained_model, "state_dict")
    assert isinstance(history, list) and len(history) == 1
    assert isinstance(elapsed, float) and elapsed >= 0.0


def test_predict_and_evaluate(monkeypatch, tmp_path):
    # Fast

