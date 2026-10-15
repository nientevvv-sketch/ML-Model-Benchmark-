"""Computer Vision benchmark: a small CNN vs. ResNet18 on CIFAR-10.

This module is independent of the existing tabular ``main.py`` benchmark.
It uses the official CIFAR-10 train/test split, reserves part of the selected
training subset for validation, and evaluates on the test subset only once,
after training has finished.

Quick CPU run:
    python cv_benchmark.py

Longer experiment:
    python cv_benchmark.py --train-samples 20000 --test-samples 5000 --epochs 5

The first run downloads CIFAR-10. Results are written to
``results/computer_vision/``.
"""


import argparse
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18







# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Resolve paths relative to this script, not the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "computer_vision"
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 channel statistics. Both models use the same preprocessing.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)




def parse_args() -> argparse.Namespace:
    """Parse settings so CPU runtime can be adjusted without editing code."""
    parser = argparse.ArgumentParser(
        description="Compare a small CNN and ResNet18 on CIFAR-10 (CPU-friendly defaults)."
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per model (default: 2).")
    parser.add_argument(
        "--train-samples", type=int, default=5000,
        help="Number of CIFAR-10 training-pool images, including validation images (default: 5000).",
    )
    parser.add_argument(
        "--test-samples", type=int, default=1000,
        help="Number of official test images used for final evaluation (default: 1000).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument(
        "--num-threads", type=int, default=min(4, torch.get_num_threads()),
        help="Maximum CPU threads used by PyTorch (default: up to 4).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed (default: 42).")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Where CIFAR-10 will be downloaded and cached.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Where CV reports and plots will be saved.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable splits, initialization, and data order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The benchmark is CPU-only; deterministic algorithms reduce run-to-run variance.
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_models(num_classes: int = 10) -> dict[str, nn.Module]:
    """Create two models that accept 32x32 RGB CIFAR-10 images.

    ResNet18 is trained from scratch, not loaded with ImageNet pretrained weights.
    Its first convolution and max-pooling layer are adapted for small images.
    """
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 32x32 -> 16x16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 16x16 -> 8x8
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    resnet = resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    return {"CNN": cnn, "ResNet18": resnet}


def load_datasets(
    data_dir: Path,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[Subset, Subset, Subset, list[str]]:
    """Load CIFAR-10 and create deterministic train/validation/test subsets.

    ``train_samples`` is the total number selected from the official training
    split; 10% of that subset is held out for validation. The official test
    split is kept separate and is only used for final evaluation.
    """
    if not 2 <= train_samples <= 50_000:
        raise ValueError("--train-samples must be between 2 and 50000.")
    if not 1 <= test_samples <= 10_000:
        raise ValueError("--test-samples must be between 1 and 10000.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    class_names = list(train_dataset.classes)

    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_dataset), generator=generator)[:train_samples].tolist()
    test_generator = torch.Generator().manual_seed(seed + 1)
    test_indices = torch.randperm(len(test_dataset), generator=test_generator)[:test_samples].tolist()

    # A deterministic 90/10 split from the selected training pool.
    shuffled = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(seed + 2)).tolist()
    validation_size = max(1, int(round(len(train_indices) * 0.1)))
    validation_positions = set(shuffled[:validation_size])
    fit_indices = [idx for pos, idx in enumerate(train_indices) if pos not in validation_positions]
    validation_indices = [idx for pos, idx in enumerate(train_indices) if pos in validation_positions]

    return (
        Subset(train_dataset, fit_indices),
        Subset(train_dataset, validation_indices),
        Subset(test_dataset, test_indices),
        class_names,
    )


def make_loader(
    dataset: Subset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a CPU-friendly data loader with deterministic shuffling."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Reliable default on Windows and avoids worker overhead on CPU.
        pin_memory=False,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training or validation epoch and return mean loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_count

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    """Train one model and record training/validation curves."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = run_epoch(
            model, validation_loader, criterion, None, device
        )
        history.append({
            "Model": name,
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
        })
        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.3f} | "
            f"val loss={val_loss:.4f}, val acc={val_accuracy:.3f}"
        )

    elapsed = time.perf_counter() - start
    return model, history, elapsed


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect labels, predicted classes, and class probabilities."""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            all_labels.append(labels.numpy())
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Compute multiclass metrics with macro averaging for balanced comparison."""
    return {
        "Model": name,
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Test Precision (macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Test Recall (macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Test F1 (macro)": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Test F1 (weighted)": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Training Time (sec)": elapsed_seconds,
        "Test Samples": int(len(y_true)),
    }


def save_classification_reports(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save class-level precision, recall, F1, and support to CSV."""
    rows: list[dict[str, Any]] = []
    for model_name, (y_true, y_pred, _) in predictions.items():
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        for class_name in class_names:
            rows.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1-score": report[class_name]["f1-score"],
                "Support": int(report[class_name]["support"]),
            })

    pd.DataFrame(rows).to_csv(output_dir / "classification_reports.csv", index=False)


def plot_confusion_matrices(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save a side-by-side confusion matrix comparison."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(17, 7))
    axes = np.atleast_1d(axes)
    for ax, (name, (y_true, y_pred, _)) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            display_labels=class_names,
            xticks_rotation=45,
            values_format="d",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle("CIFAR-10 Test Confusion Matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_rows: list[dict[str, float]], output_dir: Path) -> None:
    """Plot train/validation loss and accuracy by epoch."""
    history = pd.DataFrame(history_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, group in history.groupby("Model", sort=False):
        axes[0].plot(group["Epoch"], group["Train Loss"], marker="o", label=f"{name} train")
        axes[0].plot(group["Epoch"], group["Validation Loss"], marker="o", linestyle="--", label=f"{name} val")
        axes[1].plot(group["Epoch"], group["Train Accuracy"], marker="o", label=f"{name} train")
        axes[1].plot(group["Epoch"], group["Validation Accuracy"], marker="o", linestyle="--", label=f"{name} val")

    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    models: dict[str, nn.Module],
    test_dataset: Subset,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    seed: int,
    max_images: int = 8,
) -> None:
    """Visualize a shared set of test images and each model's predictions."""
    rng = np.random.default_rng(seed)
    count = min(max_images, len(test_dataset))
    chosen = rng.choice(len(test_dataset), size=count, replace=False).tolist()
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    model_names = list(models.keys())

    fig, axes = plt.subplots(count, len(model_names), figsize=(8 * len(model_names) / 2, 2.5 * count))
    axes = np.asarray(axes).reshape(count, len(model_names))

    for row, subset_index in enumerate(chosen):
        image, true_label = test_dataset[subset_index]
        display_image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        batch = image.unsqueeze(0).to(device)

        for col, model_name in enumerate(model_names):
            model = models[model_name]
            model.eval()
            with torch.no_grad():
                probabilities = torch.softmax(model(batch), dim=1)[0]
                predicted_label = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted_label].item())

            ax = axes[row, col]
            ax.imshow(display_image)
            ax.set_title(
                f"{model_name}\nPred: {class_names[predicted_label]} ({confidence:.0%})\n"
                f"True: {class_names[int(true_label)]}",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle("Example Predictions on the CIFAR-10 Test Subset", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")  # This script intentionally targets CPU-only execution.

    print("=== Computer Vision Benchmark: CIFAR-10 ===")
    print(f"Device: {device}")
    print(f"CPU threads: {args.num_threads}")
    print(f"Training pool: {args.train_samples} images (10% reserved for validation)")
    print(f"Final test subset: {args.test_samples} images")
    print(f"Epochs per model: {args.epochs}")
    print("Note: first run downloads CIFAR-10 if it is not already cached.\n")

    train_dataset, validation_dataset, test_dataset, class_names = load_datasets(
        args.data_dir, args.train_samples, args.test_samples, args.seed
    )
    print(f"Classes: {', '.join(class_names)}")
    print(f"Train/validation/test sizes: {len(train_dataset)}/{len(validation_dataset)}/{len(test_dataset)}")

    # The official test set is not used for training, epoch selection, or tuning.
    model_names = ["CNN", "ResNet18"]
    all_history: list[dict[str, float]] = []
    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    trained_models: dict[str, nn.Module] = {}

    for model_index, name in enumerate(model_names):
        print(f"\nTraining {name}...")
        # Seed before model construction so parameter initialization is repeatable.
        set_seed(args.seed + model_index)
        model = build_models(num_classes=len(class_names))[name]
        # Reuse the same sample order seed for both models for a fair comparison.
        train_loader = make_loader(
            train_dataset, args.batch_size, shuffle=True, seed=args.seed + 10
        )
        validation_loader = make_loader(
            validation_dataset, args.batch_size, shuffle=False, seed=args.seed + 20
        )
        model, history, elapsed = train_model(
            name, model, train_loader, validation_loader, args.epochs, device
        )
        all_history.extend(history)
        trained_models[name] = model

        test_loader = make_loader(
            test_dataset, args.batch_size, shuffle=False, seed=args.seed + 30
        )
        y_true, y_pred, probabilities = predict(model, test_loader, device)
        predictions[name] = (y_true, y_pred, probabilities)
        metrics_rows.append(evaluate_model(name, y_true, y_pred, elapsed))
        print(
            f"Final test accuracy: {accuracy_score(y_true, y_pred):.4f} | "
            f"macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
        )

    # Write only to the dedicated CV directory; tabular benchmark files are untouched.
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by="Test F1 (macro)", ascending=False
    ).reset_index(drop=True)
    metrics_df.to_csv(args.results_dir / "final_report.csv", index=False)
    pd.DataFrame(all_history).to_csv(args.results_dir / "training_history.csv", index=False)
    save_classification_reports(predictions, class_names, args.results_dir)
    plot_confusion_matrices(predictions, class_names, args.results_dir)
    plot_training_curves(all_history, args.results_dir)
    plot_predictions(
        trained_models, test_dataset, class_names, device, args.results_dir, args.seed
    )

    print("\n=== Final CV Comparison ===")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nResults saved to: {args.results_dir.resolve()}")
    for filename in [
        "final_report.csv",
        "training_history.csv",
        "classification_reports.csv",
        "confusion_matrices.png",
        "training_curves.png",
        "predictions.png",
    ]:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
