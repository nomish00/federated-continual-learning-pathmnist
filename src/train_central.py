import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class SmallCNN(nn.Module):
    """Compact CNN suitable for a CPU-only first experiment."""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_threads < 1:
        raise ValueError("epochs, batch-size, and num-threads must be positive")

    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    transform = transforms.ToTensor()

    print("Loading PathMNIST (28x28 RGB)...", flush=True)
    train_ds = PathMNIST(split="train", transform=transform, download=True)
    test_ds = PathMNIST(split="test", transform=transform, download=True)
    num_classes = len(train_ds.info["label"])
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"Classes: {num_classes} | Device: {device} | CPU threads: {args.num_threads}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    model = SmallCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples_seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.size(0)
            running_loss += loss.item() * batch_n
            examples_seen += batch_n

        train_loss = running_loss / max(examples_seen, 1)
        test_metrics = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **test_metrics}
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"accuracy={test_metrics['accuracy']:.4f} | "
            f"macro_f1={test_metrics['macro_f1']:.4f} | "
            f"balanced_acc={test_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    training_seconds = time.perf_counter() - start
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "central_model.pt"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    # Linux reports ru_maxrss in KiB. Convert to MiB.
    import resource
    peak_ram_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    model_size_bytes = model_path.stat().st_size

    result = {
        "experiment": "centralised_pathmnist_baseline",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_threads": args.num_threads,
        "device": str(device),
        "torch_version": torch.__version__,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "num_classes": num_classes,
        "training_seconds": training_seconds,
        "peak_process_ram_mib": peak_ram_mib,
        "model_size_bytes": model_size_bytes,
        "final_test_metrics": history[-1] | {"epoch": args.epochs},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSaved results to:", output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
