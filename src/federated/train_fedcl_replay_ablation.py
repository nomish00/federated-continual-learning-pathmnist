#!/usr/bin/env python3
"""
Federated continual learning replay-memory ablation for PathMNIST.

Runs the same sequential 3-task / 3-client FedAvg-with-local-replay experiment
for several replay-buffer budgets (examples per class per client).

Example:
    python src/federated/train_fedcl_replay_ablation.py \
        --memory-per-class 10 25 50 100 \
        --num-clients 3 --rounds-per-task 5 --local-epochs 1 \
        --batch-size 128 --lr 0.001 --seed 42

Dependencies:
    pip install torch torchvision medmnist scikit-learn psutil pandas

Notes:
- Each client stores dataset indices in its replay buffer (not image copies).
- Communication is an estimate of raw model tensor payload:
  one full model download and one full model upload per client per round.
- Test metrics are emitted after every round for continuity with the existing
  experiment. For final research reporting, select hyperparameters on a
  validation split and evaluate the test set only once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    import psutil
except ImportError:
    psutil = None

import medmnist
from medmnist import INFO


TASKS: List[List[int]] = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


class SmallCNN(nn.Module):
    """Compact 3-channel CNN matching the project baseline."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """Read MedMNIST labels robustly across common target shapes."""
    targets = getattr(dataset, "labels", None)
    if targets is None:
        targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Could not find labels/targets on the MedMNIST dataset.")
    return np.asarray(targets).reshape(-1).astype(int)


def make_client_partitions(
    labels: np.ndarray, num_clients: int, seed: int, alpha: float
) -> List[np.ndarray]:
    """Dirichlet class-wise partition; alpha <= 0 gives an IID split."""
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    if alpha <= 0:
        indices = rng.permutation(len(labels))
        return [np.asarray(x, dtype=np.int64) for x in np.array_split(indices, num_clients)]

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        splits = np.split(class_indices, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    result = []
    for indices in client_indices:
        rng.shuffle(indices)
        result.append(np.asarray(indices, dtype=np.int64))
    return result


def state_dict_bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def current_rss_mib() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def peak_rss_mib() -> float:
    """Best-effort process peak RSS; Linux reports ru_maxrss in KiB."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KiB; macOS: bytes.
        if os.name == "posix" and value > 10_000_000:
            return value / (1024 ** 2)
        return value / 1024
    except Exception:
        return current_rss_mib()


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    seen_classes: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    seen = list(seen_classes)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.reshape(-1).long().cpu().numpy()
            logits = model(images)
            # Exclude future/unseen classes from predictions.
            logits[:, [c for c in range(9) if c not in seen]] = -1e9
            predictions = logits.argmax(dim=1).cpu().numpy()
            mask = np.isin(labels, seen)
            y_true.extend(labels[mask].tolist())
            y_pred.extend(predictions[mask].tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=seen, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_each_task(
    model: nn.Module,
    test_loader: DataLoader,
    tasks: Sequence[Sequence[int]],
    completed_task: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate each completed task separately, masking to that task's labels."""
    results: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        cached = []
        for images, labels in test_loader:
            logits = model(images.to(device)).cpu().numpy()
            labels = labels.reshape(-1).numpy().astype(int)
            cached.append((logits, labels))

    for task_idx in range(completed_task + 1):
        classes = list(tasks[task_idx])
        truth, pred = [], []
        for logits, labels in cached:
            mask = np.isin(labels, classes)
            if not np.any(mask):
                continue
            task_logits = logits[mask].copy()
            task_logits[:, [c for c in range(9) if c not in classes]] = -1e9
            truth.extend(labels[mask].tolist())
            pred.extend(task_logits.argmax(axis=1).tolist())
        results[f"task{task_idx + 1}_accuracy"] = float(accuracy_score(truth, pred))
    return results


def local_train(
    global_model: nn.Module,
    train_dataset: Dataset,
    current_indices: np.ndarray,
    replay_indices: Sequence[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    replay_fraction: float,
    seed: int,
) -> nn.Module:
    """Train a client model on current-task data plus sampled replay examples."""
    set_seed(seed)
    local_model = SmallCNN().to(device)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    current_loader = DataLoader(
        Subset(train_dataset, current_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    replay_loader = None
    replay_iter = None
    if replay_indices and replay_fraction > 0:
        replay_loader = DataLoader(
            Subset(train_dataset, list(replay_indices)),
            batch_size=max(1, int(round(batch_size * replay_fraction))),
            shuffle=True,
            num_workers=0,
        )
        replay_iter = iter(replay_loader)

    local_model.train()
    for _ in range(epochs):
        for current_x, current_y in current_loader:
            current_x = current_x.to(device)
            current_y = current_y.reshape(-1).long().to(device)

            if replay_loader is not None and replay_iter is not None:
                try:
                    replay_x, replay_y = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_x, replay_y = next(replay_iter)
                replay_x = replay_x.to(device)
                replay_y = replay_y.reshape(-1).long().to(device)
                batch_x = torch.cat([current_x, replay_x], dim=0)
                batch_y = torch.cat([current_y, replay_y], dim=0)
            else:
                batch_x, batch_y = current_x, current_y

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return local_model


def update_replay_buffers(
    labels: np.ndarray,
    client_partitions: Sequence[np.ndarray],
    current_task_classes: Sequence[int],
    buffers: List[Dict[int, List[int]]],
    memory_per_class: int,
    seed: int,
) -> None:
    """Add up to memory_per_class examples per class/client, preserving old items."""
    rng = np.random.default_rng(seed)
    for client_id, partition in enumerate(client_partitions):
        for class_id in current_task_classes:
            candidates = partition[labels[partition] == class_id]
            existing = buffers[client_id].get(class_id, [])
            remaining = max(0, memory_per_class - len(existing))
            if remaining and len(candidates):
                # Avoid selecting an index already in the buffer.
                available = np.setdiff1d(candidates, np.asarray(existing, dtype=np.int64), assume_unique=False)
                take = min(remaining, len(available))
                if take:
                    selected = rng.choice(available, size=take, replace=False)
                    existing = existing + selected.astype(int).tolist()
            buffers[client_id][class_id] = existing[:memory_per_class]


def run_one_budget(args, memory_per_class: int, train_ds, test_loader, labels, device, out_dir):
    set_seed(args.seed)
    client_parts = make_client_partitions(labels, args.num_clients, args.seed, args.dirichlet_alpha)
    buffers: List[Dict[int, List[int]]] = [dict() for _ in range(args.num_clients)]
    global_model = SmallCNN().to(device)
    model_bytes = state_dict_bytes(global_model)
    seen_classes: List[int] = []
    best_task_accuracy: Dict[int, float] = {}
    history_rows = []
    start_time = time.perf_counter()
    total_rounds = 0

    for task_idx, task_classes in enumerate(TASKS):
        seen_classes.extend(task_classes)
        for round_idx in range(args.rounds_per_task):
            total_rounds += 1
            local_models, client_weights = [], []
            for client_id, partition in enumerate(client_parts):
                current_indices = partition[np.isin(labels[partition], task_classes)]
                replay_indices = [
                    idx
                    for class_id in sorted(buffers[client_id])
                    for idx in buffers[client_id][class_id]
                ]
                local_model = local_train(
                    global_model=global_model,
                    train_dataset=train_ds,
                    current_indices=current_indices,
                    replay_indices=replay_indices,
                    device=device,
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    replay_fraction=args.replay_fraction,
                    seed=args.seed + memory_per_class * 1000 + task_idx * 100 + round_idx * 10 + client_id,
                )
                local_models.append(local_model)
                # FedAvg weighting by the number of current + retained examples.
                client_weights.append(max(1, len(current_indices) + len(replay_indices)))

            total_weight = float(sum(client_weights))
            new_state = {}
            for key in global_model.state_dict():
                new_state[key] = sum(
                    local_models[c].state_dict()[key] * (client_weights[c] / total_weight)
                    for c in range(args.num_clients)
                )
            global_model.load_state_dict(new_state)

            metrics = evaluate(global_model, test_loader, seen_classes, device)
            task_metrics = evaluate_each_task(global_model, test_loader, TASKS, task_idx, device)
            # Track each seen task's best accuracy from the moment it first appears.
            for seen_idx in range(task_idx + 1):
                key = seen_idx + 1
                best_task_accuracy[key] = max(
                    best_task_accuracy.get(key, 0.0),
                    task_metrics[f"task{key}_accuracy"],
                )
            old_forgetting_values = [
                best_task_accuracy[i + 1] - task_metrics[f"task{i + 1}_accuracy"]
                for i in range(task_idx)
            ]
            all_forgetting_values = old_forgetting_values + [0.0]
            row = {
                "memory_per_class_per_client": memory_per_class,
                "task": task_idx + 1,
                "round_in_task": round_idx + 1,
                "global_round": total_rounds,
                "seen_classes": ",".join(map(str, seen_classes)),
                **metrics,
                **task_metrics,
                "mean_forgetting_old_tasks": float(np.mean(old_forgetting_values)) if old_forgetting_values else 0.0,
                "mean_forgetting_all_tasks": float(np.mean(all_forgetting_values)),
                "model_bytes_per_client_update": model_bytes,
                "cumulative_downlink_bytes": total_rounds * args.num_clients * model_bytes,
                "cumulative_uplink_bytes": total_rounds * args.num_clients * model_bytes,
                "cumulative_total_communication_bytes": total_rounds * args.num_clients * model_bytes * 2,
                "buffer_examples_total": sum(len(v) for b in buffers for v in b.values()),
                "process_rss_mib": current_rss_mib(),
                "peak_rss_mib": peak_rss_mib(),
            }
            history_rows.append(row)
            print(
                f"[memory={memory_per_class:>3}] task {task_idx + 1}/3 "
                f"round {round_idx + 1}/{args.rounds_per_task}: "
                f"acc={metrics['accuracy']:.4f}, macroF1={metrics['macro_f1']:.4f}, "
                f"old_forgetting={row['mean_forgetting_old_tasks']:.4f}"
            )

        # After completing a stage, store examples from the task just learned.
        update_replay_buffers(
            labels, client_parts, task_classes, buffers,
            memory_per_class, args.seed + memory_per_class * 100 + task_idx,
        )

    elapsed = time.perf_counter() - start_time
    final = history_rows[-1]
    final_task_acc = {
        f"task{i + 1}_best_accuracy": best_task_accuracy.get(i + 1, 0.0)
        for i in range(len(TASKS))
    }
    final_task_acc.update({
        f"task{i + 1}_final_accuracy": final[f"task{i + 1}_accuracy"]
        for i in range(len(TASKS))
    })
    final_task_acc.update({
        f"task{i + 1}_forgetting": max(
            0.0, final_task_acc[f"task{i + 1}_best_accuracy"] -
            final_task_acc[f"task{i + 1}_final_accuracy"]
        )
        for i in range(len(TASKS))
    })

    summary = {
        "memory_per_class_per_client": memory_per_class,
        "num_clients": args.num_clients,
        "rounds_per_task": args.rounds_per_task,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "replay_fraction": args.replay_fraction,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_mean_forgetting_old_tasks": final["mean_forgetting_old_tasks"],
        "final_mean_forgetting_all_tasks": final["mean_forgetting_all_tasks"],
        **final_task_acc,
        "training_time_seconds": elapsed,
        "peak_rss_mib": max(
            (r["peak_rss_mib"] for r in history_rows if np.isfinite(r["peak_rss_mib"])),
            default=float("nan"),
        ),
        "model_bytes_per_client_update": model_bytes,
        "cumulative_downlink_bytes": final["cumulative_downlink_bytes"],
        "cumulative_uplink_bytes": final["cumulative_uplink_bytes"],
        "cumulative_total_communication_bytes": final["cumulative_total_communication_bytes"],
        "buffer_examples_per_client": [
            sum(len(v) for v in client_buffer.values()) for client_buffer in buffers
        ],
        "buffer_examples_total": sum(
            sum(len(v) for v in client_buffer.values()) for client_buffer in buffers
        ),
        "buffer_index_storage_bytes_estimate": sum(
            sum(len(v) for v in client_buffer.values()) for client_buffer in buffers
        ) * 8,
        "buffer_image_storage_bytes_equivalent_uint8": sum(
            sum(len(v) for v in client_buffer.values()) for client_buffer in buffers
        ) * 28 * 28 * 3,
    }

    budget_dir = out_dir / f"memory_{memory_per_class}"
    budget_dir.mkdir(parents=True, exist_ok=True)

    # Include columns introduced in later continual-learning tasks.
    history_fieldnames = list(dict.fromkeys(
        key for row in history_rows for key in row.keys()
    ))

    # Include columns introduced in later continual-learning tasks.
    with (budget_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history_fieldnames
        )
        writer.writeheader()
        writer.writerows(history_rows)

    with (budget_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    torch.save(global_model.state_dict(), budget_dir / "final_model.pt")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-per-class", nargs="+", type=int, default=[10, 25, 50, 100],
                        help="Replay examples retained per class per client.")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds-per-task", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--replay-fraction", type=float, default=0.25,
                        help="Replay examples per current batch, e.g. 0.25 means ~25 replay per 100 current.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Use <=0 for IID; otherwise Dirichlet class partition alpha.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/fedcl_replay_ablation")
    args = parser.parse_args()

    if any(x < 0 for x in args.memory_per_class):
        parser.error("--memory-per-class values must be >= 0")
    if not 0 <= args.replay_fraction <= 1:
        parser.error("--replay-fraction must be between 0 and 1")
    if args.num_clients < 1 or args.rounds_per_task < 1 or args.local_epochs < 1:
        parser.error("num-clients, rounds-per-task, and local-epochs must be >= 1")

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = INFO["pathmnist"]
    dataset_class = getattr(medmnist, info["python_class"])
    transform = transforms.ToTensor()
    train_ds = dataset_class(split="train", root=str(data_dir), download=True, transform=transform)
    test_ds = dataset_class(split="test", root=str(data_dir), download=True, transform=transform)
    labels = labels_from_dataset(train_ds)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

    print(f"Device: {device}; train={len(train_ds)}; test={len(test_ds)}; clients={args.num_clients}")
    print(f"Tasks: {TASKS}; replay budgets: {args.memory_per_class}; Dirichlet alpha={args.dirichlet_alpha}")
    print("Note: each budget is an independent run with the same partition seed and model initialization seed.")

    summaries = []
    for budget in args.memory_per_class:
        print(f"\n=== Starting replay budget: {budget} examples/class/client ===")
        summary = run_one_budget(args, budget, train_ds, test_loader, labels, device, output_dir)
        summaries.append(summary)

    summary_path = output_dir / "ablation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "ablation_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== Ablation summary ===")
    for s in summaries:
        print(
            f"memory={s['memory_per_class_per_client']:>3}: "
            f"acc={s['final_accuracy']:.4f}, macroF1={s['final_macro_f1']:.4f}, "
            f"balanced={s['final_balanced_accuracy']:.4f}, "
            f"old_forgetting={s['final_mean_forgetting_old_tasks']:.4f}, "
            f"total_comm={s['cumulative_total_communication_bytes']}"
        )
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
