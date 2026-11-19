#!/usr/bin/env python3
"""
Federated continual learning replay-memory ablation for PathMNIST.

Two-phase warm-start comparison for PathMNIST continual federated learning.

Phase 1 (warmup): train Task 1 with full-model FedAvg and save a shared
checkpoint containing the model, client partitions, and local replay buffers.

Phase 2 (continue): load that exact checkpoint and train Tasks 2-3 using
full, classifier-only, or final-convolution-block-plus-classifier updates.

Example:
    # First create the common Task-1 warm-start checkpoint:
    python src/federated/train_fedcl_replay_ablation.py --phase warmup --seed 21

    # Then run each continuation mode from that checkpoint:
    python src/federated/train_fedcl_replay_ablation.py --phase continue \
        --update-mode full --seed 21 \
        --warmstart-checkpoint runs/warmstart_comparison/warmstart_seed_21.pt \
        --output-dir runs/warmstart_comparison/seed_21/full

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
        # Shuffle all examples before splitting so each client gets a random,
        # approximately equal-sized share of the training set.
        indices = rng.permutation(len(labels))
        return [np.asarray(x, dtype=np.int64) for x in np.array_split(indices, num_clients)]

    # Split each class independently. Small alpha values make its samples
    # concentrate on fewer clients, creating stronger class imbalance.
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


def get_trainable_names(model: nn.Module, update_mode: str) -> List[str]:
    """Return parameter names selected for local training."""
    if update_mode == "full":
        return [
            name for name, _ in model.named_parameters()
        ]

    if update_mode == "head":
        prefixes = ("classifier.",)

    elif update_mode == "block_head":
        prefixes = ("features.6.", "classifier.")

    else:
        raise ValueError(f"Unknown update mode: {update_mode}")

    return [
        name for name, _ in model.named_parameters()
        if name.startswith(prefixes)
    ]


def get_update_state_keys(
    model: nn.Module,
    update_mode: str
) -> List[str]:
    """Return state-dict keys that must be aggregated and communicated."""
    return get_trainable_names(model, update_mode)


def configure_trainable_parameters(
    model: nn.Module,
    update_mode: str
) -> List[str]:
    """Freeze unselected parameters and return the trainable names."""
    trainable_names = get_trainable_names(model, update_mode)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in trainable_names

    return trainable_names


def selected_state_bytes(
    model: nn.Module,
    selected_keys: Sequence[str]
) -> int:
    state = model.state_dict()
    return sum(
        state[key].numel() * state[key].element_size()
        for key in selected_keys
    )


def selected_parameter_count(
    model: nn.Module,
    selected_names: Sequence[str]
) -> int:
    parameters = dict(model.named_parameters())
    return sum(
        parameters[name].numel()
        for name in selected_names
    )



def current_rss_mib() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def peak_rss_mib() -> float:
    """Best-effort process peak RSS; Linux reports ru_maxrss in KiB."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB while macOS reports bytes; normalize both to MiB.
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
            # Compare only classes learned so far so untrained future logits
            # cannot count as classification errors.
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
    """Evaluate each task while allowing predictions over all seen classes."""
    results: Dict[str, float] = {}
    model.eval()

    cached = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device)).cpu().numpy()
            labels = labels.reshape(-1).numpy().astype(int)
            cached.append((logits, labels))

    # All classes learned up to the current stage are valid predictions.
    seen_classes = [
        c
        for task in tasks[:completed_task + 1]
        for c in task
    ]

    for task_idx in range(completed_task + 1):
        task_classes = list(tasks[task_idx])
        truth, pred = [], []

        for logits, labels in cached:
            mask = np.isin(labels, task_classes)
            if not np.any(mask):
                continue

            task_logits = logits[mask].copy()

            # Mask only future/unseen classes, not other seen tasks.
            task_logits[:, [
                c for c in range(9) if c not in seen_classes
            ]] = -1e9

            truth.extend(labels[mask].tolist())
            pred.extend(task_logits.argmax(axis=1).tolist())

        results[f"task{task_idx + 1}_accuracy"] = float(
            accuracy_score(truth, pred)
        )

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
    update_mode: str,
) -> nn.Module:
    """Train a client model on current-task data plus sampled replay examples."""
    set_seed(seed)
    local_model = SmallCNN().to(device)
    local_model.load_state_dict(global_model.state_dict())
    trainable_names = configure_trainable_parameters(
        local_model, update_mode
    )

    trainable_parameters = [
        parameter
        for parameter in local_model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
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
            # Replay is an additional mini-batch, sized relative to the
            # current-data batch, and concatenated with it below.
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
                    # Keep replay available for every current batch, even when
                    # the replay loader has fewer batches in an epoch.
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
            # Existing exemplars are retained; only the remaining per-class
            # budget can be filled from this client's current-task examples.
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



def run_warmup(args, memory_per_class, train_ds, test_loader, labels, device, out_dir):
    """Train Task 1 with full FedAvg updates and save a reusable warm-start."""
    set_seed(args.seed)
    client_parts = make_client_partitions(
        labels, args.num_clients, args.seed, args.dirichlet_alpha
    )
    buffers: List[Dict[int, List[int]]] = [dict() for _ in range(args.num_clients)]
    global_model = SmallCNN().to(device)
    full_model_bytes = state_dict_bytes(global_model)
    update_keys = get_update_state_keys(global_model, "full")
    update_bytes = selected_state_bytes(global_model, update_keys)

    history_rows = []
    start_time = time.perf_counter()
    total_rounds = 0
    task_classes = TASKS[0]

    for round_idx in range(args.rounds_per_task):
        total_rounds += 1
        local_models, client_weights = [], []
        for client_id, partition in enumerate(client_parts):
            current_indices = partition[np.isin(labels[partition], task_classes)]
            local_model = local_train(
                global_model=global_model,
                train_dataset=train_ds,
                current_indices=current_indices,
                replay_indices=[],
                device=device,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                replay_fraction=args.replay_fraction,
                seed=args.seed + memory_per_class * 1000 + client_id * 10 + round_idx,
                update_mode="full",
            )
            local_models.append(local_model)
            client_weights.append(max(1, len(current_indices)))

        total_weight = float(sum(client_weights))
        aggregated = {
            key: sum(
                local_models[c].state_dict()[key] *
                (client_weights[c] / total_weight)
                for c in range(args.num_clients)
            )
            for key in update_keys
        }
        global_model.load_state_dict(aggregated, strict=False)

        metrics = evaluate(global_model, test_loader, task_classes, device)
        task_metrics = evaluate_each_task(global_model, test_loader, TASKS, 0, device)
        row = {
            "phase": "warmup",
            "task": 1,
            "round_in_task": round_idx + 1,
            "global_round": round_idx + 1,
            **metrics,
            **task_metrics,
            "update_mode": "full",
            "trainable_parameters": sum(p.numel() for p in global_model.parameters()),
            "full_model_bytes": full_model_bytes,
            "model_bytes_per_client_update": update_bytes,
            "cumulative_downlink_bytes": total_rounds * args.num_clients * update_bytes,
            "cumulative_uplink_bytes": total_rounds * args.num_clients * update_bytes,
            "cumulative_total_communication_bytes": (
                total_rounds * args.num_clients * update_bytes * 2
            ),
            "process_rss_mib": current_rss_mib(),
            "peak_rss_mib": peak_rss_mib(),
        }
        history_rows.append(row)
        print(
            f"[warmup][seed={args.seed}] task 1 round "
            f"{round_idx + 1}/{args.rounds_per_task}: "
            f"acc={metrics['accuracy']:.4f}, macroF1={metrics['macro_f1']:.4f}"
        )

    # Populate Task-1 exemplars before saving so all branches inherit identical buffers.
    update_replay_buffers(
        labels, client_parts, task_classes, buffers,
        memory_per_class, args.seed + memory_per_class * 100,
    )

    elapsed = time.perf_counter() - start_time
    final_metrics = history_rows[-1]
    warmup_summary = {
        "seed": args.seed,
        "memory_per_class_per_client": memory_per_class,
        "task1_accuracy": final_metrics["accuracy"],
        "task1_macro_f1": final_metrics["macro_f1"],
        "task1_balanced_accuracy": final_metrics["balanced_accuracy"],
        "warmup_training_time_seconds": elapsed,
        "warmup_rounds": args.rounds_per_task,
        "warmup_communication_bytes": (
            args.rounds_per_task * args.num_clients * update_bytes * 2
        ),
        "initial_model_broadcast_bytes": args.num_clients * full_model_bytes,
        "full_model_bytes": full_model_bytes,
        "buffer_examples_per_client": [
            sum(len(v) for v in b.values()) for b in buffers
        ],
        "buffer_examples_total": sum(sum(len(v) for v in b.values()) for b in buffers),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"warmstart_seed_{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "client_memories": buffers,
            "client_partitions": client_parts,
            "warmup_summary": warmup_summary,
            "seed": args.seed,
            "memory_per_class_per_client": memory_per_class,
            "num_clients": args.num_clients,
            "dirichlet_alpha": args.dirichlet_alpha,
            "rounds_per_task": args.rounds_per_task,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "replay_fraction": args.replay_fraction,
        },
        checkpoint_path,
    )

    with (out_dir / "warmup_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    with (out_dir / "warmup_metrics.json").open("w") as f:
        json.dump(warmup_summary, f, indent=2)

    print(f"Saved Task-1 warm-start checkpoint: {checkpoint_path}")
    return warmup_summary


def run_continuation(args, memory_per_class, train_ds, test_loader, labels, device, out_dir):
    """Load a shared Task-1 checkpoint and continue Tasks 2-3 in one update mode."""
    if not args.warmstart_checkpoint:
        raise ValueError("--warmstart-checkpoint is required for --phase continue")

    # Checkpoint contains model, partition and buffers, ensuring all branches for
    # a seed start from exactly the same state.
    try:
        checkpoint = torch.load(
            args.warmstart_checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(args.warmstart_checkpoint, map_location="cpu")

    if checkpoint["seed"] != args.seed:
        raise ValueError(
            f"Checkpoint seed {checkpoint['seed']} does not match --seed {args.seed}"
        )
    if checkpoint["memory_per_class_per_client"] != memory_per_class:
        raise ValueError("Checkpoint replay budget does not match --memory-per-class")
    if checkpoint["num_clients"] != args.num_clients:
        raise ValueError("Checkpoint client count does not match --num-clients")
    if not np.isclose(checkpoint["dirichlet_alpha"], args.dirichlet_alpha):
        raise ValueError("Checkpoint Dirichlet alpha does not match --dirichlet-alpha")
    # Keep the continuation protocol identical to the warm-up run.
    for checkpoint_key, arg_value, option_name in (
        ("rounds_per_task", args.rounds_per_task, "--rounds-per-task"),
        ("local_epochs", args.local_epochs, "--local-epochs"),
        ("batch_size", args.batch_size, "--batch-size"),
        ("replay_fraction", args.replay_fraction, "--replay-fraction"),
        ("learning_rate", args.lr, "--lr"),
    ):
        saved_value = checkpoint[checkpoint_key]
        if isinstance(arg_value, float):
            matches = np.isclose(saved_value, arg_value)
        else:
            matches = saved_value == arg_value
        if not matches:
            raise ValueError(
                f"Checkpoint {checkpoint_key}={saved_value} does not match "
                f"{option_name}={arg_value}"
            )

    set_seed(args.seed)
    client_parts = checkpoint["client_partitions"]
    buffers = checkpoint["client_memories"]
    global_model = SmallCNN().to(device)
    global_model.load_state_dict(checkpoint["model_state_dict"])

    full_model_bytes = state_dict_bytes(global_model)
    trainable_names = get_trainable_names(global_model, args.update_mode)
    update_keys = get_update_state_keys(global_model, args.update_mode)
    update_bytes = selected_state_bytes(global_model, update_keys)
    trainable_params = selected_parameter_count(global_model, trainable_names)
    total_params = sum(p.numel() for p in global_model.parameters())

    # Seed forgetting tracking with the Task-1 score at the end of warm-up.
    initial_task1 = evaluate_each_task(global_model, test_loader, TASKS, 0, device)
    best_task_accuracy: Dict[int, float] = {
        1: initial_task1["task1_accuracy"]
    }
    seen_classes = list(TASKS[0])
    history_rows = []
    start_time = time.perf_counter()
    continuation_rounds = 0

    for task_idx in (1, 2):
        task_classes = TASKS[task_idx]
        seen_classes.extend(task_classes)
        for round_idx in range(args.rounds_per_task):
            continuation_rounds += 1
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
                    seed=args.seed + memory_per_class * 1000
                    + task_idx * 100 + round_idx * 10 + client_id,
                    update_mode=args.update_mode,
                )
                local_models.append(local_model)
                client_weights.append(max(1, len(current_indices) + len(replay_indices)))

            total_weight = float(sum(client_weights))
            aggregated = {
                key: sum(
                    local_models[c].state_dict()[key] *
                    (client_weights[c] / total_weight)
                    for c in range(args.num_clients)
                )
                for key in update_keys
            }
            global_model.load_state_dict(aggregated, strict=False)

            metrics = evaluate(global_model, test_loader, seen_classes, device)
            task_metrics = evaluate_each_task(
                global_model, test_loader, TASKS, task_idx, device
            )
            for seen_idx in range(task_idx + 1):
                task_number = seen_idx + 1
                best_task_accuracy[task_number] = max(
                    best_task_accuracy.get(task_number, 0.0),
                    task_metrics[f"task{task_number}_accuracy"],
                )
            old_forgetting_values = [
                best_task_accuracy[i + 1] - task_metrics[f"task{i + 1}_accuracy"]
                for i in range(task_idx + 1)
            ]
            row = {
                "phase": "continuation",
                "update_mode": args.update_mode,
                "memory_per_class_per_client": memory_per_class,
                "task": task_idx + 1,
                "round_in_task": round_idx + 1,
                "continuation_round": continuation_rounds,
                "seen_classes": ",".join(map(str, seen_classes)),
                **metrics,
                **task_metrics,
                "mean_forgetting_old_tasks": float(np.mean(old_forgetting_values)),
                "trainable_parameters": trainable_params,
                "trainable_parameter_fraction": trainable_params / total_params,
                "full_model_bytes": full_model_bytes,
                "model_bytes_per_client_update": update_bytes,
                "continuation_downlink_bytes": (
                    continuation_rounds * args.num_clients * update_bytes
                ),
                "continuation_uplink_bytes": (
                    continuation_rounds * args.num_clients * update_bytes
                ),
                "continuation_communication_bytes": (
                    continuation_rounds * args.num_clients * update_bytes * 2
                ),
                "buffer_examples_total": sum(len(v) for b in buffers for v in b.values()),
                "process_rss_mib": current_rss_mib(),
                "peak_rss_mib": peak_rss_mib(),
            }
            history_rows.append(row)
            print(
                f"[{args.update_mode}][seed={args.seed}] task {task_idx + 1}/3 "
                f"round {round_idx + 1}/{args.rounds_per_task}: "
                f"acc={metrics['accuracy']:.4f}, macroF1={metrics['macro_f1']:.4f}, "
                f"old_forgetting={row['mean_forgetting_old_tasks']:.4f}"
            )

        update_replay_buffers(
            labels, client_parts, task_classes, buffers,
            memory_per_class, args.seed + memory_per_class * 100 + task_idx,
        )

    elapsed = time.perf_counter() - start_time
    final = history_rows[-1]
    final_task_metrics = evaluate_each_task(global_model, test_loader, TASKS, 2, device)
    task1_final = final_task_metrics["task1_accuracy"]
    task2_final = final_task_metrics["task2_accuracy"]
    task3_final = final_task_metrics["task3_accuracy"]
    task_best = {
        "task1_best_accuracy": best_task_accuracy[1],
        "task2_best_accuracy": best_task_accuracy.get(2, 0.0),
        "task3_best_accuracy": best_task_accuracy.get(3, 0.0),
    }
    task_final = {
        "task1_final_accuracy": task1_final,
        "task2_final_accuracy": task2_final,
        "task3_final_accuracy": task3_final,
    }
    task_forgetting = {
        f"task{i}_forgetting": max(
            0.0, task_best[f"task{i}_best_accuracy"] - task_final[f"task{i}_final_accuracy"]
        )
        for i in (1, 2, 3)
    }

    warmup = checkpoint["warmup_summary"]
    continuation_comm = final["continuation_communication_bytes"]
    warmup_comm = warmup["warmup_communication_bytes"]
    initial_broadcast = warmup["initial_model_broadcast_bytes"]
    summary = {
        "phase": "continuation",
        "update_mode": args.update_mode,
        "seed": args.seed,
        "memory_per_class_per_client": memory_per_class,
        "num_clients": args.num_clients,
        "rounds_per_task": args.rounds_per_task,
        "continuation_rounds": 2 * args.rounds_per_task,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "replay_fraction": args.replay_fraction,
        "dirichlet_alpha": args.dirichlet_alpha,
        "warmup_task1_accuracy": warmup["task1_accuracy"],
        "warmup_task1_macro_f1": warmup["task1_macro_f1"],
        "warmup_training_time_seconds": warmup["warmup_training_time_seconds"],
        "continuation_training_time_seconds": elapsed,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_mean_forgetting_old_tasks": final["mean_forgetting_old_tasks"],
        **task_best,
        **task_final,
        **task_forgetting,
        "trainable_parameters": trainable_params,
        "trainable_parameter_fraction": trainable_params / total_params,
        "full_model_bytes": full_model_bytes,
        "update_bytes_per_client_per_round": update_bytes,
        "continuation_communication_bytes": continuation_comm,
        "warmup_communication_bytes_common_to_all_modes": warmup_comm,
        "initial_model_broadcast_bytes_common_to_all_modes": initial_broadcast,
        "total_communication_including_warmup_and_initial_broadcast": (
            continuation_comm + warmup_comm + initial_broadcast
        ),
        "peak_rss_mib": max(
            (r["peak_rss_mib"] for r in history_rows if np.isfinite(r["peak_rss_mib"])),
            default=float("nan"),
        ),
        "buffer_examples_per_client": [
            sum(len(v) for v in b.values()) for b in buffers
        ],
        "buffer_examples_total": sum(sum(len(v) for v in b.values()) for b in buffers),
        "buffer_index_storage_bytes_estimate": (
            sum(sum(len(v) for v in b.values()) for b in buffers) * 8
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        fieldnames = list(dict.fromkeys(k for row in history_rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    torch.save(global_model.state_dict(), out_dir / "final_model.pt")
    return summary



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["warmup", "continue"], default="warmup",
        help="warmup trains Task 1 with full FedAvg; continue loads that checkpoint for Tasks 2-3.",
    )
    parser.add_argument("--memory-per-class", nargs=1, type=int, default=[10],
                        help="Replay examples retained per class per client (warm-start uses one budget).")
    parser.add_argument("--update-mode", choices=["full", "head", "block_head"], default="full",
                        help="Continuation update mode. Warm-up always uses full-model updates.")
    parser.add_argument("--warmstart-checkpoint", type=str, default=None,
                        help="Checkpoint produced by --phase warmup; required for --phase continue.")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds-per-task", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/warmstart_comparison")
    args = parser.parse_args()

    memory_per_class = args.memory_per_class[0]
    if memory_per_class < 0:
        parser.error("--memory-per-class must be >= 0")
    if not 0 <= args.replay_fraction <= 1:
        parser.error("--replay-fraction must be between 0 and 1")
    if args.num_clients < 1 or args.rounds_per_task < 1 or args.local_epochs < 1:
        parser.error("num-clients, rounds-per-task, and local-epochs must be >= 1")
    if args.phase == "continue" and not args.warmstart_checkpoint:
        parser.error("--warmstart-checkpoint is required when --phase continue")

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
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        f"Phase: {args.phase}; device={device}; train={len(train_ds)}; "
        f"test={len(test_ds)}; clients={args.num_clients}; seed={args.seed}"
    )
    print(
        f"Replay budget={memory_per_class}/class/client; tasks={TASKS}; "
        f"Dirichlet alpha={args.dirichlet_alpha}; update mode={args.update_mode}"
    )

    if args.phase == "warmup":
        run_warmup(
            args, memory_per_class, train_ds, test_loader, labels, device, output_dir
        )
    else:
        summary = run_continuation(
            args, memory_per_class, train_ds, test_loader, labels, device, output_dir
        )
        print("\n=== Continuation summary ===")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved continuation results to: {output_dir}")


if __name__ == "__main__":
    main()
