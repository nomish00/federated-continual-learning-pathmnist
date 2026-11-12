
#!/usr/bin/env python3
"""
Federated continual learning baseline on PathMNIST.

- 3 simulated IID clients
- 3 sequential tasks: classes [0,1,2], [3,4,5], [6,7,8]
- FedAvg aggregation
- No replay buffer or continual-learning regularisation
- Evaluation on all tasks seen so far
- Future-class logits are masked during evaluation

Run from the project root:
    python src/federated/train_fedcl_iid.py --rounds-per-task 1

Full experiment:
    python src/federated/train_fedcl_iid.py --rounds-per-task 5
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST


TASKS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallCNN(nn.Module):
    """Compact CNN for 28x28 RGB PathMNIST images."""

    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_labels(dataset):
    """Return dataset labels as a one-dimensional NumPy array."""
    return np.asarray(dataset.labels).reshape(-1).astype(int)


def make_iid_client_partitions(indices, num_clients, rng):
    """Shuffle task indices and split them approximately equally."""
    shuffled = np.asarray(indices).copy()
    rng.shuffle(shuffled)

    splits = np.array_split(shuffled, num_clients)
    return [split.tolist() for split in splits]


def train_one_client(
    global_state,
    model,
    train_dataset,
    client_indices,
    device,
    batch_size,
    local_epochs,
    learning_rate,
):
    """Train one client from the same global model state."""

    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    loader = DataLoader(
        Subset(train_dataset, client_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0

    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.reshape(-1).long().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_examples += batch_count

    average_loss = (
        total_loss / total_examples
        if total_examples > 0
        else 0.0
    )

    return (
        copy.deepcopy(model.state_dict()),
        average_loss,
        len(client_indices),
    )


def fedavg(client_states, client_sizes):
    """Sample-weighted FedAvg aggregation."""
    total_size = sum(client_sizes)

    if total_size == 0:
        raise ValueError("No client training examples were provided.")

    global_state = copy.deepcopy(client_states[0])

    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key])

        for client_state, client_size in zip(
            client_states, client_sizes
        ):
            weight = client_size / total_size
            global_state[key] += (
                client_state[key] * weight
            )

    return global_state


@torch.no_grad()
def evaluate(
    model,
    dataset,
    indices,
    seen_classes,
    device,
    batch_size,
):
    """
    Evaluate selected examples.

    Logits for classes not introduced yet are masked. This avoids
    penalising the model for predicting future classes.
    """

    if len(indices) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
            "n_examples": 0,
        }

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []

    allowed_classes = torch.tensor(
        seen_classes,
        dtype=torch.long,
        device=device,
    )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.reshape(-1).long().to(device)

        logits = model(images)

        # Keep only classes that have been introduced so far.
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[:, allowed_classes] = (
            logits[:, allowed_classes]
        )

        predictions = masked_logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )
        all_targets.extend(labels.cpu().numpy().tolist())

    return {
        "accuracy": float(
            accuracy_score(all_targets, all_predictions)
        ),
        "macro_f1": float(
            f1_score(
                all_targets,
                all_predictions,
                labels=seen_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                all_targets, all_predictions
            )
        ),
        "n_examples": len(all_targets),
    }


def get_peak_ram_mib():
    """Return peak process RAM in MiB on Linux."""
    try:
        import resource

        peak = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss

        # Linux reports ru_maxrss in KiB.
        return float(peak / 1024.0)

    except Exception:
        return None


def calculate_forgetting(history, current_stage, current_accuracies):
    """
    Calculate mean forgetting over previously introduced tasks.

    For each old task:
        forgetting = best accuracy observed since introduction
                    - current accuracy

    The current stage's measurement is included in the best value,
    so forgetting cannot be negative.
    """

    old_task_forgetting = []

    for task_index in range(current_stage):
        task_key = f"task_{task_index + 1}_accuracy"
        current_accuracy = current_accuracies.get(task_key)

        if current_accuracy is None:
            continue

        previous_values = [
            row[task_key]
            for row in history
            if (
                row["stage"] >= task_index + 1
                and row[task_key] is not None
            )
        ]

        observed_values = previous_values + [
            current_accuracy
        ]

        best_accuracy = max(observed_values)

        old_task_forgetting.append(
            best_accuracy - current_accuracy
        )

    if not old_task_forgetting:
        return 0.0

    return float(np.mean(old_task_forgetting))


def main():
    parser = argparse.ArgumentParser(
        description="FedCL IID experiment on PathMNIST"
    )

    parser.add_argument(
        "--rounds-per-task",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/fedcl_iid",
    )

    args = parser.parse_args()

    if args.rounds_per_task < 1:
        parser.error("--rounds-per-task must be at least 1")

    if args.num_clients < 1:
        parser.error("--num-clients must be at least 1")

    if args.local_epochs < 1:
        parser.error("--local-epochs must be at least 1")

    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PathMNIST...")
    transform = transforms.ToTensor()

    train_dataset = PathMNIST(
        split="train",
        transform=transform,
        download=True,
    )

    test_dataset = PathMNIST(
        split="test",
        transform=transform,
        download=True,
    )

    train_labels = get_labels(train_dataset)
    test_labels = get_labels(test_dataset)

    print(
        f"Train samples: {len(train_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )
    print(
        f"Clients: {args.num_clients} | "
        f"Tasks: {len(TASKS)} | Device: {device}"
    )

    # Find test indices for each task.
    test_indices_by_task = {}

    for task_index, task_classes in enumerate(TASKS):
        test_indices_by_task[task_index] = np.where(
            np.isin(test_labels, task_classes)
        )[0].tolist()

    model = SmallCNN(num_classes=9).to(device)
    global_state = copy.deepcopy(model.state_dict())

    # Raw tensor payload size, excluding protocol overhead.
    model_payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_state.values()
    )

    cumulative_downlink_bytes = 0
    cumulative_uplink_bytes = 0

    history = []
    task_best_accuracy = {
        task_index: None
        for task_index in range(len(TASKS))
    }

    rng = np.random.default_rng(args.seed)
    experiment_start = time.time()

    for stage, current_task_classes in enumerate(TASKS, start=1):
        seen_classes = [
            class_id
            for task in TASKS[:stage]
            for class_id in task
        ]

        print("\n" + "=" * 65)
        print(
            f"STAGE {stage}/{len(TASKS)} | "
            f"New classes: {current_task_classes} | "
            f"Seen classes: {seen_classes}"
        )
        print("=" * 65)

        # Select training examples belonging to this task only.
        task_train_indices = np.where(
            np.isin(train_labels, current_task_classes)
        )[0]

        client_partitions = make_iid_client_partitions(
            task_train_indices,
            args.num_clients,
            rng,
        )

        for client_id, client_indices in enumerate(
            client_partitions, start=1
        ):
            print(
                f"Client {client_id}: "
                f"{len(client_indices)} training examples"
            )

        for round_number in range(
            1, args.rounds_per_task + 1
        ):
            round_start = time.time()

            client_states = []
            client_sizes = []
            client_losses = []

            # Every client receives the current global model.
            # Each client trains only on the current task.
            for client_id, client_indices in enumerate(
                client_partitions, start=1
            ):
                client_model = SmallCNN(num_classes=9)

                client_state, client_loss, client_size = (
                    train_one_client(
                        global_state=global_state,
                        model=client_model,
                        train_dataset=train_dataset,
                        client_indices=client_indices,
                        device=device,
                        batch_size=args.batch_size,
                        local_epochs=args.local_epochs,
                        learning_rate=args.learning_rate,
                    )
                )

                client_states.append(client_state)
                client_sizes.append(client_size)
                client_losses.append(client_loss)

            # Server aggregates client model parameters.
            global_state = fedavg(
                client_states,
                client_sizes,
            )

            # Count one model download and one model upload
            # for every client in this communication round.
            round_downlink = (
                args.num_clients * model_payload_bytes
            )
            round_uplink = (
                args.num_clients * model_payload_bytes
            )

            cumulative_downlink_bytes += round_downlink
            cumulative_uplink_bytes += round_uplink

            model.load_state_dict(global_state)
            model.eval()

            # Evaluate every task introduced so far.
            task_accuracies = {}
            task_results = {}

            for task_index in range(stage):
                task_test_indices = (
                    test_indices_by_task[task_index]
                )

                result = evaluate(
                    model=model,
                    dataset=test_dataset,
                    indices=task_test_indices,
                    seen_classes=seen_classes,
                    device=device,
                    batch_size=args.batch_size,
                )

                task_key = f"task_{task_index + 1}_accuracy"
                task_accuracies[task_key] = result["accuracy"]
                task_results[task_index] = result

                previous_best = task_best_accuracy[task_index]

                if previous_best is None:
                    task_best_accuracy[task_index] = (
                        result["accuracy"]
                    )
                else:
                    task_best_accuracy[task_index] = max(
                        previous_best,
                        result["accuracy"],
                    )

            # Combined evaluation across all classes seen so far.
            combined_test_indices = np.where(
                np.isin(test_labels, seen_classes)
            )[0].tolist()

            combined_result = evaluate(
                model=model,
                dataset=test_dataset,
                indices=combined_test_indices,
                seen_classes=seen_classes,
                device=device,
                batch_size=args.batch_size,
            )

            mean_forgetting = calculate_forgetting(
                history=history,
                current_stage=stage - 1,
                current_accuracies=task_accuracies,
            )

            round_seconds = time.time() - round_start

            row = {
                "stage": stage,
                "round": round_number,
                "new_classes": ",".join(
                    str(x) for x in current_task_classes
                ),
                "seen_classes": ",".join(
                    str(x) for x in seen_classes
                ),
                "client_loss": float(np.average(
                    client_losses,
                    weights=client_sizes,
                )),
                "combined_accuracy": combined_result["accuracy"],
                "combined_macro_f1": combined_result["macro_f1"],
                "combined_balanced_accuracy": (
                    combined_result["balanced_accuracy"]
                ),
                "mean_forgetting": mean_forgetting,
                "round_seconds": round_seconds,
                "peak_ram_mib": get_peak_ram_mib(),
                "model_payload_bytes_per_client": (
                    model_payload_bytes
                ),
                "round_downlink_bytes": round_downlink,
                "round_uplink_bytes": round_uplink,
                "cumulative_downlink_bytes": (
                    cumulative_downlink_bytes
                ),
                "cumulative_uplink_bytes": (
                    cumulative_uplink_bytes
                ),
            }

            for task_key, accuracy in task_accuracies.items():
                row[task_key] = accuracy

            history.append(row)

            task_summary = " | ".join(
                f"T{i + 1}={task_results[i]['accuracy']:.4f}"
                for i in range(stage)
            )

            print(
                f"Stage {stage}/{len(TASKS)} "
                f"Round {round_number}/{args.rounds_per_task} | "
                f"loss={row['client_loss']:.4f} | "
                f"seen_acc={row['combined_accuracy']:.4f} | "
                f"macro_f1={row['combined_macro_f1']:.4f} | "
                f"balanced_acc="
                f"{row['combined_balanced_accuracy']:.4f} | "
                f"forgetting={mean_forgetting:.4f}"
            )
            print(f"    Task accuracies: {task_summary}")

            # Save progress after every round.
            pd.DataFrame(history).to_csv(
                output_dir / "history.csv",
                index=False,
            )

    total_seconds = time.time() - experiment_start

    # Final forgetting: best accuracy for each task minus its
    # accuracy at the end of the final stage.
    final_row = history[-1]
    final_task_accuracies = {}

    for task_index in range(len(TASKS)):
        task_key = f"task_{task_index + 1}_accuracy"
        final_accuracy = final_row.get(task_key)

        if final_accuracy is None:
            continue

        best_accuracy = task_best_accuracy[task_index]

        final_task_accuracies[task_key] = {
            "best_accuracy_since_introduction": best_accuracy,
            "final_accuracy": final_accuracy,
            "forgetting": float(
                best_accuracy - final_accuracy
            ),
        }

    final_forgetting_values = [
        item["forgetting"]
        for item in final_task_accuracies.values()
        if item["forgetting"] is not None
    ]

    final_mean_forgetting = (
        float(np.mean(final_forgetting_values))
        if final_forgetting_values
        else 0.0
    )

    model.load_state_dict(global_state)
    torch.save(
        model.state_dict(),
        output_dir / "fedcl_global_model.pt",
    )

    metrics = {
        "experiment": "FedCL IID baseline",
        "dataset": "PathMNIST",
        "num_clients": args.num_clients,
        "tasks": TASKS,
        "rounds_per_task": args.rounds_per_task,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "final_combined_accuracy": (
            final_row["combined_accuracy"]
        ),
        "final_combined_macro_f1": (
            final_row["combined_macro_f1"]
        ),
        "final_combined_balanced_accuracy": (
            final_row["combined_balanced_accuracy"]
        ),
        "final_mean_forgetting": final_mean_forgetting,
        "final_task_accuracies_and_forgetting": (
            final_task_accuracies
        ),
        "training_seconds": total_seconds,
        "peak_ram_mib": get_peak_ram_mib(),
        "model_payload_bytes_per_client": (
            model_payload_bytes
        ),
        "cumulative_downlink_bytes": (
            cumulative_downlink_bytes
        ),
        "cumulative_uplink_bytes": (
            cumulative_uplink_bytes
        ),
        "cumulative_total_communication_bytes": (
            cumulative_downlink_bytes
            + cumulative_uplink_bytes
        ),
        "communication_note": (
            "Raw model tensor payload estimate. Excludes "
            "protocol headers, metadata, compression, encryption "
            "and transport overhead."
        ),
    }

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print(
        f"Final seen-class accuracy: "
        f"{metrics['final_combined_accuracy']:.4f}"
    )
    print(
        f"Final macro-F1: "
        f"{metrics['final_combined_macro_f1']:.4f}"
    )
    print(
        f"Final balanced accuracy: "
        f"{metrics['final_combined_balanced_accuracy']:.4f}"
    )
    print(
        f"Final mean forgetting: "
        f"{metrics['final_mean_forgetting']:.4f}"
    )
    print(f"Training time: {total_seconds:.2f} seconds")
    print(
        "Total communication: "
        f"{metrics['cumulative_total_communication_bytes']:,} bytes"
    )
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
