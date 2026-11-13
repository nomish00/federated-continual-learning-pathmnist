
#!/usr/bin/env python3

"""
Federated Continual Learning with Experience Replay
===================================================

Dataset: PathMNIST
Clients: 3 (simulated)
Tasks:
    Task 1: classes 0, 1, 2
    Task 2: classes 3, 4, 5
    Task 3: classes 6, 7, 8

Method:
    - Each client trains on its current task.
    - Each client maintains a local replay buffer.
    - During later tasks, local batches combine current data
      with examples sampled from the replay buffer.
    - The server aggregates client models using FedAvg.

The model architecture is imported from train_central.py
to ensure comparability with the previous experiments.
"""

import argparse
import copy
import csv
import json
import random
import resource
import sys
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from medmnist import PathMNIST
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

# Make src/ importable when this script is run from the project root.
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from train_central import SmallCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TASKS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------

def get_peak_ram_mib():
    """
    Linux ru_maxrss is reported in KiB.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)

    return usage.ru_maxrss / 1024.0


# ---------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------

def get_targets(dataset):
    return np.asarray(dataset.labels).reshape(-1).astype(int)


def create_client_partitions(indices, num_clients, rng):
    """
    Random IID partition of the supplied indices.

    Each client receives approximately the same number
    of examples.
    """
    shuffled = rng.permutation(indices)

    partitions = np.array_split(shuffled, num_clients)

    return [
        partition.astype(int).tolist()
        for partition in partitions
    ]


def create_task_partitions(
    train_targets,
    task_classes,
    num_clients,
    rng,
):
    """
    Select current-task examples and distribute them
    randomly across clients.
    """
    indices = np.where(
        np.isin(train_targets, task_classes)
    )[0]

    return create_client_partitions(
        indices,
        num_clients,
        rng,
    )


# ---------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------

def update_replay_buffer(
    memory,
    current_indices,
    train_targets,
    task_classes,
    memory_per_class,
    rng,
):
    """
    Add examples from the newly learned task.

    The buffer is maintained separately for each class.
    Existing examples from previous tasks are retained.

    memory format:
        {
            class_id: [dataset_index, dataset_index, ...]
        }
    """

    for class_id in task_classes:

        # This helper is called once after a stage, so the selected items
        # become fixed replay exemplars for later stages.
        class_indices = [
            idx
            for idx in current_indices
            if train_targets[idx] == class_id
        ]

        if not class_indices:
            continue

        number_to_store = min(
            memory_per_class,
            len(class_indices),
        )

        selected = rng.choice(
            class_indices,
            size=number_to_store,
            replace=False,
        )

        memory[class_id] = [
            int(idx) for idx in selected
        ]

    return memory


def get_replay_indices(memory):
    """
    Return all stored examples as a flat list.
    """
    indices = []

    for class_indices in memory.values():
        indices.extend(class_indices)

    return indices


# ---------------------------------------------------------
# Local training
# ---------------------------------------------------------

def train_local_model(
    global_state,
    train_dataset,
    current_indices,
    replay_indices,
    device,
    local_epochs,
    batch_size,
    learning_rate,
    replay_fraction,
):
    """
    Train one client model.

    If replay data is available, each batch contains:
        - current-task examples
        - replay examples

    Replay fraction controls the proportion of the batch
    allocated to replay examples.

    The global model is used as the starting point.
    """

    model = SmallCNN().to(device)
    model.load_state_dict(global_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    if replay_indices:

        # Replay occupies a fraction of each combined batch; current examples
        # use the remaining slots. With no memory, use a full current-data batch.
        replay_batch_size = max(
            1,
            int(round(batch_size * replay_fraction)),
        )

        current_batch_size = (
            batch_size - replay_batch_size
        )

    else:
        current_batch_size = batch_size
        replay_batch_size = 0

    current_loader = DataLoader(
        Subset(train_dataset, current_indices),
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=0,
    )

    if replay_indices:

        replay_loader = DataLoader(
            Subset(train_dataset, replay_indices),
            batch_size=replay_batch_size,
            shuffle=True,
            num_workers=0,
        )

        replay_iterator = iter(replay_loader)

    else:
        replay_loader = None
        replay_iterator = None

    model.train()

    total_loss = 0.0
    total_examples = 0

    for _ in range(local_epochs):

        for current_images, current_labels in current_loader:

            current_labels = (
                current_labels.long().view(-1)
            )

            if replay_loader is not None:

                try:
                    replay_images, replay_labels = next(
                        replay_iterator
                    )

                except StopIteration:

                    # Cycle through the replay set if it has fewer batches
                    # than the current-task loader.
                    replay_iterator = iter(replay_loader)

                    replay_images, replay_labels = next(
                        replay_iterator
                    )

                replay_labels = (
                    replay_labels.long().view(-1)
                )

                images = torch.cat(
                    [current_images, replay_images],
                    dim=0,
                )

                labels = torch.cat(
                    [current_labels, replay_labels],
                    dim=0,
                )

            else:

                images = current_images
                labels = current_labels

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            batch_count = labels.size(0)

            total_loss += loss.item() * batch_count
            total_examples += batch_count

    average_loss = total_loss / max(
        total_examples,
        1,
    )

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    return local_state, average_loss


# ---------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------

def fedavg_aggregate(client_states, client_weights):
    """
    Weighted FedAvg aggregation.

    Client weights are based on the size of each client's
    local training dataset, including replay examples.
    """

    total_weight = sum(client_weights)

    global_state = {}

    for key in client_states[0]:

        if torch.is_floating_point(
            client_states[0][key]
        ):

            # Average learnable floating-point values by each client's
            # effective local dataset size.
            aggregated = torch.zeros_like(
                client_states[0][key]
            )

            for state, weight in zip(
                client_states,
                client_weights,
            ):

                aggregated += (
                    state[key] * (weight / total_weight)
                )

            global_state[key] = aggregated

        else:

            # For non-floating-point buffers, copy the
            # first client's value.
            global_state[key] = (
                client_states[0][key].clone()
            )

    return global_state


def calculate_model_payload_bytes(model_state):
    """
    Estimate raw tensor payload size.

    Does not include network/protocol overhead.
    """

    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model_state.values()
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    test_dataset,
    test_targets,
    seen_classes,
    task_definitions,
    device,
):
    """
    Evaluate only classes introduced so far.

    Logits for unseen classes are masked so that the model
    is not penalised for classes it has not learned yet.

    Returns:
        combined accuracy
        combined macro-F1
        combined balanced accuracy
        task-specific accuracies
    """

    model.eval()

    seen_indices = np.where(
        np.isin(test_targets, seen_classes)
    )[0]

    test_loader = DataLoader(
        Subset(test_dataset, seen_indices.tolist()),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    all_predictions = []
    all_labels = []

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.long().view(-1)

        outputs = model(images)

        # Mask classes that have not yet been introduced.
        unseen_classes = [
            class_id
            for class_id in range(9)
            if class_id not in seen_classes
        ]

        if unseen_classes:
            outputs[:, unseen_classes] = -1e9

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    combined_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    combined_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=seen_classes,
        average="macro",
        zero_division=0,
    )

    combined_balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    task_accuracies = {}

    for task_index, task_classes in enumerate(
        task_definitions
    ):

        if not set(task_classes).issubset(
            set(seen_classes)
        ):
            continue

        task_mask = np.isin(
            y_true,
            task_classes,
        )

        task_true = y_true[task_mask]
        task_pred = y_pred[task_mask]

        task_accuracy = accuracy_score(
            task_true,
            task_pred,
        )

        task_accuracies[
            f"task_{task_index + 1}_accuracy"
        ] = float(task_accuracy)

    return (
        float(combined_accuracy),
        float(combined_macro_f1),
        float(combined_balanced_accuracy),
        task_accuracies,
    )


# ---------------------------------------------------------
# Forgetting calculation
# ---------------------------------------------------------

def calculate_forgetting(
    history,
    task_index,
    final_accuracy,
):
    """
    Forgetting for a task is:

        best accuracy since task introduction
        minus
        current/final accuracy

    Only observations after task introduction are used.
    """

    task_key = f"task_{task_index + 1}_accuracy"

    introduced_stage = task_index + 1

    observed = [
        row[task_key]
        for row in history
        if (
            int(row["stage"]) >= introduced_stage
            and row.get(task_key) is not None
        )
    ]

    if not observed:
        return None, None

    best_accuracy = max(observed)

    forgetting = max(
        0.0,
        best_accuracy - final_accuracy,
    )

    return (
        float(best_accuracy),
        float(forgetting),
    )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FedCL with local experience replay on PathMNIST"
        )
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rounds-per-task",
        type=int,
        default=5,
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
        "--lr",
        dest="learning_rate",
        type=float,
        default=0.001,
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
        "--memory-per-class",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/fedcl_replay",
    )

    args = parser.parse_args()

    if args.num_clients < 1:
        parser.error("--num-clients must be at least 1")

    if args.rounds_per_task < 1:
        parser.error("--rounds-per-task must be at least 1")

    if args.memory_per_class < 1:
        parser.error("--memory-per-class must be at least 1")

    if not 0.0 < args.replay_fraction < 1.0:
        parser.error("--replay-fraction must be between 0 and 1")

    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")

    set_seed(args.seed)

    torch.set_num_threads(args.num_threads)

    device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("FEDERATED CONTINUAL LEARNING WITH EXPERIENCE REPLAY")
    print("=" * 65)

    print(f"Clients: {args.num_clients}")
    print(f"Rounds per task: {args.rounds_per_task}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Memory per class/client: {args.memory_per_class}")
    print(f"Replay fraction: {args.replay_fraction}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # -----------------------------------------------------
    # Load PathMNIST
    # -----------------------------------------------------

    print("\nLoading PathMNIST...")

    transform = transforms.ToTensor()

    train_dataset = PathMNIST(
        split="train",
        download=True,
        transform=transform,
    )

    test_dataset = PathMNIST(
        split="test",
        download=True,
        transform=transform,
    )

    train_targets = get_targets(train_dataset)
    test_targets = get_targets(test_dataset)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Classes: 9")

    # -----------------------------------------------------
    # Initialize global model
    # -----------------------------------------------------

    global_model = SmallCNN().to(device)

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    model_payload_bytes = calculate_model_payload_bytes(
        global_state
    )

    print(
        f"Model payload per client: "
        f"{model_payload_bytes:,} bytes"
    )

    # -----------------------------------------------------
    # Initialize client memories
    # -----------------------------------------------------

    client_memories = [
        {}
        for _ in range(args.num_clients)
    ]

    client_rngs = [
        np.random.default_rng(
            args.seed + 1000 + client_id
        )
        for client_id in range(args.num_clients)
    ]

    partition_rng = np.random.default_rng(
        args.seed
    )

    history = []

    seen_classes = []

    global_round = 0

    experiment_start = time.time()

    # -----------------------------------------------------
    # Sequential task learning
    # -----------------------------------------------------

    # Keep one global model across stages; only the data available to clients
    # changes as each new class group is introduced.
    for task_index, task_classes in enumerate(TASKS):

        stage = task_index + 1

        seen_classes.extend(task_classes)

        print("\n" + "=" * 65)

        print(
            f"STAGE {stage}/{len(TASKS)} | "
            f"New classes: {task_classes} | "
            f"Seen classes: {seen_classes}"
        )

        print("=" * 65)

        # Create current-task partitions.
        client_partitions = create_task_partitions(
            train_targets,
            task_classes,
            args.num_clients,
            partition_rng,
        )

        for client_id, indices in enumerate(
            client_partitions
        ):

            print(
                f"Client {client_id + 1}: "
                f"{len(indices)} current-task examples"
            )

        # -------------------------------------------------
        # Communication rounds
        # -------------------------------------------------

        for round_index in range(
            args.rounds_per_task
        ):

            global_round += 1

            round_start = time.time()

            client_states = []
            client_weights = []
            client_losses = []

            for client_id in range(
                args.num_clients
            ):

                current_indices = (
                    client_partitions[client_id]
                )

                replay_indices = get_replay_indices(
                    client_memories[client_id]
                )

                local_state, local_loss = train_local_model(
                    global_state=global_state,
                    train_dataset=train_dataset,
                    current_indices=current_indices,
                    replay_indices=replay_indices,
                    device=device,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    replay_fraction=args.replay_fraction,
                )

                client_states.append(local_state)

                # Weight by current examples plus replay examples so clients
                # with more local training data contribute proportionally.
                client_weights.append(
                    len(current_indices) + len(replay_indices)
                )

                client_losses.append(local_loss)

            # Server aggregates local models into the checkpoint for the next
            # communication round (and, later, the next task).
            global_state = fedavg_aggregate(
                client_states,
                client_weights,
            )

            global_model.load_state_dict(
                global_state
            )

            # -------------------------------------------------
            # Evaluate global model
            # -------------------------------------------------

            (
                combined_accuracy,
                combined_macro_f1,
                combined_balanced_accuracy,
                task_accuracies,
            ) = evaluate_model(
                model=global_model,
                test_dataset=test_dataset,
                test_targets=test_targets,
                seen_classes=seen_classes,
                task_definitions=TASKS,
                device=device,
            )

            # -------------------------------------------------
            # Calculate forgetting at this round
            # -------------------------------------------------

            temporary_row = {
                "stage": stage,
                **task_accuracies,
            }

            previous_history = history + [
                temporary_row
            ]

            old_task_forgetting = []

            for old_task_index in range(
                stage - 1
            ):

                task_key = (
                    f"task_{old_task_index + 1}_accuracy"
                )

                current_accuracy = task_accuracies.get(
                    task_key
                )

                if current_accuracy is None:
                    continue

                _, forgetting = calculate_forgetting(
                    previous_history,
                    old_task_index,
                    current_accuracy,
                )

                if forgetting is not None:
                    old_task_forgetting.append(
                        forgetting
                    )

            if old_task_forgetting:
                mean_old_forgetting = float(
                    np.mean(old_task_forgetting)
                )
            else:
                mean_old_forgetting = None

            round_seconds = (
                time.time() - round_start
            )

            row = {
                "global_round": global_round,
                "stage": stage,
                "round_in_stage": round_index + 1,
                "new_classes": ",".join(
                    map(str, task_classes)
                ),
                "seen_classes": ",".join(
                    map(str, seen_classes)
                ),
                "mean_client_loss": float(
                    np.mean(client_losses)
                ),
                "combined_accuracy": combined_accuracy,
                "combined_macro_f1": combined_macro_f1,
                "combined_balanced_accuracy": (
                    combined_balanced_accuracy
                ),
                "mean_forgetting_old_tasks": (
                    mean_old_forgetting
                ),
                "round_seconds": round_seconds,
            }

            for task_number in range(
                1,
                len(TASKS) + 1,
            ):

                task_key = (
                    f"task_{task_number}_accuracy"
                )

                row[task_key] = task_accuracies.get(
                    task_key
                )

            history.append(row)

            forgetting_text = (
                f"{mean_old_forgetting:.4f}"
                if mean_old_forgetting is not None
                else "N/A"
            )

            print(
                f"Stage {stage} | "
                f"Round {round_index + 1}/"
                f"{args.rounds_per_task} | "
                f"accuracy={combined_accuracy:.4f} | "
                f"macro_f1={combined_macro_f1:.4f} | "
                f"balanced_acc={combined_balanced_accuracy:.4f} | "
                f"old_forgetting={forgetting_text}"
            )

        # Add exemplars only after all rounds for this task; the replay set
        # therefore remains unchanged within the stage.
        # -------------------------------------------------
        # Update each client's local replay memory
        # -------------------------------------------------

        print(
            f"\nUpdating replay buffers after Stage {stage}..."
        )

        for client_id in range(
            args.num_clients
        ):

            client_memories[client_id] = (
                update_replay_buffer(
                    memory=client_memories[client_id],
                    current_indices=client_partitions[
                        client_id
                    ],
                    train_targets=train_targets,
                    task_classes=task_classes,
                    memory_per_class=args.memory_per_class,
                    rng=client_rngs[client_id],
                )
            )

            memory_size = len(
                get_replay_indices(
                    client_memories[client_id]
                )
            )

            print(
                f"Client {client_id + 1} replay memory: "
                f"{memory_size} examples"
            )

    # -----------------------------------------------------
    # Final evaluation and forgetting
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 65)

    (
        final_accuracy,
        final_macro_f1,
        final_balanced_accuracy,
        final_task_accuracies,
    ) = evaluate_model(
        model=global_model,
        test_dataset=test_dataset,
        test_targets=test_targets,
        seen_classes=list(range(9)),
        task_definitions=TASKS,
        device=device,
    )

    final_forgetting = {}

    old_task_forgetting_values = []

    all_task_forgetting_values = []

    for task_index in range(len(TASKS)):

        task_key = (
            f"task_{task_index + 1}_accuracy"
        )

        final_task_accuracy = (
            final_task_accuracies[task_key]
        )

        best_accuracy, forgetting = calculate_forgetting(
            history,
            task_index,
            final_task_accuracy,
        )

        final_forgetting[task_key] = {
            "best_accuracy_since_introduction": (
                best_accuracy
            ),
            "final_accuracy": final_task_accuracy,
            "forgetting": forgetting,
        }

        all_task_forgetting_values.append(
            forgetting
        )

        if task_index < len(TASKS) - 1:
            old_task_forgetting_values.append(
                forgetting
            )

    final_mean_forgetting_old_tasks = float(
        np.mean(old_task_forgetting_values)
    )

    final_mean_forgetting_all_tasks = float(
        np.mean(all_task_forgetting_values)
    )

    total_training_seconds = (
        time.time() - experiment_start
    )

    # -----------------------------------------------------
    # Communication accounting
    # -----------------------------------------------------

    # The communication estimate assumes a full model download and upload for
    # every client in every round; it excludes protocol/transport overhead.
    total_rounds = (
        len(TASKS) * args.rounds_per_task
    )

    downlink_bytes = (
        total_rounds
        * args.num_clients
        * model_payload_bytes
    )

    uplink_bytes = (
        total_rounds
        * args.num_clients
        * model_payload_bytes
    )

    total_communication_bytes = (
        downlink_bytes + uplink_bytes
    )

    # -----------------------------------------------------
    # Replay memory accounting
    # -----------------------------------------------------

    memory_sizes = [
        len(get_replay_indices(memory))
        for memory in client_memories
    ]

    total_memory_examples = sum(
        memory_sizes
    )

    # PathMNIST source images are 28x28 RGB uint8.
    # Compare index-based storage with the equivalent uncompressed image
    # footprint; this does not represent actual process memory use.
    estimated_replay_image_bytes = (
        total_memory_examples * 28 * 28 * 3
    )

    replay_index_bytes = (
        total_memory_examples * 8
    )

    # -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

    model_path = (
        output_dir / "fedcl_replay_global_model.pt"
    )

    torch.save(
        global_model.state_dict(),
        model_path,
    )

    # -----------------------------------------------------
    # Save history CSV
    # -----------------------------------------------------

    history_path = (
        output_dir / "history.csv"
    )

    if history:

        with open(
            history_path,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )

            writer.writeheader()
            writer.writerows(history)

    # -----------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------

    metrics = {
        "experiment": "FedCL with local experience replay",
        "dataset": "PathMNIST",
        "num_clients": args.num_clients,
        "tasks": TASKS,
        "rounds_per_task": args.rounds_per_task,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "memory_per_class_per_client": (
            args.memory_per_class
        ),
        "replay_fraction": args.replay_fraction,
        "seed": args.seed,
        "device": str(device),

        "final_combined_accuracy": (
            final_accuracy
        ),

        "final_combined_macro_f1": (
            final_macro_f1
        ),

        "final_combined_balanced_accuracy": (
            final_balanced_accuracy
        ),

        "final_mean_forgetting_old_tasks": (
            final_mean_forgetting_old_tasks
        ),

        "final_mean_forgetting_all_tasks": (
            final_mean_forgetting_all_tasks
        ),

        "final_task_accuracies_and_forgetting": (
            final_forgetting
        ),

        "training_seconds": (
            total_training_seconds
        ),

        "peak_ram_mib": get_peak_ram_mib(),

        "model_payload_bytes_per_client": (
            model_payload_bytes
        ),

        "cumulative_downlink_bytes": (
            downlink_bytes
        ),

        "cumulative_uplink_bytes": (
            uplink_bytes
        ),

        "cumulative_total_communication_bytes": (
            total_communication_bytes
        ),

        "client_replay_memory_examples": (
            memory_sizes
        ),

        "total_replay_memory_examples": (
            total_memory_examples
        ),

        "estimated_replay_image_storage_bytes": (
            estimated_replay_image_bytes
        ),

        "replay_index_storage_bytes": (
            replay_index_bytes
        ),

        "communication_note": (
            "Raw model tensor payload estimate. "
            "Excludes protocol headers, metadata, "
            "compression, encryption and transport overhead."
        ),

        "replay_memory_note": (
            "Estimated uint8 image storage assumes "
            "28x28 RGB images. Replay examples are "
            "represented by dataset indices in this "
            "implementation; index storage is reported "
            "separately. This is not total process RAM."
        ),
    }

    metrics_path = (
        output_dir / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\nFinal combined accuracy: "
          f"{final_accuracy:.4f}")

    print("Final combined macro-F1: "
          f"{final_macro_f1:.4f}")

    print("Final combined balanced accuracy: "
          f"{final_balanced_accuracy:.4f}")

    print("Final mean forgetting (old tasks): "
          f"{final_mean_forgetting_old_tasks:.4f}")

    print("Final mean forgetting (all tasks): "
          f"{final_mean_forgetting_all_tasks:.4f}")

    print(
        "Total communication: "
        f"{total_communication_bytes:,} bytes"
    )

    print(
        "Total replay examples across clients: "
        f"{total_memory_examples}"
    )

    print(
        "Estimated replay image storage: "
        f"{estimated_replay_image_bytes:,} bytes"
    )

    print(
        f"Training time: {total_training_seconds:.2f} seconds"
    )

    print(
        f"Peak RAM: {get_peak_ram_mib():.2f} MiB"
    )

    print("\nSaved outputs:")
    print(f"  Model:   {model_path}")
    print(f"  History: {history_path}")
    print(f"  Metrics: {metrics_path}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
