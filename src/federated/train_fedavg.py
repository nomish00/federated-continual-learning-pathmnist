import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST

# Allow running this file directly from the project root:
# python src/federated/train_fedavg.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_central import SmallCNN, set_seed, evaluate  # noqa: E402


def make_iid_partitions(dataset_size, num_clients, seed):
    """Create disjoint, approximately equal IID partitions of the training set."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)
    partitions = np.array_split(indices, num_clients)
    return [part.astype(int).tolist() for part in partitions]


def make_dirichlet_partitions(labels, num_clients, alpha, seed):
    """Create non-IID class-skewed partitions using a Dirichlet distribution.

    A smaller alpha produces stronger class imbalance. The minimum-size
    condition avoids accidentally creating empty or tiny clients.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be greater than zero")

    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    min_size = 0
    min_required = max(1, int(0.1 * len(labels) / num_clients))
    attempts = 0

    while min_size < min_required:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not create balanced-enough Dirichlet partitions")

        client_indices = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Avoid giving most of a class to clients that already have
            # substantially more samples than average.
            proportions = np.asarray([
                p * (len(client_indices[c]) < len(labels) / num_clients)
                for c, p in enumerate(proportions)
            ])
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)
            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        min_size = min(len(indices) for indices in client_indices)

    return [rng.permutation(indices).astype(int).tolist() for indices in client_indices]


def client_class_counts(labels, partitions):
    labels = np.asarray(labels).reshape(-1)
    return [
        {str(int(class_id)): int(np.sum(labels[indices] == class_id))
         for class_id in np.unique(labels)}
        for indices in partitions
    ]


def train_local_model(global_state, train_dataset, indices, args, device):
    """Train one client from the current global model and return its state."""
    local_model = SmallCNN(num_classes=9).to(device)
    local_model.load_state_dict(global_state)
    loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    local_model.train()
    total_loss = 0.0
    total_examples = 0

    for _ in range(args.local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = local_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            n = labels.size(0)
            total_loss += loss.item() * n
            total_examples += n

    mean_loss = total_loss / max(total_examples, 1)
    return copy.deepcopy(local_model.state_dict()), mean_loss


def fedavg(client_states, client_sizes):
    """Sample-weighted average of client model parameters."""
    total_size = float(sum(client_sizes))
    averaged_state = {}

    for key in client_states[0]:
        # This starter model has only floating-point parameters/buffers.
        averaged_state[key] = torch.zeros_like(client_states[0][key])
        for state, size in zip(client_states, client_sizes):
            averaged_state[key] += state[key] * (size / total_size)

    return averaged_state


def state_dict_nbytes(state):
    """Raw tensor payload size, excluding protocol/serialization overhead."""
    return sum(t.numel() * t.element_size() for t in state.values())


def main():
    parser = argparse.ArgumentParser(description="FedAvg on PathMNIST with IID or Dirichlet non-IID clients")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if min(args.rounds, args.num_clients, args.local_epochs,
           args.batch_size, args.num_threads) < 1:
        raise ValueError("rounds, clients, local-epochs, batch-size and threads must be positive")

    set_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    transform = transforms.ToTensor()

    print("Loading PathMNIST...", flush=True)
    train_ds = PathMNIST(split="train", transform=transform, download=True)
    test_ds = PathMNIST(split="test", transform=transform, download=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    train_labels = np.asarray(train_ds.labels).reshape(-1)
    if args.partition == "iid":
        partitions = make_iid_partitions(len(train_ds), args.num_clients, args.seed)
        partition_description = "iid_random_equal_size"
    else:
        partitions = make_dirichlet_partitions(
            train_labels, args.num_clients, args.dirichlet_alpha, args.seed
        )
        partition_description = f"dirichlet_alpha_{args.dirichlet_alpha}"

    client_sizes = [len(p) for p in partitions]
    class_counts = client_class_counts(train_labels, partitions)
    print(
        f"Train={len(train_ds)} | Test={len(test_ds)} | "
        f"Clients={args.num_clients} | Client samples={client_sizes}",
        flush=True,
    )
    for client_id, counts in enumerate(class_counts, start=1):
        print(f"Client {client_id} class counts: {counts}", flush=True)

    global_model = SmallCNN(num_classes=9).to(device)
    global_state = copy.deepcopy(global_model.state_dict())
    model_payload_bytes = state_dict_nbytes(global_state)

    history = []
    total_down_bytes = 0
    total_up_bytes = 0
    start_time = time.perf_counter()

    for round_idx in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        client_states = []
        client_losses = []

        for client_id, indices in enumerate(partitions, start=1):
            local_state, local_loss = train_local_model(
                global_state, train_ds, indices, args, device
            )
            client_states.append(local_state)
            client_losses.append(local_loss)

        # Each client receives the global model and uploads a full model state.
        total_down_bytes += model_payload_bytes * args.num_clients
        total_up_bytes += model_payload_bytes * args.num_clients

        global_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(global_state)
        metrics = evaluate(global_model, test_loader, device)

        row = {
            "round": round_idx,
            "mean_client_train_loss": float(np.average(client_losses, weights=client_sizes)),
            **metrics,
            "round_seconds": time.perf_counter() - round_start,
            "cumulative_downlink_bytes": total_down_bytes,
            "cumulative_uplink_bytes": total_up_bytes,
            "cumulative_communication_bytes": total_down_bytes + total_up_bytes,
        }
        history.append(row)

        print(
            f"Round {round_idx}/{args.rounds} | "
            f"client_loss={row['mean_client_train_loss']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} | "
            f"round_time={row['round_seconds']:.1f}s",
            flush=True,
        )

    total_seconds = time.perf_counter() - start_time
    default_name = (
        "fedavg_iid" if args.partition == "iid"
        else f"fedavg_dirichlet_a{args.dirichlet_alpha:g}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / default_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fedavg_global_model.pt"
    torch.save(global_state, model_path)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    import resource
    peak_ram_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    result = {
        "experiment": f"fedavg_pathmnist_{args.partition}",
        "seed": args.seed,
        "num_clients": args.num_clients,
        "client_samples": client_sizes,
        "client_class_counts": class_counts,
        "partition": partition_description,
        "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_threads": args.num_threads,
        "device": str(device),
        "training_seconds": total_seconds,
        "peak_process_ram_mib": peak_ram_mib,
        "model_payload_bytes_per_client": model_payload_bytes,
        "cumulative_downlink_bytes": total_down_bytes,
        "cumulative_uplink_bytes": total_up_bytes,
        "cumulative_communication_bytes": total_down_bytes + total_up_bytes,
        "communication_note": (
            "Raw model tensor payload estimate; excludes protocol, metadata, "
            "transport and security overhead."
        ),
        "final_test_metrics": history[-1],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSaved results to:", output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
