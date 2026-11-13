# PathMNIST: Central, Federated, and Continual-Learning Experiments

This project contains CPU-capable PathMNIST training scripts for a central
baseline, simulated federated learning, and federated continual learning with
and without replay. The scripts use the official MedMNIST train and test splits
and save metrics, per-epoch/round history, and model weights.

## Project layout

```text
src/
  train_central.py
  federated/
    train_fedavg.py
    train_fedavg_iid_backup.py
    train_fedcl_iid.py
    train_fedcl_replay.py
    train_fedcl_replay_ablation.py
data/       # dataset files (downloaded as needed)
outputs/    # central and federated results
runs/       # replay-ablation results
logs/       # optional command logs
```

Run commands from `pathmnist_cpu_starter/`, the directory containing this
README. Create and activate an environment, then install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install its CPU wheels before the project
requirements:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

PathMNIST is downloaded automatically by the scripts if it is not already
available. Most scripts use the MedMNIST cache; the replay ablation script
defaults to `./data`.

## Scripts and example commands

The examples below use short runs for smoke testing. Omit the shortened
epoch/round arguments for the scripts' defaults, or adjust the options for
your experiment.

### Central baseline

`src/train_central.py` trains the compact CNN centrally on all training data.
It evaluates the official test split after each epoch.

```bash
mkdir -p logs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python src/train_central.py --epochs 1 --batch-size 128 --num-threads 4 \
  2>&1 | tee logs/central_smoke_test.log
```

Key options include `--epochs`, `--batch-size`, `--lr`, `--num-threads`,
`--seed`, and `--output-dir`. The default output directory is `outputs/`.

### Federated averaging (FedAvg)

`src/federated/train_fedavg.py` simulates multiple clients and performs
sample-weighted FedAvg. It supports approximately equal IID partitions and
class-skewed Dirichlet partitions.

```bash
# IID clients
python src/federated/train_fedavg.py \
  --rounds 1 --num-clients 3 --local-epochs 1 --partition iid \
  --num-threads 4

# Non-IID clients (smaller alpha means stronger class imbalance)
python src/federated/train_fedavg.py \
  --rounds 1 --num-clients 3 --local-epochs 1 \
  --partition dirichlet --dirichlet-alpha 0.5 --num-threads 4
```

Its main options are `--rounds`, `--num-clients`, `--local-epochs`,
`--batch-size`, `--lr`, `--partition {iid,dirichlet}`,
`--dirichlet-alpha`, `--seed`, and `--output-dir`. By default results go to
`outputs/fedavg_iid/` or `outputs/fedavg_dirichlet_a<alpha>/`.

`src/federated/train_fedavg_iid_backup.py` is a standalone older IID FedAvg
copy. Prefer `train_fedavg.py --partition iid` for new runs; the backup writes
to `outputs/fedavg_iid/` by default, so running both with defaults overwrites
the same result files.

### Federated continual learning (FedCL)

The FedCL scripts learn three sequential class groups: `[0, 1, 2]`, `[3, 4, 5]`,
and `[6, 7, 8]`. Both simulate clients and aggregate with FedAvg.

```bash
# Sequential learning without replay
python src/federated/train_fedcl_iid.py \
  --rounds-per-task 1 --num-clients 3 --num-threads 4

# Sequential learning with a local replay buffer
python src/federated/train_fedcl_replay.py \
  --rounds-per-task 1 --num-clients 3 \
  --memory-per-class 10 --replay-fraction 0.25 --num-threads 4
```

Both scripts also accept `--local-epochs`, `--batch-size`, `--seed`, and
`--output-dir` (as well as a learning-rate option). The no-replay run defaults
to `outputs/fedcl_iid/`; replay defaults to `outputs/fedcl_replay/`.

### Replay-memory ablation

`src/federated/train_fedcl_replay_ablation.py` runs the replay experiment for
multiple memory budgets (examples per class per client) and writes a combined
summary. It defaults to budgets 10, 25, 50, and 100.

```bash
python src/federated/train_fedcl_replay_ablation.py \
  --memory-per-class 10 25 \
  --rounds-per-task 1 --num-clients 3 --local-epochs 1 --threads 4
```

Use `--memory-per-class` to select budgets, `--dirichlet-alpha` to configure
client class partitions (`<= 0` selects IID), `--replay-fraction` to set the
replay share of a training batch, and `--output-dir` or `--data-dir` to change
the default paths. Results default to `runs/fedcl_replay_ablation/`.

## Results

Each individual run writes:

- `metrics.json` — run settings, final metrics, and resource/communication
  measurements where supported
- `history.csv` — metrics and training history by epoch, round, or stage
- a `.pt` file — saved model weights

The replay ablation writes these files under a `memory_<budget>/` subdirectory
for each budget, plus `ablation_summary.csv` and
`ablation_summary.json` at the output root. Pass a distinct `--output-dir` to
each run if you want to preserve multiple runs; the scripts overwrite matching
files in an existing output directory.

FedAvg communication figures are raw model-tensor payload estimates, not
network-level measurements. Replay buffers store training-set indices. The
continual-learning scripts mask unseen classes during evaluation; the
ablation reports test metrics after each round, so use a separate validation
protocol for model selection and reserve the official test split for final
reporting.

## Reproducible experiments

The training scripts default to seed 42 and four CPU threads; the federated
scripts also default to three clients. Record the code version, Python/PyTorch
versions, CPU and RAM, seed, partition settings, training parameters, and
output metrics for each run. Repeat with multiple seeds and report mean and
standard deviation rather than treating a single run as a final result.
