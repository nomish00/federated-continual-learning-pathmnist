# PathMNIST CPU starter

A small, reproducible **centralised baseline** for the first stage of your project.
It trains a compact CNN on PathMNIST, evaluates the official test split, and records
accuracy, macro-F1, balanced accuracy, training time, peak process RAM, and model size.

This is deliberately **not federated learning yet**. Establish this baseline first;
then reuse the same model, preprocessing, and test set for the federated experiments.

## 1. Server setup (Ubuntu, SSH)

Check your environment:

```bash
lsb_release -a
lscpu | head -n 20
free -h
df -h .
python3 --version
```

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git tmux htop
```

Create the project environment:

```bash
mkdir -p ~/aimlb/pathmnist_cpu_starter
cd ~/aimlb/pathmnist_cpu_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CPU-only PyTorch and the experiment dependencies:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install medmnist scikit-learn matplotlib pandas
```

If you copied this ZIP to the server, unzip it in `~/aimlb/`, then `cd` into
`pathmnist_cpu_starter`. The ZIP already contains `src/train_central.py`.

## 2. Verify the installation

```bash
python - <<'PY'
import torch, medmnist
from medmnist import PathMNIST
print("PyTorch:", torch.__version__)
print("MedMNIST:", medmnist.__version__)
print("CUDA available:", torch.cuda.is_available())
ds = PathMNIST(split="train", download=True)
print("Training samples:", len(ds))
print("Image shape:", ds[0][0].size)
print("Label:", ds[0][1])
PY
```

`CUDA available: False` is expected on a CPU-only server.

## 3. Run a quick smoke test

Start with one epoch and a modest CPU thread count:

```bash
mkdir -p logs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python src/train_central.py --epochs 1 --batch-size 128 --num-threads 4 \
2>&1 | tee logs/smoke_test.log
```

If it runs successfully, run a slightly longer baseline:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python src/train_central.py --epochs 5 --batch-size 128 --num-threads 4 \
2>&1 | tee logs/central_baseline.log
```

Use `tmux` so the run continues if your SSH connection drops:

```bash
tmux new -s pathmnist
# run the training command inside tmux
# detach: Ctrl-b, then d
# reattach later:
tmux attach -t pathmnist
```

## 4. Outputs

The script creates `outputs/` containing:

- `metrics.json` — test metrics and run metadata
- `history.csv` — training loss and validation metrics by epoch
- `central_model.pt` — model weights

The first run downloads PathMNIST into the MedMNIST cache. Keep the dataset
version, random seed, split, image size, and training settings fixed when comparing
later methods.

## 5. What to record in your experiment log

For every run, record:

- date, code version/commit, Python and PyTorch versions
- CPU model, number of CPU threads, RAM
- random seed, epochs, batch size, learning rate
- official train/test split and PathMNIST image size
- accuracy, macro-F1, balanced accuracy
- training time, peak RAM, model size

Do not treat a single run as a final scientific result. Later, repeat with multiple
seeds and report mean and standard deviation.

## 6. Next stage

After this baseline works, the next steps are:

1. Partition only the **training** split among three simulated clients.
2. Create IID and controlled non-IID partitions.
3. Implement FedAvg and evaluate on the same untouched official test set.
4. Add sequential learning stages and measure forgetting on old classes.
5. Compare full-model updates with a clearly defined partial-update method.

Do not introduce all of these at once. First confirm that the centralised baseline
runs and that you understand its outputs.
