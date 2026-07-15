# AlphaCFG

AlphaCFG is a unified framework for grammar-guided alpha-factor discovery.
Each method owns its grammar and MCTS state transitions, while data loading,
replay training, rewards, evaluation, and output management are shared.

## Framework Overview

The following diagram shows the complete AlphaCFG research and training
workflow.

![AlphaCFG framework overview](./figure/AlphaCFG.png)

## Example Backtest Result

The following archived figure summarizes the backtest results produced by the
original AlphaCFG experiments.

![AlphaCFG backtest result](./figure/Result.png)

## 1. Project Structure

```text
.
|-- alphacfg/
|   |-- data/                  # Qlib data, expressions, and IC calculation
|   |-- env/                   # Shared input features
|   |-- grammar/               # Grammar metadata and tree similarity
|   |-- models/                # Linear alpha pool
|   |-- network_backends/      # LSTM/CNN/Transformer/TreeLSTM
|   |-- methods/               # RPN/CFG-Syn/CFG-Sem/CFG-Sem-k
|   |-- mcts_core.py           # Shared PUCT search engine
|   `-- unified_runner.py      # Search, rewards, training, and outputs
|-- scripts/
|   |-- run_mcts_experiment.py
|   |-- update_qlib_cn_daily_akshare.py
|   |-- check_cfg_sem_k_networks.py
|   `-- check_seed_reproducibility.py
|-- data/                      # Project-local Qlib providers
|-- outputs/                   # One directory for every experiment
|-- run_variant.py             # Unified experiment launcher
`-- requirement.txt
```

Available search methods:

```text
rpn
cfg-syn
cfg-sem
cfg-sem-k
```

Available network backends:

```text
lstm
cnn
transformer
treelstm
```

RPN represents states as token sequences and therefore does not support
TreeLSTM. The CFG methods support all four network backends.

## 2. Data

AlphaCFG reads Qlib binary providers with at least the following structure:

```text
<provider_uri>/
|-- calendars/day.txt
|-- instruments/<market>.txt
`-- features/<symbol>/
    |-- open.day.bin
    |-- close.day.bin
    |-- high.day.bin
    |-- low.day.bin
    |-- volume.day.bin
    `-- vwap.day.bin
```

The current grammar uses daily `open`, `close`, `high`, `low`, `volume`, and
`vwap`. The default target is the forward 20-trading-day return:

```text
Ref($close, -20) / $close - 1
```

### 2.1 China A-Share Data

By default, AlphaCFG looks for `data/cn_data_rolling`. If that directory does
not exist, it checks these locations in order:

1. `ALPHACFG_QLIB_PROVIDER_URI`;
2. `~/.qlib/qlib_data/cn_data_rolling`.

The included incremental updater uses the
[AKShare `stock_zh_a_daily`](https://akshare.akfamily.xyz/data/stock/stock.html)
API, whose underlying source is Sina Finance daily market data. The updater
writes unadjusted OHLCV values, computes `vwap` as `amount / volume`, and
currently writes `factor` as `1`.

The updater extends an existing Qlib provider. It requires an existing
calendar, instrument lists, and historical feature files; it cannot initialize
a completely empty directory.

Update an existing local provider to a selected trading date:

```bash
.venv/bin/python scripts/update_qlib_cn_daily_akshare.py \
  --provider-uri data/cn_data_rolling \
  --end-date 2026-07-01 \
  --markets csi100 csi300 csi500 \
  --workers 8
```

Qlib also exposes a CLI for downloading its public China daily sample data:

```bash
.venv/bin/python -m qlib.cli.data qlib_data \
  --target_dir data/cn_data \
  --region cn \
  --interval 1d
```

The Qlib project currently notes that some public download endpoints may be
temporarily unavailable. If the command fails, consult
[Qlib Data Preparation](https://github.com/microsoft/qlib#data-preparation)
for the current community mirror, or use a compliant data source of your own.

### 2.2 US Equity Data

Qlib provides a download entry point for its US daily sample data:

```bash
.venv/bin/python -m qlib.cli.data qlib_data \
  --target_dir data/us_data \
  --region us \
  --interval 1d
```

According to the
[Qlib data documentation](https://qlib.readthedocs.io/en/latest/component/data.html),
the public sample dataset is produced by its public collection pipeline and is
intended for research demonstrations. Qlib also warns that public market data
may be incomplete and should not be treated as execution-grade data.

Before running a US experiment, set the provider, region, and an instrument
universe that actually exists in that provider, such as `sp500`:

```bash
export ALPHACFG_QLIB_PROVIDER_URI="$PWD/data/us_data"
export ALPHACFG_QLIB_REGION=us
```

Switch back to the project-local China provider with:

```bash
export ALPHACFG_QLIB_PROVIDER_URI="$PWD/data/cn_data_rolling"
export ALPHACFG_QLIB_REGION=cn
```

> Audit the data before using research results. Public datasets may contain
> suspension, delisting, adjustment, historical-constituent, and survivorship
> issues. Production research should record the data version, retrieval date,
> adjustment convention, and universe construction process.

## 3. Environment Setup

### 3.1 Verified Environment

| Component | Version or requirement |
| --- | --- |
| Operating system | Linux x86_64 |
| Python | 3.9 |
| NVIDIA driver | Supports CUDA 12.8 or newer |
| PyTorch | 2.8.0 with CUDA 12.8 |
| Qlib | 0.9.7 |
| DGL | 1.1.3 |

`requirement.txt` contains only runtime dependencies required by this project.
It intentionally excludes desktop packages, Jupyter, system-management tools,
and unrelated packages from the host machine.

### 3.2 Create the Virtual Environment

```bash
cd /path/to/AlphaCFG
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirement.txt
```

For a CPU-only machine, install the official CPU PyTorch wheel first:

```bash
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install dgl==1.1.3
python -m pip install -r requirement.txt
```

Verify the installation:

```bash
.venv/bin/python - <<'PY'
import torch
import dgl
import qlib

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
print("dgl:", dgl.__version__)
print("qlib:", qlib.__version__)
PY
```

## 4. Running Experiments

All methods use `run_variant.py` as the entry point.

| Argument | Meaning |
| --- | --- |
| `--variant` | `rpn`, `cfg-syn`, `cfg-sem`, or `cfg-sem-k` |
| `--mode` | `single`, `pool`, or `mask` |
| `--network` | `lstm`, `cnn`, `transformer`, or `treelstm` |
| `--seed` | Global random seed |
| `--instrument` | Qlib universe such as `csi300` or `sp500` |
| `--max-expression-length` | Maximum expression length |
| `--mcts-sim` | MCTS simulations per action |
| `--device` | `cpu`, `cuda:0`, and so on |

### 4.1 Smoke Test

```bash
.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode single \
  --network lstm \
  --seed 0 \
  --instrument csi300 \
  --num-iterations 1 \
  --inner-batches 1 \
  --num-games 2 \
  --mcts-sim 2 \
  --mcts-parallel 1 \
  --device cpu \
  --run-tag smoke
```

### 4.2 Single-Factor Search

A new valid expression receives `abs(IC)`, a duplicate expression receives
`0`, and an invalid expression receives `-1`.

```bash
.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode single \
  --network treelstm \
  --seed 1 \
  --instrument csi300 \
  --max-expression-length 20 \
  --num-iterations 100 \
  --num-games 50 \
  --mcts-sim 64 \
  --mcts-parallel 8 \
  --device cuda:0 \
  --run-tag paper-single
```

### 4.3 Alpha-Pool Search

Pool mode maintains a linear alpha ensemble. The current training reward is:

```text
(1 - max_tree_similarity) * max(new_pool_ic, 0)
```

```bash
.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode pool \
  --network treelstm \
  --seed 1 \
  --instrument csi300 \
  --pool-capacity 10 \
  --max-expression-length 10 \
  --num-iterations 200 \
  --inner-batches 2 \
  --num-games 50 \
  --num-batches 8 \
  --batch-size 64 \
  --mcts-sim 64 \
  --mcts-parallel 8 \
  --device cuda:0 \
  --run-tag paper-pool
```

### 4.4 Mask Completion

```bash
.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode mask \
  --network lstm \
  --mask-expression "Mul1 Rank Q num Q" \
  --device cuda:0
```

### 4.5 US Equity Example

```bash
export ALPHACFG_QLIB_PROVIDER_URI="$PWD/data/us_data"
export ALPHACFG_QLIB_REGION=us

.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode single \
  --network lstm \
  --instrument sp500 \
  --train-start 2010-01-01 \
  --train-end 2017-12-31 \
  --valid-start 2018-01-01 \
  --valid-end 2019-12-31 \
  --test-start 2021-01-01 \
  --test-end 2024-12-31 \
  --device cuda:0
```

## 5. Outputs

Every run creates an independent directory:

```text
outputs/<method>/<method>_<mode>_<network>_pool<capacity>_len<length>_<tag>_<time>/
```

| File | Contents |
| --- | --- |
| `run_config.json` | Complete command configuration |
| `run_metadata.json` | Start time, device, network, and reward formula |
| `training_metrics.csv` | Per-iteration losses, ICs, gradients, and rewards |
| `single_factors.csv` | Single expressions and ICs in single/mask mode |
| `pool_dicts.json` | Pool expressions, weights, and split ICs |
| `checkpoint_latest.pt` | Latest Policy, Value, and optimizer state |

## 6. Validation and Reproducibility

```bash
# Forward, backward, and gradient checks for all CFG-Sem-k networks
.venv/bin/python scripts/check_cfg_sem_k_networks.py

# Fixed-seed reproducibility check
.venv/bin/python scripts/check_seed_reproducibility.py \
  --network lstm \
  --seed 0

# Display grammar metadata for every method
.venv/bin/python scripts/describe_grammar_specs.py

# Reward unit tests
.venv/bin/python -m unittest \
  tests.test_single_reward \
  tests.test_pool_reward \
  -v
```
