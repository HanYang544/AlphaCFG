# AlphaCFG Quick Start

This project searches alpha expressions with policy/value networks and MCTS.

## Environment

Use the isolated environment created under the project root:

```bash
/home/yanghan/AlphaCFG/.venv/bin/python
```

Run with GPU 1:

```bash
/home/yanghan/AlphaCFG/.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode pool \
  --network lstm \
  --device cuda:1
```

## Main Concepts

| File | Purpose |
| ---- | ------- |
| `../../network_backends/treelstm/networks/feature_.py` | TreeLSTM feature extractor for expression trees |
| `../../network_backends/treelstm/networks/policy_.py` | Policy network that predicts the next grammar action |
| `../../network_backends/treelstm/networks/value_.py` | Value network that estimates expression value |
| `mcts.py` | CFG-Sem-k grammar/network adapter for the shared MCTS core |
| `../../unified_runner.py` | Shared pool/single/mask self-play, reward, training, and output runner |
| `run_pool.py` | Thin pool-mode wrapper around the unified runner |
| `run_single.py` | Thin single-factor wrapper around the unified runner |
| `run_mask.py` | Thin mask-improvement wrapper around the unified runner |

## Mask Improvement Mode

Mask mode starts from an unfinished prefix expression that still contains
`Q`, `J`, or `num` non-terminals. Search only fills the remaining
non-terminals. A new expression uses the absolute single-factor IC as its
reward, while a duplicate expression receives zero reward.

Example:

```bash
/home/yanghan/AlphaCFG/.venv/bin/python run_variant.py \
  --variant cfg-sem-k \
  --mode mask \
  --network lstm \
  --mask-expression "Mul1 Rank CSRank Q num Q"
```

## Reproducibility

Use `alphacfg.seeding.seed_everything(seed, deterministic=True)` before model
creation and data generation. The unified launcher already does this.

For exact repeatability, keep `MCTS_PARALLEL = 1`. Parallel MCTS can still be
statistically reproducible, but thread scheduling may change the exact search
path even when all random seeds are fixed.

## MCTS Structure

The search engine itself lives in `alphacfg/mcts_core.py`. This folder's
`mcts.py` only defines CFG-Sem-k specific methods: initial grammar state,
`continue_out_game` length control, and policy/value evaluation input format.
