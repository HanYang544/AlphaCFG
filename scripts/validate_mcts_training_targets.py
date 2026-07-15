"""Validate MCTS policy targets, similarity-aware value targets, and search budgets."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.data.calculator__ import QLibStockDataCalculator
from alphacfg.data.expression import Feature, Ref
from alphacfg.data.stock_data import FeatureType, StockData
from alphacfg.experiment_context import experiment_import_context
from alphacfg.runtime import OUTPUT_ROOT, write_json
from alphacfg.seeding import seed_everything
from alphacfg.unified_runner import (
    ReplayBuffer,
    _dedup_parameters,
    _final_expression,
    _import_runtime_modules,
    _make_feature_extractor,
    _policy_loss,
    _policy_outputs,
    _restore_pool,
    _snapshot_pool,
    _value_output,
    parallel_self_play,
    process_rewards,
    self_play_game,
    train_policy,
    train_value,
)
from alphacfg.models.linear_alpha_pool import MseAlphaPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", choices=["pool", "single"], default="pool")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--instrument", default="csi300")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--target-horizon", type=int, default=20)
    parser.add_argument("--pool-capacity", type=int, default=10)
    parser.add_argument("--max-expression-length", type=int, default=8)
    parser.add_argument("--collection-games", type=int, default=16)
    parser.add_argument("--collection-sims", type=int, default=8)
    parser.add_argument("--fit-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sim-budgets", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--budget-seeds", type=int, default=6)
    parser.add_argument("--output-dir", default="training_target_validation")
    return parser.parse_args()


def policy_metrics(policy_net, replay, device) -> dict:
    policy_net.eval()
    losses = []
    with torch.no_grad():
        for state_input, target_policy, _, aux in replay.buffer:
            loss = _policy_loss(
                _policy_outputs("cfg-sem-k", policy_net, state_input, aux),
                target_policy,
                device,
            )
            if loss is not None:
                losses.append(float(loss.cpu()))
    return {
        "kl": float(np.mean(losses)) if losses else math.nan,
        "samples": len(losses),
    }


def value_metrics(value_net, replay) -> dict:
    value_net.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for state_input, _, reward, _ in replay.buffer:
            predictions.append(float(_value_output("cfg-sem-k", value_net, state_input).mean().cpu()))
            targets.append(float(reward))
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2)) if len(target) else math.nan
    if len(target) > 1 and pred.std() > 0 and target.std() > 0:
        correlation = float(np.corrcoef(pred, target)[0, 1])
    else:
        correlation = math.nan
    return {
        "mse": mse,
        "correlation": correlation,
        "samples": int(len(target)),
        "target_mean": float(target.mean()) if len(target) else math.nan,
        "target_std": float(target.std()) if len(target) else math.nan,
        "target_min": float(target.min()) if len(target) else math.nan,
        "target_max": float(target.max()) if len(target) else math.nan,
    }


def copy_replay(source: ReplayBuffer, destination: ReplayBuffer) -> None:
    for sample in source.buffer:
        destination.push(*sample)


def make_args_for_self_play(args, num_games: int, sims: int) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.num_games = num_games
    copied.mcts_sim = sims
    copied.mcts_parallel = 1
    copied.batch_size_eval = 1
    copied.c_puct = 1.0
    copied.temperature = 1.0
    copied.root_dirichlet_epsilon = 0.0
    copied.root_dirichlet_alpha = 0.03
    copied.mask_expression = ""
    return copied


def collect_real_trajectories(args, policy_net, value_net, mcts_mod, device, calculator, pool):
    train_replay = ReplayBuffer(100000)
    holdout_replay = ReplayBuffer(100000)
    generated_exprs: set[str] = set()
    accepted_records = []
    split = max(1, int(args.collection_games * 0.75))
    self_play_args = make_args_for_self_play(args, 1, args.collection_sims)

    for game_index in range(args.collection_games):
        games = parallel_self_play(
            "cfg-sem-k",
            self_play_args,
            policy_net,
            value_net,
            mcts_mod,
            device,
            iteration=game_index + 1,
        )
        game_replay = ReplayBuffer(10000)
        summary = process_rewards(
            "cfg-sem-k",
            games,
            game_replay,
            args.reward_mode,
            calculator,
            generated_exprs,
            pool,
        )
        accepted_records.extend(summary["records"])
        copy_replay(game_replay, train_replay if game_index < split else holdout_replay)

    return train_replay, holdout_replay, accepted_records


def evaluate_budget_candidate(
    args,
    budget,
    seed,
    policy_net,
    value_net,
    mcts_mod,
    device,
    calculator,
    pool,
):
    seed_everything(seed, deterministic=True)
    pool_snapshot = _snapshot_pool(pool) if pool is not None else None
    old_ic = float(pool.evaluate_ensemble()) if pool is not None else math.nan
    replay = ReplayBuffer(10000)
    row = {
        "mcts_sim": budget,
        "seed": seed,
        "completed": 0,
        "accepted": 0,
        "single_ic": math.nan,
        "single_abs_ic": math.nan,
        "pool_ic_before": old_ic,
        "pool_ic_after": old_ic,
        "pool_ic_delta": 0.0 if pool is not None else math.nan,
        "trajectory_reward_mean": -1.0,
        "trajectory_reward_max": -1.0,
        "expression": "",
    }
    try:
        game = self_play_game(
            "cfg-sem-k",
            policy_net,
            value_net,
            mcts_mod,
            args.max_expression_length,
            budget,
            1,
            1,
            device,
            1.0,
            1.0,
            0.0,
            0.03,
        )
        if game and any(step["terminal"] for step in game):
            row["completed"] = 1
            expression = _final_expression("cfg-sem-k", game[-1]["final_state"])
            row["expression"] = str(expression)
            single_ic = float(calculator.calc_single_IC_ret(expression))
            row["single_ic"] = single_ic
            row["single_abs_ic"] = abs(single_ic)
            summary = process_rewards(
                "cfg-sem-k",
                [game],
                replay,
                args.reward_mode,
                calculator,
                set(),
                pool,
            )
            row["accepted"] = int(summary["factor_count"] > 0)
            if pool is not None:
                new_ic = float(pool.evaluate_ensemble())
                row["pool_ic_after"] = new_ic
                row["pool_ic_delta"] = new_ic - old_ic
            rewards = [sample[2] for sample in replay.buffer]
            if rewards:
                row["trajectory_reward_mean"] = float(np.mean(rewards))
                row["trajectory_reward_max"] = float(np.max(rewards))
    except Exception as exc:  # Preserve failed paired trials in the report.
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if pool is not None:
            _restore_pool(pool, pool_snapshot)
    return row


def summarize_budgets(rows) -> list[dict]:
    summaries = []
    budgets = sorted({row["mcts_sim"] for row in rows})
    for budget in budgets:
        selected = [row for row in rows if row["mcts_sim"] == budget]
        finite_ic = [row["single_abs_ic"] for row in selected if math.isfinite(row["single_abs_ic"])]
        summaries.append({
            "mcts_sim": budget,
            "trials": len(selected),
            "completion_rate": float(np.mean([row["completed"] for row in selected])),
            "acceptance_rate": float(np.mean([row["accepted"] for row in selected])),
            "mean_single_abs_ic": float(np.mean(finite_ic)) if finite_ic else math.nan,
            "mean_pool_ic_delta": (
                float(np.mean([row["pool_ic_delta"] for row in selected]))
                if all(math.isfinite(row["pool_ic_delta"]) for row in selected)
                else math.nan
            ),
            "mean_trajectory_reward": float(np.mean([row["trajectory_reward_mean"] for row in selected])),
        })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict) -> None:
    policy = payload["policy_fit"]
    value = payload["value_fit"]
    lines = [
        "# MCTS and Network Training Target Validation",
        "",
        "## Policy Fit to the MCTS Distribution",
        "",
        f"- Training KL: {policy['train_before']['kl']:.6g} -> {policy['train_after']['kl']:.6g}",
        f"- Holdout KL: {policy['holdout_before']['kl']:.6g} -> {policy['holdout_after']['kl']:.6g}",
        "",
        (
            "## Value Fit to the Similarity-Penalized Reward"
            if payload["config"]["reward_mode"] == "pool"
            else "## Value Fit to the Single-Factor abs(IC) Reward"
        ),
        "",
        f"- Training MSE: {value['train_before']['mse']:.6g} -> {value['train_after']['mse']:.6g}",
        f"- Training correlation: {value['train_before']['correlation']:.6g} -> {value['train_after']['correlation']:.6g}",
        f"- Holdout MSE: {value['holdout_before']['mse']:.6g} -> {value['holdout_after']['mse']:.6g}",
        f"- Holdout correlation: {value['holdout_before']['correlation']:.6g} -> {value['holdout_after']['correlation']:.6g}",
        "",
        "## MCTS Simulation Budget Comparison",
        "",
        "| Simulations | Completion Rate | Acceptance Rate | Mean Single-Factor abs(IC) | Mean Pool IC Gain | Mean Trajectory Reward |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["budget_summary"]:
        lines.append(
            f"| {row['mcts_sim']} | {row['completion_rate']:.4f} | "
            f"{row['acceptance_rate']:.4f} | {row['mean_single_abs_ic']:.6g} | "
            f"{row['mean_pool_ic_delta']:.6g} | {row['mean_trajectory_reward']:.6g} |"
        )
    policy_confirmed = (
        policy["train_after"]["kl"] < policy["train_before"]["kl"]
        and policy["holdout_after"]["kl"] < policy["holdout_before"]["kl"]
    )
    value_train_fit = value["train_after"]["mse"] < value["train_before"]["mse"]
    value_generalized = value["holdout_after"]["mse"] < value["holdout_before"]["mse"]
    budget_rows = payload["budget_summary"]
    simulation_monotonic = all(
        right["mean_single_abs_ic"] >= left["mean_single_abs_ic"]
        and (
            payload["config"]["reward_mode"] == "single"
            or right["mean_pool_ic_delta"] >= left["mean_pool_ic_delta"]
        )
        for left, right in zip(budget_rows, budget_rows[1:])
    )
    lines.extend([
        "",
        "## Conclusions",
        "",
        f"- Policy fits the MCTS action distribution: {'PASS' if policy_confirmed else 'FAIL'}.",
        f"- Value fits the training trajectory rewards: {'PASS' if value_train_fit else 'FAIL'}.",
        f"- Value generalizes to holdout trajectories: {'PASS' if value_generalized else 'FAIL'}.",
        f"- More simulations produce monotonic improvement: {'PASS' if simulation_monotonic else 'FAIL'}.",
        "",
        "MCTS search relies on value predictions instead of calculating the true IC at every leaf.",
        "More simulations therefore optimize the current network estimate more thoroughly; they do not guarantee monotonically better factors when value generalization is weak.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)
    output_dir = OUTPUT_ROOT / "cfg-sem-k" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with experiment_import_context("cfg-sem-k", "lstm"):
        mods = _import_runtime_modules("cfg-sem-k", "lstm")
        feature, embed_dim = _make_feature_extractor(mods["networks.feature_"], "cfg-sem-k", device)
        policy_net = mods["networks.policy_"].PolicyNetwork(feature, embed_dim).to(device)
        value_net = mods["networks.value_"].ValueNetwork(feature, embed_dim).to(device)
        checkpoint_iteration = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            policy_net.load_state_dict(checkpoint["policy_state_dict"])
            value_net.load_state_dict(checkpoint["value_state_dict"])
            checkpoint_iteration = checkpoint.get("iteration")

        data = StockData(args.instrument, args.train_start, args.train_end, device=device)
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -args.target_horizon) / close - 1
        calculator = QLibStockDataCalculator(data, target)
        pool = (
            MseAlphaPool(
                args.pool_capacity,
                calculator,
                ic_lower_bound=None,
                l1_alpha=5e-3,
                device=device,
            )
            if args.reward_mode == "pool"
            else None
        )

        train_replay, holdout_replay, records = collect_real_trajectories(
            args,
            policy_net,
            value_net,
            mods["mcts"],
            device,
            calculator,
            pool,
        )

        policy_before_train = policy_metrics(policy_net, train_replay, device)
        policy_before_holdout = policy_metrics(policy_net, holdout_replay, device)
        optimizer = torch.optim.Adam(_dedup_parameters(policy_net, value_net), lr=args.learning_rate)
        policy_train_stats = train_policy(
            "cfg-sem-k",
            policy_net,
            optimizer,
            train_replay,
            args.fit_batches,
            args.batch_size,
            device,
        )
        policy_after_train = policy_metrics(policy_net, train_replay, device)
        policy_after_holdout = policy_metrics(policy_net, holdout_replay, device)

        value_before_train = value_metrics(value_net, train_replay)
        value_before_holdout = value_metrics(value_net, holdout_replay)
        value_train_stats = train_value(
            "cfg-sem-k",
            value_net,
            optimizer,
            train_replay,
            args.fit_batches,
            args.batch_size,
            device,
        )
        value_after_train = value_metrics(value_net, train_replay)
        value_after_holdout = value_metrics(value_net, holdout_replay)

        budget_rows = []
        for budget in args.sim_budgets:
            for offset in range(args.budget_seeds):
                budget_rows.append(evaluate_budget_candidate(
                    args,
                    budget,
                    args.seed + 10000 + offset,
                    policy_net,
                    value_net,
                    mods["mcts"],
                    device,
                    calculator,
                    pool,
                ))

    payload = {
        "config": vars(args),
        "trajectory_data": {
            "train_states": len(train_replay),
            "holdout_states": len(holdout_replay),
            "accepted_factors": len(records),
            "pool_size": pool.size if pool is not None else 0,
            "checkpoint_iteration": checkpoint_iteration,
        },
        "policy_fit": {
            "train_before": policy_before_train,
            "train_after": policy_after_train,
            "holdout_before": policy_before_holdout,
            "holdout_after": policy_after_holdout,
            "optimizer_stats": policy_train_stats,
        },
        "value_fit": {
            "train_before": value_before_train,
            "train_after": value_after_train,
            "holdout_before": value_before_holdout,
            "holdout_after": value_after_holdout,
            "optimizer_stats": value_train_stats,
        },
        "budget_summary": summarize_budgets(budget_rows),
    }
    write_json(output_dir / "validation_summary.json", payload)
    write_csv(output_dir / "mcts_budget_trials.csv", budget_rows)
    write_report(output_dir / "validation_report.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
