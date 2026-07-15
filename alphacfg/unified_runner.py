"""Unified MCTS training runner for AlphaCFG method variants."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from alphacfg.data.calculator__ import QLibStockDataCalculator
from alphacfg.data.evaluator_ import parse_expression_from_string
from alphacfg.data.expression import Feature, Ref, OutOfDataRangeError
from alphacfg.data.stock_data import FeatureType, StockData
from alphacfg.experiment_context import LOCAL_NETWORK, NETWORK_BACKENDS, default_network, validate_network
from alphacfg.grammar.tree_similarity import expression_to_prefix_str
from alphacfg.models.linear_alpha_pool import MseAlphaPool
from alphacfg.runtime import build_output_dir, write_json, write_run_files
from alphacfg.seeding import seed_everything


LOG = logging.getLogger(__name__)

POOL_REWARD_FORMULA = "(1 - max_tree_similarity) * max(new_pool_ic, 0)"
SINGLE_REWARD_FORMULA = "abs(single_ic), duplicate=0"


SINGLE_TRAINING_HEADER = [
    "iteration",
    "generated_games",
    "terminal_expressions",
    "generated_factors",
    "average_abs_ic",
    "replay_samples",
    "pool_size",
    "pool_eval_count",
    "average_policy_loss",
    "policy_gradient_norm",
    "policy_parameter_delta",
    "average_value_loss",
    "value_gradient_norm",
    "value_parameter_delta",
]

POOL_TRAINING_HEADER = [
    "iteration",
    "train_ic",
    "valid_ic",
    "test_ic",
    "pool_size",
    "pool_eval_count",
    "policy_loss",
    "value_loss",
    "average_reward",
    "average_tree_similarity",
]


class ReplayBuffer:
    """Shared replay buffer for CFG and RPN states."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_input, target_policy, reward, aux):
        self.buffer.append((state_input, target_policy, float(reward), aux))

    def sample(self, batch_size: int):
        if not self.buffer:
            return []
        items = list(self.buffer)
        if len(items) < batch_size:
            return random.choices(items, k=batch_size)
        return random.sample(items, batch_size)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified AlphaCFG MCTS runner.")
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--reward-mode", choices=["pool", "single", "mask"], default="pool")
    parser.add_argument(
        "--mask-expression",
        default="",
        help="Initial non-terminal prefix expression for mask mode, e.g. 'Mul1 Q Rank CSRank Q num'.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pool-capacity", type=int, default=10)
    parser.add_argument("--max-expression-length", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--inner-batches", type=int, default=2)
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-buffer-size", type=int, default=20000)
    parser.add_argument("--mcts-sim", type=int, default=64)
    parser.add_argument("--mcts-parallel", type=int, default=8)
    parser.add_argument("--batch-size-eval", type=int, default=2)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--root-dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--instrument", default="csi300")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--valid-start", default="2018-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2021-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--target-horizon", type=int, default=20)
    parser.add_argument("--network-name", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--log-dir", default="")
    return parser.parse_args(argv)


def setup_logging() -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _variant_from_cwd(default: str = "") -> str:
    if default:
        return default
    name = Path.cwd().name.replace("_", "-")
    return name if name in {"rpn", "cfg-syn", "cfg-sem", "cfg-sem-k"} else "cfg-sem-k"


def _clear_network_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name in {"mcts", "networks", "env", "tree"}
            or module_name.startswith("networks.")
            or module_name.startswith("env.")
        ):
            del sys.modules[module_name]


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _configure_network_imports(variant: str, network: str) -> None:
    """Put the selected feature backend before the method-local modules."""
    _clear_network_modules()
    method_root = Path.cwd()
    backend_root = None if network == LOCAL_NETWORK else NETWORK_BACKENDS[network]
    ordered_paths = [path for path in (backend_root, method_root) if path is not None]
    for path in reversed(ordered_paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _import_runtime_modules(variant: str, network: str):
    _configure_network_imports(variant, network)
    method_root = Path.cwd()
    modules = {"networks.feature_": importlib.import_module("networks.feature_")}
    for leaf in ("policy_", "value_"):
        local_path = method_root / "networks" / f"{leaf}.py"
        if local_path.exists():
            modules[f"networks.{leaf}"] = _load_module_from_path(f"networks.{leaf}", local_path)
        else:
            modules[f"networks.{leaf}"] = importlib.import_module(f"networks.{leaf}")
    modules["mcts"] = importlib.import_module("mcts")
    return modules


def _make_feature_extractor(feature_mod, variant: str, device: torch.device):
    embed_dim = 128
    if variant == "rpn":
        token_mod = importlib.import_module("env.token_v")
        vocab_size = max(token_mod.ENV_TOKEN_MAP.values()) + 1
        return feature_mod.FeatureExtractor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            nlayers=2,
            dropout=0.1,
        ).to(device), embed_dim
    return feature_mod.FeatureExtractor(embed_dim=embed_dim, h_size=embed_dim, dropout=0.1).to(device), embed_dim


def _dedup_parameters(*modules):
    return list(dict.fromkeys(param for module in modules for param in module.parameters()))


def _parameter_snapshot(parameters):
    return [param.detach().clone() for param in parameters]


def _parameter_delta(before, parameters) -> float:
    return sum((p.detach() - old).float().pow(2).sum().item() for old, p in zip(before, parameters)) ** 0.5


def _gradient_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def _terminal_policy_placeholder(raw_pi):
    if isinstance(raw_pi, dict):
        return {key: 0.0 for key in raw_pi}
    if torch.is_tensor(raw_pi):
        return torch.zeros_like(raw_pi)
    if isinstance(raw_pi, np.ndarray):
        return np.zeros_like(raw_pi)
    return {}


def _state_input_for_training(variant: str, state, continue_out_game=None):
    if variant == "rpn":
        token_mod = importlib.import_module("env.token_v")
        return token_mod.convert_state_tokens_to_ids(state.state), state.get_possible_actions()
    return state.state_description(), continue_out_game


def _final_expression(variant: str, state):
    if variant == "rpn":
        return state.expression
    return parse_expression_from_string(state.state_description())


def _target_policy_from_raw(raw_pi):
    if not isinstance(raw_pi, dict):
        return []
    return [
        {"category": category, "action": action, "prob": float(prob)}
        for (category, action), prob in raw_pi.items()
    ]


def _build_initial_mask_state(variant: str, expression: str, max_length: int):
    if not expression:
        return None
    if variant != "cfg-sem-k":
        raise ValueError("mask mode currently supports cfg-sem-k only")
    env_mod = importlib.import_module("env.alpha_env")
    env = env_mod.AlphaEnv()
    env.set_state_from_expression(expression)
    return env


def _initial_continue_out_game(mcts_mod, state):
    node = mcts_mod.Node(state=state)
    return mcts_mod.CFGSemKMethods.child_continue_out_game(state, node.continue_out_game) if hasattr(mcts_mod, "CFGSemKMethods") else node.continue_out_game


def self_play_game(
    variant: str,
    policy_net,
    value_net,
    mcts_mod,
    max_length: int,
    mcts_sim: int,
    mcts_parallel: int,
    batch_size_eval: int,
    device: torch.device,
    c_puct: float,
    temperature: float,
    root_dirichlet_epsilon: float,
    root_dirichlet_alpha: float,
    mask_expression: str = "",
):
    if hasattr(mcts_mod, "CFGSemKMethods"):
        mcts_mod.CFGSemKMethods.max_expression_length = max_length
    initial_state = _build_initial_mask_state(variant, mask_expression, max_length)
    mcts = mcts_mod.MCTS(
        policy_net,
        value_net,
        mcts_sim,
        mcts_parallel,
        batch_size_eval,
        device,
        root_dirichlet_epsilon=root_dirichlet_epsilon,
        root_dirichlet_alpha=root_dirichlet_alpha,
    )
    if initial_state is not None:
        mcts.root = mcts_mod.Node(
            state=initial_state,
            continue_out_game=_initial_continue_out_game(mcts_mod, initial_state),
        )
    game = []
    length = 0
    while (not mcts.root.state.is_terminal_state()) and length <= max_length:
        state_input, aux = _state_input_for_training(
            variant,
            mcts.root.state,
            getattr(mcts.root, "continue_out_game", None),
        )
        raw_pi, move = mcts.search(False, c_puct=c_puct, temperature=temperature)
        if move is None:
            break
        game.append({
            "state_input": state_input,
            "aux": aux,
            "raw_pi": raw_pi,
            "terminal": False,
            "final_state": mcts.root.state,
        })
        length += 1

    if mcts.root.state.is_terminal_state():
        for step in game:
            step["terminal"] = True
            step["final_state"] = mcts.root.state
        if game:
            state_input, aux = _state_input_for_training(
                variant,
                mcts.root.state,
                getattr(mcts.root, "continue_out_game", None),
            )
            game.append({
                "state_input": state_input,
                "aux": aux,
                "raw_pi": _terminal_policy_placeholder(game[0]["raw_pi"]),
                "terminal": True,
                "final_state": mcts.root.state,
            })
    return game


def parallel_self_play(variant: str, args, policy_net, value_net, mcts_mod, device, iteration: int):
    # Keep this single-process by default. Multiprocessing with full model objects
    # is expensive and was a major source of duplicated legacy code.
    games = []
    for idx in range(args.num_games):
        seed_everything(args.seed + iteration * 1000 + idx, deterministic=False)
        games.append(self_play_game(
            variant,
            policy_net,
            value_net,
            mcts_mod,
            args.max_expression_length,
            args.mcts_sim,
            args.mcts_parallel,
            args.batch_size_eval,
            device,
            args.c_puct,
            args.temperature,
            args.root_dirichlet_epsilon,
            args.root_dirichlet_alpha,
            args.mask_expression,
        ))
    return games


def _snapshot_pool(pool) -> dict[str, Any]:
    """保存 pool 的可回滚状态，用于拒绝让组合 IC 下降的新因子。"""
    return {
        "size": pool.size,
        "exprs": list(pool.exprs),
        "single_ics": pool.single_ics.copy(),
        "weights": pool._weights.copy(),
        "mutual_ics": pool._mutual_ics.copy(),
        "extra_info": list(pool._extra_info),
        "best_obj": pool.best_obj,
        "best_ic_ret": pool.best_ic_ret,
        "update_history": list(pool.update_history),
        "failure_cache": set(pool._failure_cache),
        "eval_cnt": pool.eval_cnt,
    }


def _restore_pool(pool, snapshot: dict[str, Any], eval_cnt: int | None = None) -> None:
    """恢复 pool 状态；eval_cnt 可选择保留尝试次数。"""
    pool.size = snapshot["size"]
    pool.exprs = list(snapshot["exprs"])
    pool.single_ics = snapshot["single_ics"].copy()
    pool._weights = snapshot["weights"].copy()
    pool._mutual_ics = snapshot["mutual_ics"].copy()
    pool._extra_info = list(snapshot["extra_info"])
    pool.best_obj = snapshot["best_obj"]
    pool.best_ic_ret = snapshot["best_ic_ret"]
    pool.update_history = list(snapshot["update_history"])
    pool._failure_cache = set(snapshot["failure_cache"])
    pool.eval_cnt = snapshot["eval_cnt"] if eval_cnt is None else eval_cnt


def _max_tree_similarity_before_insert(pool, expression_or_prefix) -> float:
    """Measure one trajectory state's novelty before inserting its final candidate."""
    if pool.size == 0:
        return 0.0
    prefix_expression = (
        expression_or_prefix
        if isinstance(expression_or_prefix, str)
        else expression_to_prefix_str(expression_or_prefix)
    )
    similarity = float(pool.compute_avg_similarity_with_pool(prefix_expression))
    if not math.isfinite(similarity):
        raise ValueError("tree similarity is not finite")
    return min(1.0, max(0.0, similarity))


def _accepted_pool_reward(new_pool_ic: float, max_tree_similarity: float) -> float:
    """Reward a strong pool while discounting candidates structurally close to it."""
    return (1.0 - max_tree_similarity) * max(new_pool_ic, 0.0)


def _trajectory_tree_similarities_before_insert(pool, game, final_expression) -> list[float]:
    """Calculate a separate max-pool similarity for every state in a trajectory."""
    final_prefix = expression_to_prefix_str(final_expression)
    return [
        _max_tree_similarity_before_insert(
            pool,
            step["state_input"] if isinstance(step["state_input"], str) else final_prefix,
        )
        for step in game
    ]


def process_rewards(
    variant: str,
    games,
    replay: ReplayBuffer,
    mode: str,
    calculator,
    generated_exprs: set[str],
    pool=None,
):
    records = []
    for game in games:
        final_expr = None
        expr_key = ""
        ic = 0.0
        reward = -1.0
        max_tree_similarity = 0.0
        trajectory_tree_similarities = []
        trajectory_rewards = None
        duplicate = False
        cache_expr = False
        record_factor = False
        for step in reversed(game):
            if step["terminal"]:
                try:
                    final_expr = _final_expression(variant, step["final_state"])
                    if final_expr is None:
                        raise ValueError("missing final expression")
                    expr_key = str(final_expr)
                    if expr_key in generated_exprs:
                        duplicate = True
                        reward = 0.0
                        break
                    cache_expr = True
                    if mode in {"single", "mask"}:
                        ic = float(calculator.calc_single_IC_ret(final_expr))
                        if not math.isfinite(ic):
                            raise ValueError("single IC is not finite")
                        reward = abs(ic)
                        record_factor = True
                    else:
                        old_ic = float(pool.evaluate_ensemble())
                        trajectory_tree_similarities = _trajectory_tree_similarities_before_insert(
                            pool, game, final_expr
                        )
                        max_tree_similarity = max(trajectory_tree_similarities, default=0.0)
                        pool_snapshot = _snapshot_pool(pool)
                        old_eval_cnt = pool.eval_cnt
                        pool.try_new_expr(final_expr)
                        attempted_eval_cnt = pool.eval_cnt
                        new_ic = float(pool.evaluate_ensemble())
                        if attempted_eval_cnt == old_eval_cnt:
                            ic = 0.0
                            reward = -1.0
                        elif new_ic < old_ic:
                            _restore_pool(pool, pool_snapshot, eval_cnt=attempted_eval_cnt)
                            ic = old_ic
                            reward = 0.0
                        else:
                            ic = new_ic
                            trajectory_rewards = [
                                _accepted_pool_reward(ic, similarity)
                                for similarity in trajectory_tree_similarities
                            ]
                            reward = float(np.mean(trajectory_rewards)) if trajectory_rewards else 0.0
                            record_factor = True
                except (OutOfDataRangeError, ValueError, AttributeError, TypeError, IndexError, KeyError):
                    ic = 0.0
                    reward = -1.0
                break
        if not expr_key:
            if game:
                for step in game:
                    replay.push(
                        step["state_input"],
                        _target_policy_from_raw(step["raw_pi"]),
                        -1.0,
                        step["aux"],
                    )
            continue
        if duplicate:
            for step in game:
                replay.push(
                    step["state_input"],
                    _target_policy_from_raw(step["raw_pi"]),
                    reward,
                    step["aux"],
                )
            continue
        if cache_expr:
            generated_exprs.add(expr_key)
        if record_factor:
            if trajectory_rewards is None:
                trajectory_rewards = [reward] * len(game)
                trajectory_tree_similarities = [0.0] * len(game)
            records.append({
                "expression": expr_key,
                "ic": ic,
                "abs_ic": abs(ic),
                "reward": reward,
                "max_tree_similarity": max_tree_similarity,
                "trajectory_rewards": trajectory_rewards or [],
                "trajectory_tree_similarities": trajectory_tree_similarities,
            })
        for step_index, step in enumerate(game):
            step_reward = (
                trajectory_rewards[step_index]
                if trajectory_rewards is not None
                else reward
            )
            replay.push(
                step["state_input"],
                _target_policy_from_raw(step["raw_pi"]),
                step_reward,
                step["aux"],
            )
    accepted_rewards = [
        reward
        for record in records
        for reward in record["trajectory_rewards"]
    ]
    accepted_similarities = [
        similarity
        for record in records
        for similarity in record["trajectory_tree_similarities"]
    ]
    return {
        "records": records,
        "factor_count": len(records),
        "average_abs_ic": float(np.mean([r["abs_ic"] for r in records])) if records else 0.0,
        "average_reward": float(np.mean(accepted_rewards)) if accepted_rewards else 0.0,
        "average_tree_similarity": float(np.mean(accepted_similarities)) if accepted_similarities else 0.0,
    }


def _policy_outputs(variant: str, policy_net, state_input, aux):
    if variant == "rpn":
        device = next(policy_net.parameters()).device
        src = torch.tensor(state_input, dtype=torch.long, device=device).unsqueeze(0)
        return policy_net(src, possible_actions=aux)

    token_mod = importlib.import_module("env.token_v")
    token_map = token_mod.create_token_map()
    tokens = token_mod.parse_expression_to_tokens(state_input, token_map)
    first_token = token_mod.get_first_token(tokens, token_map)
    try:
        return policy_net(state_input, first_token=first_token, continue_out_game=aux)
    except TypeError:
        return policy_net(state_input, first_token=first_token)


def _value_output(variant: str, value_net, state_input):
    if variant == "rpn":
        device = next(value_net.parameters()).device
        src = torch.tensor(state_input, dtype=torch.long, device=device).unsqueeze(0)
        return value_net(src).squeeze(-1)
    return value_net(state_input).squeeze(-1)


def _policy_loss(action_outputs, target_policy, device):
    if not action_outputs or not target_policy:
        return None
    lookup = {(item["category"], item["action"]): float(item["prob"]) for item in target_policy}
    output_probs = torch.stack([item["prob"] for item in action_outputs]).to(device)
    target = torch.tensor(
        [lookup.get((item["category"], item["action"]), 0.0) for item in action_outputs],
        dtype=output_probs.dtype,
        device=device,
    )
    if target.sum() <= 0:
        return None
    target = target / target.sum()
    output = output_probs / (output_probs.sum() + 1e-10)
    return nn.KLDivLoss(reduction="batchmean")(torch.log(output + 1e-10), target)


def train_policy(variant, policy_net, optimizer, replay, num_batches, batch_size, device):
    policy_net.train()
    total = 0.0
    grad = 0.0
    valid = 0
    for _ in range(num_batches):
        batch = replay.sample(batch_size)
        if not batch:
            continue
        optimizer.zero_grad()
        loss_sum = torch.tensor(0.0, device=device)
        count = 0
        for state_input, target_policy, _, aux in batch:
            loss = _policy_loss(_policy_outputs(variant, policy_net, state_input, aux), target_policy, device)
            if loss is None:
                continue
            loss_sum = loss_sum + loss
            count += 1
        if count:
            loss_sum = loss_sum / count
            loss_sum.backward()
            grad += _gradient_norm(policy_net.parameters())
            optimizer.step()
            total += float(loss_sum.detach().cpu())
            valid += 1
    return {"loss": total / valid if valid else 0.0, "gradient_norm": grad / valid if valid else 0.0}


def train_value(variant, value_net, optimizer, replay, num_batches, batch_size, device):
    value_net.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    grad = 0.0
    valid = 0
    for _ in range(num_batches):
        batch = replay.sample(batch_size)
        if not batch:
            continue
        optimizer.zero_grad()
        loss_sum = torch.tensor(0.0, device=device)
        for state_input, _, reward, _ in batch:
            pred = _value_output(variant, value_net, state_input)
            target = torch.tensor([reward], dtype=torch.float32, device=device)
            loss_sum = loss_sum + loss_fn(pred, target)
        loss = loss_sum / len(batch)
        loss.backward()
        grad += _gradient_norm(value_net.parameters())
        optimizer.step()
        total += float(loss.detach().cpu())
        valid += 1
    return {"loss": total / valid if valid else 0.0, "gradient_norm": grad / valid if valid else 0.0}


def _ensure_header(path: Path, header: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        csv.writer(file_obj).writerow(header)


def _write_sorted_single_factors(path: Path, records: dict[str, dict[str, Any]]) -> None:
    """Persist unique single-factor records sorted by absolute IC."""
    rows = sorted(records.values(), key=lambda row: abs(float(row["ic"])), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["iteration", "expression", "ic", "abs_ic"])
        for row in rows:
            writer.writerow([
                row["iteration"],
                row["expression"],
                row["ic"],
                abs(float(row["ic"])),
            ])


def _run_name(args, variant: str, network_name: str) -> str:
    effective_network = network_name
    parts = [
        variant,
        args.reward_mode,
        effective_network or "local",
        f"pool{args.pool_capacity}",
        f"len{args.max_expression_length}",
    ]
    if args.run_tag:
        parts.append(args.run_tag)
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def main(default_variant: str = "") -> None:
    args = parse_args()
    variant = _variant_from_cwd(args.variant_name or default_variant)
    network_name = validate_network(variant, args.network_name or default_network(variant))
    setup_logging()
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    mods = _import_runtime_modules(variant, network_name)
    feature_extractor, embed_dim = _make_feature_extractor(mods["networks.feature_"], variant, device)
    policy_net = mods["networks.policy_"].PolicyNetwork(feature_extractor, embed_dim).to(device)
    value_net = mods["networks.value_"].ValueNetwork(feature_extractor, embed_dim).to(device)
    optimizer = torch.optim.Adam(_dedup_parameters(policy_net, value_net), lr=args.learning_rate)

    train_data = StockData(args.instrument, args.train_start, args.train_end, device=device)
    valid_data = StockData(args.instrument, args.valid_start, args.valid_end, device=device)
    test_data = StockData(args.instrument, args.test_start, args.test_end, device=device)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -args.target_horizon) / close - 1
    calculator_train = QLibStockDataCalculator(train_data, target)
    calculator_valid = QLibStockDataCalculator(valid_data, target)
    calculator_test = QLibStockDataCalculator(test_data, target)

    effective_network = network_name
    run_name = _run_name(args, variant, network_name)
    output_dir = build_output_dir(variant, run_name if not args.log_dir else str(Path(args.log_dir) / run_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_files(output_dir, args, {
        "variant": variant,
        "mode": args.reward_mode,
        "device": str(device),
        "effective_network": effective_network,
        "reward_formula": POOL_REWARD_FORMULA if args.reward_mode == "pool" else SINGLE_REWARD_FORMULA,
    })
    metrics_file = output_dir / "training_metrics.csv"
    single_file = output_dir / "single_factors.csv"
    pool_file = output_dir / "pool_dicts.json"
    _ensure_header(metrics_file, POOL_TRAINING_HEADER if args.reward_mode == "pool" else SINGLE_TRAINING_HEADER)
    if args.reward_mode in {"single", "mask"}:
        _ensure_header(single_file, ["iteration", "expression", "ic", "abs_ic"])

    pool = None
    if args.reward_mode == "mask" and not args.mask_expression:
        raise ValueError("--reward-mode mask requires --mask-expression")
    if args.reward_mode == "pool":
        pool = MseAlphaPool(args.pool_capacity, calculator_train, ic_lower_bound=None, l1_alpha=5e-3, device=device)
    replay = ReplayBuffer(args.replay_buffer_size)
    generated_exprs: set[str] = set()
    single_records: dict[str, dict[str, Any]] = {}
    last_pool_metrics = None
    last_single_summary = None
    best_valid = -float("inf")
    stale = 0
    patience = args.early_stop_patience or max(1, int(args.num_iterations * 0.2))

    LOG.info("Run outputs: %s", output_dir)
    for iteration in range(1, args.num_iterations + 1):
        games = []
        for _ in range(args.inner_batches):
            games.extend(parallel_self_play(variant, args, policy_net, value_net, mods["mcts"], device, iteration))
        summary = process_rewards(
            variant,
            games,
            replay,
            args.reward_mode,
            calculator_train,
            generated_exprs,
            pool,
        )
        if args.reward_mode in {"single", "mask"}:
            last_single_summary = {"iteration": iteration, **{k: summary[k] for k in ("factor_count", "average_abs_ic")}}
            for record in summary["records"]:
                single_records[record["expression"]] = {
                    "iteration": iteration,
                    "expression": record["expression"],
                    "ic": record["ic"],
                }
            _write_sorted_single_factors(single_file, single_records)

        p_before = _parameter_snapshot(policy_net.parameters())
        p_stats = train_policy(variant, policy_net, optimizer, replay, args.num_batches, args.batch_size, device)
        p_delta = _parameter_delta(p_before, policy_net.parameters())
        v_before = _parameter_snapshot(value_net.parameters())
        v_stats = train_value(variant, value_net, optimizer, replay, args.num_batches, args.batch_size, device)
        v_delta = _parameter_delta(v_before, value_net.parameters())

        terminal = sum(1 for game in games if any(step["terminal"] for step in game))
        torch.save({
            "iteration": iteration,
            "policy_state_dict": policy_net.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "reward_mode": args.reward_mode,
            "reward_formula": POOL_REWARD_FORMULA if args.reward_mode == "pool" else SINGLE_REWARD_FORMULA,
        }, output_dir / "checkpoint_latest.pt")

        if pool is not None and pool.size > 0:
            train_ic, train_ric = pool.test_ensemble(calculator_train)
            valid_ic, valid_ric = pool.test_ensemble(calculator_valid)
            test_ic, test_ric = pool.test_ensemble(calculator_test)
            last_pool_metrics = {
                "train_ic": train_ic, "train_rank_ic": train_ric,
                "valid_ic": valid_ic, "valid_rank_ic": valid_ric,
                "test_ic": test_ic, "test_rank_ic": test_ric,
            }
            with pool_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps({"iteration": iteration, "pool_dict": pool.to_json_dict(), **last_pool_metrics}, ensure_ascii=False) + "\n")
            with metrics_file.open("a", newline="", encoding="utf-8") as file_obj:
                csv.writer(file_obj).writerow([
                    iteration,
                    train_ic,
                    valid_ic,
                    test_ic,
                    pool.size,
                    pool.eval_cnt,
                    p_stats["loss"],
                    v_stats["loss"],
                    summary["average_reward"],
                    summary["average_tree_similarity"],
                ])
            if valid_ic > best_valid:
                best_valid = valid_ic
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        elif pool is not None:
            with metrics_file.open("a", newline="", encoding="utf-8") as file_obj:
                csv.writer(file_obj).writerow([
                    iteration,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    pool.eval_cnt,
                    p_stats["loss"],
                    v_stats["loss"],
                    summary["average_reward"],
                    summary["average_tree_similarity"],
                ])
        elif pool is None:
            with metrics_file.open("a", newline="", encoding="utf-8") as file_obj:
                csv.writer(file_obj).writerow([
                    iteration, len(games), terminal, summary["factor_count"],
                    summary["average_abs_ic"], len(replay), 0, 0,
                    p_stats["loss"], p_stats["gradient_norm"], p_delta,
                    v_stats["loss"], v_stats["gradient_norm"], v_delta,
                ])
        replay.clear()

    write_json(output_dir / "run_summary.json", {
        "completed_iterations": iteration,
        "pool_size": pool.size if pool is not None else 0,
        "pool_eval_count": pool.eval_cnt if pool is not None else 0,
        "last_pool_metrics": last_pool_metrics,
        "last_single_summary": last_single_summary,
        "reward_formula": POOL_REWARD_FORMULA if args.reward_mode == "pool" else SINGLE_REWARD_FORMULA,
    })
    LOG.info("Search completed.")
