# train_preparation.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import copy
import os
import random
import numpy as np
import pandas as pd
import json
import csv
import logging
from collections import deque
from functools import partial
from torch.nn.utils.rnn import pad_sequence  # <-- For padding
from env.token_v import convert_state_tokens_to_ids
from mcts import MCTS
from networks.policy_ import PolicyNetwork, evaluate_policy_network
from networks.value_ import ValueNetwork, evaluate_expression_value
from env.alpha_env import AlphaEnv
from data.evaluator_ import *
from data.calculator__ import *
from models.linear_alpha_pool import MseAlphaPool
from networks.feature_ import FeatureExtractor

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#####################################
#           ReplayBuffer            #
#####################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_probs_list, value, State):
        """
        Push data into the buffer
        """
        self.buffer.append((state, action_probs_list, value, State))

    def sample(self, batch_size):
        """
        Random sampling: Select batch_size data from the buffer.
        If there aren't enough samples in the buffer, sampling with replacement is used to ensure batch_size data are returned.
        """
        if len(self.buffer) == 0:
            return []
        buffer_list = list(self.buffer)
        if len(buffer_list) < batch_size:
            # Sampling with replacement if there are not enough samples
            return random.choices(buffer_list, k=batch_size)
        else:
            # Otherwise, random sampling without replacement
            return random.sample(buffer_list, batch_size)

    def __len__(self):
        return len(self.buffer)


#####################################
#     mcts_policy_to_target         #
#####################################
def mcts_policy_to_target(action_probs_dict):
    """
    Converts MCTS generated action probabilities (in dictionary form) to a list.
    Input: 
        action_probs_dict: dict, key is (category, action), value is the probability
        Example: {("feature","low"):0.2, ("binaryop","Add"):0.5, ("stop",None):0.3}
    Output:
        target_list: list, each element is like
           { "category": cat, "action": act, "prob": p }
    """
    target_list = []
    for (cat, act), prob in action_probs_dict.items():
        target_list.append({
            "category": cat,
            "action": act,
            "prob": prob
        })
    return target_list


#####################################
#     compute_policy_loss           #
#####################################
def compute_policy_loss(
    action_outputs: torch.Tensor,
    target_policy: torch.Tensor,
    classification_loss_fn,
    device
):
    """
    Compute the KL divergence loss between the policy network output and the target policy.
    Parameters:
      action_outputs: shape = [n_actions], probabilities or logits from the policy network for each action
      target_policy:  shape = [n_actions], the target distribution given by MCTS (matching the order of action_outputs)
    """
    # Apply softmax to the prediction
    pred_dist = torch.softmax(action_outputs, dim=0)

    # Ensure target_policy is a valid probability distribution
    sum_tp = torch.sum(target_policy)
    if sum_tp < 1e-12:
        # If all zeros, return a 0-loss or handle as necessary
        return torch.tensor(0.0, device=device, requires_grad=True)

    target_dist = target_policy / (sum_tp + 1e-12)

    # Using KLDivLoss: KLDivLoss(input, target)
    # Input is usually in log_softmax form
    log_pred = torch.log(pred_dist + 1e-12)
    loss = classification_loss_fn(log_pred, target_dist)
    return loss


#####################################
#        Sequential Batch Training - Policy      #
#####################################
def train_policy_net(policy_net, optimizer, replay_buffer, num_batches, batch_size, device):
    """
    Sequential batch training of the policy network:
      1) Randomly sample a batch from the replay_buffer;
      2) Pad states to obtain states_padded;
      3) For each data in the batch, calculate:
         (a) Policy network output distribution
         (b) Target distribution from MCTS
         (c) Compute KL divergence between them
      4) Backpropagation
    """
    policy_net.train()
    classification_loss_fn = nn.KLDivLoss(reduction='batchmean')

    total_policy_loss = 0.0
    valid_samples = 0

    num_data = len(replay_buffer)
    if num_data == 0:
        return 0.0

    for _ in range(num_batches):
        batch_data = replay_buffer.sample(batch_size)
        if not batch_data:
            continue

        # batch_data: each data is (state_tensor, action_probs_list, value, state_env)
        states, action_probs_list, _, possible_actions = zip(*batch_data)

        # ============ 1) Padding ============= 
        states_padded = pad_sequence(states, batch_first=True, padding_value=0)
        states_padded = states_padded.to(device)

        optimizer.zero_grad()
        policy_loss_batch = torch.tensor(0.0, device=device)

        # ============ 2) Compute Loss for each data in the batch ============ 
        for i in range(len(batch_data)):
            # Target policy (from MCTS) for the current sample --> a list/dictionary
            target_policy_list = action_probs_list[i]  # list of dict

            # Get the input for a single data
            single_input = states_padded[i].unsqueeze(0)  
            # shape = [1, seq_len], assuming the policy network requires batch_size=1

            # Policy network output (list of dict)
            pred_outputs_list = policy_net(single_input, possible_actions=possible_actions[i])

            if (not pred_outputs_list) or (not target_policy_list):
                continue

            # Convert pred_outputs_list to a tensor [n_actions_pred]
            pred_prob_vector = [item["prob"] for item in pred_outputs_list]
            pred_tensor = torch.tensor(
                pred_prob_vector,
                dtype=torch.float32, device=device,
                requires_grad=True
            )

            # Align target_policy_list to the same order as pred_outputs_list
            target_dict = {
                (d["category"], d["action"]): d["prob"]
                for d in target_policy_list
            }

            target_vector = []
            for out_item in pred_outputs_list:
                k = (out_item["category"], out_item["action"])
                target_vector.append(target_dict.get(k, 0.0))
            target_tensor = torch.tensor(target_vector, dtype=torch.float32, device=device)

            # Compute KL divergence
            loss_i = compute_policy_loss(
                pred_tensor, target_tensor,
                classification_loss_fn, device
            )
            policy_loss_batch = policy_loss_batch + loss_i
            valid_samples += 1

        # ============ 3) Backpropagation and accumulate loss ============ 
        if policy_loss_batch.requires_grad and valid_samples > 0:
            policy_loss_batch.backward()
            optimizer.step()
            total_policy_loss += policy_loss_batch.item()

    avg_policy_loss = total_policy_loss / (valid_samples if valid_samples > 0 else 1)
    return avg_policy_loss


#####################################
#        Sequential Batch Training - Value       #
#####################################
def train_value_net(value_net, optimizer, replay_buffer, num_batches, batch_size, device):
    """
    1. Randomly sample batch_size data;
    2. Pad the states;
    3. Perform forward pass, compute mean squared error and backpropagate.
    """
    value_net.train()
    regression_loss_fn = nn.MSELoss()
    total_value_loss = 0.0
    total_samples = 0

    num_data = len(replay_buffer)
    if num_data == 0:
        return 0.0

    for _ in range(num_batches):
        batch_data = replay_buffer.sample(batch_size)
        if not batch_data:
            continue

        states, _, values, _ = zip(*batch_data)

        # Apply padding
        states_padded = pad_sequence(states, batch_first=True, padding_value=0)
        states_padded = states_padded.to(device)

        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        # Forward pass
        value_output = value_net(states_padded).squeeze(-1)

        # Mean squared error
        value_loss = regression_loss_fn(value_output, values_tensor)
        value_loss.backward()
        optimizer.step()

        total_value_loss += value_loss.item()
        total_samples += states_padded.size(0)

    avg_value_loss = total_value_loss / total_samples if total_samples > 0 else 0.0
    return avg_value_loss


#####################################
#         Self-play & Parallel Related       #
#####################################
def self_play_game(policy_net, value_net, LEN, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device=torch.device('cpu')):
    """
    Perform self-play using the current policy network, value network, and MCTS, and return the game data.
    """
    mcts = MCTS(policy_net, value_net, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device)
    game_data = []  # (state_tensor, action_probs, current_expression, reached_terminal, final_state)

    length = 0
    while (not mcts.root.state.is_terminal_state()) and length <= (LEN - 1):
        current_expression_token = mcts.root.state.state
        possible_actions = mcts.root.state.get_possible_actions()
        state_tensor = torch.tensor(convert_state_tokens_to_ids(current_expression_token), dtype=torch.long).to(device)
        coef = 1
        if length < 4:
            final_probas, _ = mcts.search(competitive=False, c_puct=1, temperature=1)
        else:
            final_probas, _ = mcts.search(competitive=False, c_puct=1, temperature=1)
        game_data.append((state_tensor, final_probas, coef, False, possible_actions, mcts.root.state))
        length += 1

    # If reached terminal state, update reached_terminal to True
    if mcts.root.state.is_terminal_state():
        term_tokens = mcts.root.state.state
        state_tensor = torch.tensor(
            convert_state_tokens_to_ids(term_tokens),
            dtype=torch.long, device=device
        )
        no_actions = []  # No actions at the terminal state
        # Convert terminal state probabilities to the same dictionary format
        if isinstance(game_data[0][1], dict):
            term_probas = {a: 0.0 for a in game_data[0][1]}  # Or {}
        else:  # For safety, also support the Tensor branch
            term_probas = torch.zeros_like(game_data[0][1])

        game_data.append(
            (state_tensor, term_probas, 1,
             True, no_actions, mcts.root.state)
        )

        for idx in range(len(game_data)):
            game_data[idx] = (
                game_data[idx][0],      # state_tensor
                game_data[idx][1],      # final_probas
                game_data[idx][2],      # coef
                True,                   # reached_terminal set to True
                possible_actions,       # final_state
                mcts.root.state
            )

    return game_data

def run_self_play(args):
    (policy_net, value_net, LEN, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device, seed) = args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    return self_play_game(policy_net, value_net, LEN, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device)

def init_pool(*args):
    pass

def parallel_self_play(policy_net, value_net, LEN, num_games, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device, num_workers):
    seeds = [random.randint(0, 1e8) for _ in range(num_games)]
    args_list = [(policy_net, value_net, LEN, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device, s) for s in seeds]

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=num_workers, initializer=partial(init_pool, device)) as pool:
        results = pool.map(partial(run_self_play), args_list)
    return results


#####################################
#   process_game_data_and_calculate_rewards
#####################################
def process_game_data_and_calculate_rewards(all_game_data, replay_buffer, file, pool, logo=False):
    """
    Collect self-play game data, calculate final rewards and store them in the ReplayBuffer.
    """

    for game in all_game_data:
        final_ic = 0
        final_expression = None
        
        # Find the terminal state and update final_expression and final_ic
        for step in reversed(game):
            if step[-3]:  # reached_terminal=True
                final_expression = step[-1].expression
                try:
                    pool.try_new_expr(final_expression)
                    final_ic = pool.evaluate_ensemble()
                    if final_ic < 0:
                        final_ic = 0
                    logger.info(f"{final_expression},{final_ic}")     
                except OutOfDataRangeError:
                        final_ic = 0
                break


        # Record
        if final_expression is not None:
            # Update coef for each step when saving to buffer
            for step_idx, step in enumerate(game):
                state_tensor, raw_pi, _, _, possible_actions, _  = step
                reward = final_ic 
                target_policy = mcts_policy_to_target(raw_pi)
                replay_buffer.push(state_tensor, target_policy, reward, possible_actions)
                if logo:
                    with open(file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            state_tensor.tolist(),
                            target_policy,
                            reward,
                            final_expression,
                            possible_actions
                        ])


#####################################
#   process_game_data_and_calculate_single_rewards
#####################################
def process_game_data_and_calculate_single_rewards(all_game_data, replay_buffer, file, calculator, logo=False):
    """
    Collect self-play game data, calculate final rewards and store them in the ReplayBuffer.
    """
    
    # Collect the expressions and IC from the current batch of games
    expression_ic_list = []

    for game in all_game_data:
        final_ic = 0
        final_expression = None

        # Find the terminal state from the back
        for step in reversed(game):
            if step[-3]:  # reached_terminal=True
                final_expression = step[-1].expression
                try:
                    final_ic = calculator.calc_single_IC_ret(final_expression)
                    if math.isnan(final_ic):
                        final_ic = 0
                    logger.info(f"{final_expression},{final_ic}")
                except OutOfDataRangeError:
                    final_ic = 0
                    logger.info(f"{final_expression} OutOfDataRangeError!")
                break

        if final_expression:
            expression_ic_list.append((final_expression, final_ic))

        # Record
        if final_expression is not None:
            # Update coef for each step when saving to buffer
            for step_idx, step in enumerate(game):
                state_tensor, raw_pi, _, _, possible_actions, _  = step
                reward = abs(final_ic) 
                target_policy = mcts_policy_to_target(raw_pi)
                replay_buffer.push(state_tensor, target_policy, reward, possible_actions)
                if logo:
                    with open(file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            state_tensor.tolist(),
                            target_policy,
                            reward,
                            final_expression,
                            possible_actions
                        ])
    
    # Return the expressions and IC from the current batch of games
    return expression_ic_list
