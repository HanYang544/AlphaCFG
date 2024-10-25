# train_preparation.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import copy
import os
import random
import numpy as np
import pandas as pd
import json

from mcts import MCTS, MCTSNode
from policy_ import PolicyNetwork, evaluate_policy_network, ACTION_MAP  # 确保导入 ACTION_MAP
from value_ import ValueNetwork, evaluate_expression_value
from alpha_env import AlphaEnv
from token_v import create_token_map, parse_expression_to_tokens, get_first_token
from evaluator_ import *

from feature_ import FeatureExtractor

import logging
from collections import deque
from torch.nn.utils.rnn import pad_sequence

import csv

# 类别名称映射（确保与 policy_.py 中的一致）
CATEGORY_MAP = {
    0: 'Vari',
    1: 'constant',
    2: 'unaryop',
    3: 'binaryop',
    4: 'rollingop',
    5: 'pairedrollingop',
    # 6: 'num'  移除 num 类
}

# 配置日志
logger = logging.getLogger(__name__)

# 数据集类
class TrainingDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action_probs_list, value, expression = self.buffer[idx]
        return state, action_probs_list, value, expression

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_probs_list, value, expression):
        self.buffer.append((state, action_probs_list, value, expression))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    # 修改方法：将缓冲区数据保存到 CSV 文件，包含策略网络和值网络的输出
    def save_to_csv(self, filename, policy_net, value_net, device):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'action_probs', 'value', 'expression', 'value_net_output', 'policy_net_output'])
            for state, action_probs_list, value, expression in self.buffer:
                # 将 state 张量转换为列表
                state_list = state.cpu().numpy().tolist()
                # 将 action_probs_list 转换为 JSON 字符串
                action_probs_str = json.dumps(action_probs_list)
                # 将 value 张量转换为浮点数
                value_float = value.item() if isinstance(value, torch.Tensor) else float(value)

                # 计算 value_net 输出
                state_input = state.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
                value_net.eval()  # 确保模型处于评估模式
                with torch.no_grad():
                    value_net_output = value_net(state_input).item()

                # 计算 policy_net 输出
                # 我们需要 first_token
                token_map = create_token_map()
                tokens = parse_expression_to_tokens(expression, token_map)
                first_token = get_first_token(tokens, token_map)
                if first_token is None:
                    policy_net_output_str = None
                else:
                    policy_net.eval()  # 确保模型处于评估模式
                    with torch.no_grad():
                        possible_actions = [action['category'] for action in action_probs_list]
                        action_outputs = policy_net(state_input, first_token=first_token, possible_actions=possible_actions)
                    # action_outputs 是一个包含动作字典的列表
                    # 将其转换为可序列化的形式
                    policy_net_output_serializable = []
                    if action_outputs is not None:
                        for output in action_outputs:
                            output_serializable = {}
                            for key, val in output.items():
                                if isinstance(val, torch.Tensor):
                                    output_serializable[key] = val.cpu().numpy().tolist()
                                else:
                                    output_serializable[key] = val
                            policy_net_output_serializable.append(output_serializable)
                        policy_net_output_str = json.dumps(policy_net_output_serializable)
                    else:
                        policy_net_output_str = None

                # 将数据写入 CSV 文件
                writer.writerow([state_list, action_probs_str, value_float, expression, value_net_output, policy_net_output_str])

# 自定义的 collate_fn 函数，进行 padding
def custom_collate_fn(batch):
    states, action_probs_list, values, expressions = zip(*batch)

    # 填充 state_tensors，使所有序列具有相同长度
    states_padded = pad_sequence(states, batch_first=True, padding_value=0)  # (batch_size, max_len)

    # 将 values 转换为张量
    values = torch.tensor(values, dtype=torch.float32)

    # expressions 作为列表返回
    return states_padded, action_probs_list, values, expressions

def mcts_policy_to_target(action_probs_dict, token_map):
    """
    将 MCTS 生成的策略字典转换为 PolicyNetwork 所期望的格式。
    确保所有类别的所有动作都有相应的概率分布，即使概率为 0。
    
    参数：
        action_probs_dict (dict): MCTS 生成的动作及其概率，例如 {('unaryop', 'Abs'): 0.3, ...}
        token_map (dict): 词汇表映射，用于获取类别名称。
        
    返回：
        list: 包含动作字典的列表，每个字典包含 'category', 'action', 'prob'。
    """
    target_policy = []

    # 遍历 ACTION_MAP，确保每个类别的每个动作都有概率
    for category, actions in ACTION_MAP.items():
        for action in actions:
            # 如果 MCTS 结果中有该动作的概率，则使用其概率，否则设为0
            prob = action_probs_dict.get((category, action), 0.0)
            target_policy.append({
                'category': category,
                'action': action,
                'prob': prob
            })

    return target_policy


def compute_policy_loss(action_outputs, target_policy, classification_loss_fn, device):
    """
    计算策略损失。

    参数：
        action_outputs (torch.Tensor): 策略网络的输出张量，每个元素代表一个动作类别的概率。
        target_policy (list): 目标策略，包含每个动作的类别、动作和概率。
        classification_loss_fn (nn.Module): 分类损失函数。
        device (torch.device): 设备。

    返回：
        torch.Tensor: 计算得到的策略损失。
    """
    # 获取类别总数和动作总数
    num_categories = action_outputs.shape[0]  # 使用 action_outputs 的大小来初始化目标分布的大小

    # 创建与 action_outputs 相同结构的 target_distribution
    target_distribution = torch.zeros(num_categories, device=device)

    # 创建类别和动作的索引映射
    category_idx = 0

    # 填充 target_distribution，确保每个类别的每个动作都有概率
    for category, actions in ACTION_MAP.items():
        for action in actions:
            if category_idx >= num_categories:
                # 如果类别索引超出 action_outputs 范围，停止填充
                logging.warning(f"Skipping action ({category}, {action}) due to index {category_idx} out of bounds for action_outputs size {num_categories}")
                break
            # 获取目标策略中的概率
            prob = next((item['prob'] for item in target_policy if item['category'] == category and item['action'] == action), 0.0)
            target_distribution[category_idx] = prob
            category_idx += 1

    # 归一化目标分布
    if target_distribution.sum() > 0:
        target_distribution /= target_distribution.sum()

    # 确保输出分布的尺寸与目标分布匹配
    if action_outputs.shape[0] != num_categories:
        raise ValueError(f"Action outputs size {action_outputs.shape[0]} does not match the number of categories {num_categories}")

    # 归一化输出分布
    output_distribution = torch.softmax(action_outputs, dim=0)

    logging.debug(f"Target Distribution: {target_distribution}")
    logging.debug(f"Output Distribution: {output_distribution}")

    # 计算 KL散度损失
    loss = classification_loss_fn(torch.log(output_distribution + 1e-10), target_distribution)

    logging.debug(f"计算的策略损失: {loss.item()}")
    return loss



def train_policy_net(policy_net, optimizer, replay_buffer, batch_size, device):
    # 检查重放缓冲区中是否有足够的数据进行训练
    if len(replay_buffer) < batch_size:
        logging.debug("Replay buffer size less than batch size. Skipping training.")
        return 0.0  # 如果数据不足，则跳过训练

    policy_net.train()  # 将网络设置为训练模式
    classification_loss_fn = nn.KLDivLoss(reduction='batchmean')  # 初始化分类损失函数
    total_policy_loss = 0.0  # 初始化总策略损失
    valid_samples = 0  # 有效样本数

    dataset = TrainingDataset(replay_buffer.buffer)  # 创建数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)  # 创建数据加载器

    for batch_idx, (states, action_probs_list, _, expressions) in enumerate(dataloader):
        states = states.to(device)  # 将状态移至指定的设备

        optimizer.zero_grad()  # 清空之前的梯度信息
        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 初始化单批次的策略损失为张量

        for i in range(states.size(0)):
            target_policy = action_probs_list[i]  # 获取目标策略
            if not target_policy:
                logging.debug(f"样本 {i} 被跳过：action_probs 为空。")
                continue  # 如果目标策略为空，则跳过

            token_map = create_token_map()
            tokens = parse_expression_to_tokens(expressions[i], token_map)
            first_token = get_first_token(tokens, token_map)
            logging.debug(f"表达式: '{tokens}'")
            if first_token is None:
                state_as_list = states[i].tolist()  # 将状态张量转换为列表，以便于记录
                logging.debug(f"样本 {i} 被跳过：无法获取第一个令牌。状态: {state_as_list}, 目标策略: {target_policy}, 表达式: '{expressions[i]}'")
                continue  # 如果没有令牌，则跳过

            # 调用策略网络进行前向传播
            action_outputs = policy_net(
                states[i].unsqueeze(0),
                first_token=first_token,
                possible_actions=list(set([action['category'] for action in target_policy]))
            )
            
            # 确保 action_outputs 是一个 Tensor 并具有梯度
            action_outputs = torch.tensor([item['prob'] for item in action_outputs], dtype=torch.float32, device=device, requires_grad=True)

            if action_outputs.requires_grad:
                logging.debug("Action outputs contain gradients.")
            else:
                logging.error("Action outputs do not contain gradients. Check the network architecture.")

            # 检查 policy_net 输出
            if action_outputs is None:
                logging.debug(f"样本 {i} 的 action_outputs 为 None。跳过损失计算。")
                continue

            logging.debug(f"样本 {i} 的 action_outputs: {action_outputs}")

            # 计算策略损失
            loss = compute_policy_loss(action_outputs, target_policy, classification_loss_fn, device)
            logging.debug(f"样本 {i} 的策略损失: {loss.item()}")
            # 累加损失，避免原地操作
            policy_loss = policy_loss + loss

        # 如果策略损失需要梯度，则进行反向传播和优化
        if policy_loss.requires_grad:
            policy_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            valid_samples += 1
            logging.debug(f"批次 {batch_idx} 的策略损失累加: {policy_loss.item()}")
        else:
            logging.error(f"批次 {batch_idx} 的策略损失不需要梯度。")

    avg_policy_loss = total_policy_loss / valid_samples if valid_samples > 0 else 0.0  # 计算平均策略损失
    logging.info(f"训练后的平均策略损失: {avg_policy_loss}")
    return avg_policy_loss

def train_value_net(value_net, optimizer, replay_buffer, batch_size, device):
    # 检查重放缓冲区中是否有足够的数据进行训练
    if len(replay_buffer) < batch_size:
        logging.debug("Replay buffer size less than batch size. Skipping training.")
        return 0.0

    value_net.train()  # 将网络设置为训练模式
    regression_loss_fn = nn.MSELoss()  # 初始化回归损失函数
    total_value_loss = 0.0  # 初始化总价值损失
    total_samples = 0  # 总样本数

    dataset = TrainingDataset(replay_buffer.buffer)  # 创建数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)  # 创建数据加载器

    for batch_idx, (states, _, values, _) in enumerate(dataloader):
        states = states.to(device)  # 将状态移至指定的设备
        values = values.to(device).float()  # 将值转换为浮点数并移至设备

        optimizer.zero_grad()  # 清空之前的梯度信息
        value_output = value_net(states).squeeze(-1)  # 进行前向传播并移除最后一个维度
        value_loss = regression_loss_fn(value_output, values)  # 计算价值损失

        value_loss.backward()  # 反向传播
        optimizer.step()  # 执行优化步骤

        total_value_loss += value_loss.item()  # 累加价值损失
        total_samples += states.size(0)  # 累加样本数

    avg_value_loss = total_value_loss / total_samples if total_samples > 0 else 0.0  # 计算平均价值损失
    return avg_value_loss


# 辅助函数：递归收集所有完整路径
def collect_complete_paths(node, current_path, all_paths):
        """
        递归遍历 MCTS 树，收集所有从当前节点到终止节点的完整路径。
        """
        if node.is_terminal():
            all_paths.append((copy.deepcopy(current_path), node))
            return
        if not node.is_expanded:
            return
        for action, child in node.children.items():
            current_path.append(action)
            collect_complete_paths(child, current_path, all_paths)
            current_path.pop()

def self_play(mcts, env, replay_buffer, device, temperature):
        """
        执行一次自我对弈，生成训练数据并存储到经验回放缓冲区。
        """
        # 深拷贝当前环境状态，确保不修改原始环境
        state = copy.deepcopy(env)

        # 使用 MCTS 搜索最佳动作
        best_action = mcts.search(state)

        # 获取根节点
        root_node = mcts.root

        # 获取根节点的所有子节点的访问计数
        action_counts = {action: child.visit_count for action, child in root_node.children.items()}

        if not action_counts:
            logging.warning("MCTS 搜索后，根节点没有子节点。请检查 MCTS 实现是否正确。")
            return None, None, []

        # 计算 π(a) ∝ N(s, a)^(1/τ)
        action_probs = {action: (count ** (1.0 / temperature)) for action, count in action_counts.items()}

        # 归一化概率
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {action: prob / total for action, prob in action_probs.items()}
        else:
            # 如果总和为0，使用均匀分布
            num_actions = len(action_counts)
            action_probs = {action: 1.0 / num_actions for action in action_counts}

        # 获取当前状态的描述并转换为张量
        current_expression = state.state_description()
        tokens = parse_expression_to_tokens(current_expression, create_token_map())
        state_tensor = torch.tensor(tokens, dtype=torch.long).to(device)

        # 根节点的价值（使用 root_node.Q）
        # 这里我们将根节点的 value_tensor 设置为 0，因为它不是终止状态
        value_tensor = torch.tensor([0.0], dtype=torch.float32).to(device)

        # 将根节点的状态和策略目标概率推送到经验回放缓冲区
        # 量化 'constant' 类别的动作值
        if action_probs:
            processed_action_probs = {}
            for a, prob in action_probs.items():
                cat, act = a
                if cat == 'constant':
                    # 四舍五入到四位小数
                    act = round(act, 4)  # 修改精度为四位小数
                    processed_action_probs[(cat, act)] = prob
                else:
                    processed_action_probs[a] = prob
            logging.debug(f"Pushing to replay buffer: State={state_tensor}, Action Probs={processed_action_probs}, Value={value_tensor}, Expression={current_expression}")
            target_policy = mcts_policy_to_target(processed_action_probs, create_token_map())
            replay_buffer.push(
                state_tensor,
                target_policy,  # 使用转换后的策略
                value_tensor,
                current_expression  # 传递表达式描述
            )
            logging.debug(f"已添加有效样本到Replay Buffer: Expression={current_expression}")
        else:
            logging.warning("Root node action probabilities are empty. Skipping this sample.")

        # 收集所有完整路径及其终止节点
        all_paths = []
        collect_complete_paths(root_node, [], all_paths)

        # 初始化用于保存所有完整表达式的信息
        complete_expressions = []

        for path, terminal_node in all_paths:
            # 重置环境
            current_state = copy.deepcopy(env)
            node = root_node  # 从根节点开始

            # 初始化用于存储路径上各状态的数据
            path_data = []

            for action in path:
                action_category, action_value = action
                current_state.apply_action(action_category, action_value)

                # 获取当前节点的子节点访问次数
                if node.is_expanded:
                    action_counts = {a: child.visit_count for a, child in node.children.items()}
                    # 计算策略目标概率
                    action_probs = {a: (count ** (1.0 / temperature)) for a, count in action_counts.items()}
                    total = sum(action_probs.values())
                    if total > 0:
                        action_probs = {a: prob / total for a, prob in action_probs.items()}
                        logging.debug(f"Normalized action_probs: {action_probs}")
                    else:
                        # 如果总和为0，使用均匀分布
                        num_actions = len(action_counts)
                        action_probs = {a: 1.0 / num_actions for a in action_counts}
                        logging.debug(f"Uniform action_probs (total=0): {action_probs}")
                else:
                    action_probs = {}

                # 获取当前状态的描述并转换为张量
                current_expression = current_state.state_description()
                tokens = parse_expression_to_tokens(current_expression, create_token_map())
                state_tensor = torch.tensor(tokens, dtype=torch.long).to(device)

                # 对非终止状态，将 value_tensor 设置为 0.0
                value_tensor = torch.tensor([0.0], dtype=torch.float32).to(device)

                # 处理动作概率
                if action_probs:
                    processed_action_probs = {}
                    for a, prob in action_probs.items():
                        cat, act = a
                        if cat == 'constant':
                            # 四舍五入到四位小数
                            act = round(act, 4)
                            processed_action_probs[(cat, act)] = prob
                        else:
                            processed_action_probs[a] = prob
                    logging.debug(f"Collected data: State={state_tensor}, Action Probs={processed_action_probs}, Value={value_tensor}, Expression={current_expression}")

                    # 使用新函数转换 MCTS 策略为目标格式
                    target_policy = mcts_policy_to_target(processed_action_probs, create_token_map())

                    # 将数据添加到 path_data 列表
                    path_data.append((state_tensor, target_policy, value_tensor, current_expression))
                else:
                    logging.warning("Action probabilities are empty at this node. Skipping this sample.")

                # 移动到下一个节点
                if action in node.children:
                    node = node.children[action]
                else:
                    logging.debug(f"Action {action} not found in node children. Exiting loop.")
                    break  # 如果子节点不存在，退出循环

            # 获取最终的表达式
            final_expression = current_state.state_description()

            # 计算当前状态的价值（IC 值）
            try:
                expression_obj = parse_expression_from_string(final_expression)
                ic = evaluate_expression_ic(expression_obj)
                #此行疑似有问题，暂且去掉 ic = abs(ic)
                logging.info(f"生成的因子表达式已评估，IC: {ic:.4f}")
            except Exception as e:
                ic = 0.0
                logging.warning(f"评估表达式时出错: {e}. IC 值设为 0。")

            # 保存完整表达式的信息
            actions_sequence = path.copy()
            complete_expressions.append({
                'path': actions_sequence,
                'expression': final_expression,
                'IC': ic
            })

            # 更新终止状态的 value_tensor 为 IC 值
            if path_data:
                # 更新终止状态的 value_tensor
                state_tensor, target_policy, _, expression = path_data[-1]
                value_tensor = torch.tensor([ic], dtype=torch.float32).to(device)
                path_data[-1] = (state_tensor, target_policy, value_tensor, expression)

                # 将所有数据推送到经验回放缓冲区
                for state_tensor, target_policy, value_tensor, expression in path_data:
                    replay_buffer.push(state_tensor, target_policy, value_tensor, expression)
                    logging.debug(f"已添加样本到Replay Buffer: Expression={expression}, IC={ic}")
            else:
                logging.warning("Path data is empty. No data to push to replay buffer.")

        # 选择 IC 值最高的表达式作为最佳表达式
        if complete_expressions:
            best_expr_obj = max(complete_expressions, key=lambda x: x['IC'])
            best_expression = best_expr_obj['expression']
            best_ic = best_expr_obj['IC']
        else:
            best_expression = None
            best_ic = 0.0

        if best_expression:
            logging.info(f"最佳表达式: {best_expression}, IC: {best_ic:.4f}")
        else:
            logging.info(f"没有生成有效的表达式。")

        logging.info(f"共有 {len(complete_expressions)} 个完整表达式被生成。")
        return best_expression, best_ic, complete_expressions

