# train.py

import logging  # 首先导入 logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import copy
import random
import json

from train_preparation import *  # 在配置日志之后再导入

def setup_logging():
    # 移除所有现有的处理器，防止重复或覆盖
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置为 DEBUG 以获取详细日志
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()  # 获取 root logger
    return logger

def main():
    # 获取 root logger
    logger = logging.getLogger()
    logger.debug("Starting main training script.")

    try:
        # 设备配置，修改这里，确保使用正确的 GPU
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 超参数
        NUM_ITERATIONS = 50  # 增加训练轮数，适用于中等至高复杂度任务
        N_SIMULATIONS = 10000  # 增加模拟次数，提升搜索深度和广度
        N_THREADS = 400  # 根据实际CPU核心数调整，优化并行效率
        BATCH_SIZE = 128  # 适度增加批量大小，加快训练速度
        LEARNING_RATE = 1e-3  # 保持初始学习率，使用学习率调度器
        REPLAY_BUFFER_SIZE = 20000  # 增加缓冲区大小，提升数据多样性
        SAVE_MODEL_EVERY = 1  # 每1轮保存一次模型
        SAVE_TREE_EVERY = 1  # 每1轮导出一次搜索树
        TEMPERATURE = 1.0  # 新增温度参数

        # 初始化环境
        env = AlphaEnv()
        env.reset()
        logger.debug("环境初始化完成。")

        # 创建词汇表
        token_map = create_token_map()
        vocab_size = len(token_map) + 1  # 加1是为了包含未定义的 token
        logger.debug(f"词汇表大小: {vocab_size}")

        embed_dim = 64
        nhead = 8
        nhid = 128
        nlayers = 2
        dropout = 0.1

        # 初始化特征提取器
        feature_extractor = FeatureExtractor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout
        ).to(device)
        logger.debug("特征提取器初始化完成。")

        # 初始化策略网络和价值网络
        policy_net = PolicyNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)
        value_net = ValueNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)
        logger.debug("策略网络和价值网络初始化完成。")

        # 将所有参数放入一个集合中，自动去重
        all_parameters = set(policy_net.parameters()).union(set(value_net.parameters()))

        # 将参数集合转换回列表，并创建优化器
        optimizer = optim.Adam(list(all_parameters), lr=LEARNING_RATE)

        logger.debug("优化器初始化完成。")

        # 初始化经验回放缓冲区
        replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        logger.debug("经验回放缓冲区初始化完成。")

        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        logger.debug("日志目录已创建。")

        # 定义 CSV 文件路径
        best_expressions_file = 'logs/best_expressions.csv'
        all_expressions_file = 'logs/all_expressions.csv'
        training_metrics_file = 'logs/training_metrics.csv'
        replay_buffer_file = 'logs/replay_buffer.csv'  # 新增缓冲区 CSV 文件路径

        # 初始化 CSV 文件，写入表头
        if not os.path.exists(best_expressions_file):
            with open(best_expressions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'expression', 'IC'])
            logger.debug(f"创建文件并写入表头: {best_expressions_file}")

        if not os.path.exists(all_expressions_file):
            with open(all_expressions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'path', 'expression', 'IC'])
            logger.debug(f"创建文件并写入表头: {all_expressions_file}")

        if not os.path.exists(training_metrics_file):
            with open(training_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'average_policy_loss', 'average_value_loss'])
            logger.debug(f"创建文件并写入表头: {training_metrics_file}")

        # 初始化 Replay Buffer CSV 文件，写入表头
        if not os.path.exists(replay_buffer_file):
            with open(replay_buffer_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['state', 'action_probs', 'value', 'expression', 'value_net_output', 'policy_net_output'])
            logger.debug(f"创建文件并写入表头: {replay_buffer_file}")

        # 训练迭代
        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"=== 训练轮数: {iteration} ===")

            # 每次迭代重新初始化 MCTS
            mcts = MCTS(
                policy_net=policy_net,
                value_net=value_net,
                c_puct=0.1,  # 根据建议调整
                n_simulations=N_SIMULATIONS,
                n_threads=N_THREADS,
                device=device
            )
            logger.debug(f"MCTS 初始化完成: 轮数 {iteration}")

            # 自我对弈生成数据，并构建搜索树
            best_expression, best_ic, complete_expressions = self_play(mcts, env, replay_buffer, device, TEMPERATURE)

            if best_expression is None:
                logger.warning(f"第 {iteration} 轮迭代中没有生成有效的表达式。跳过保存步骤。")
                continue

            # 保存最佳表达式到 best_expressions.csv
            with open(best_expressions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, best_expression, best_ic])
            logger.debug(f"已保存最佳表达式到 {best_expressions_file}: {best_expression}, IC: {best_ic}")

            # 保存所有完整表达式到 all_expressions.csv
            with open(all_expressions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for expr in complete_expressions:
                    path_str = ' -> '.join([f"{cat}:{act}" for cat, act in expr['path']])
                    writer.writerow([iteration, path_str, expr['expression'], expr['IC']])
            logger.debug(f"已保存所有完整表达式到 {all_expressions_file}")

            # 导出搜索树
            if iteration % SAVE_TREE_EVERY == 0:
                tree_dot = f'mcts_tree_iter_{iteration}.dot'
                tree_pdf = f'mcts_tree_iter_{iteration}.pdf'
                mcts.export_search_tree(filename_dot=tree_dot, filename_pdf=tree_pdf)
                logger.info(f"搜索树已保存为 {tree_dot} 和 {tree_pdf}")

            # 训练策略网络
            avg_policy_loss = train_policy_net(policy_net, optimizer, replay_buffer, BATCH_SIZE, device)
            logger.info(f"轮数 {iteration} 的平均策略损失: {avg_policy_loss}")

            # 训练价值网络
            avg_value_loss = train_value_net(value_net, optimizer, replay_buffer, BATCH_SIZE, device)
            logger.info(f"轮数 {iteration} 的平均价值损失: {avg_value_loss}")

            # 如果有训练损失，保存到 training_metrics.csv
            if avg_policy_loss is not None and avg_value_loss is not None:
                with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, avg_policy_loss, avg_value_loss])
                logger.debug(f"已保存训练指标到 {training_metrics_file}")

            # 保存 Replay Buffer 数据到 CSV
            replay_buffer.save_to_csv(replay_buffer_file, policy_net, value_net, device)
            logger.info(f"Replay Buffer 已保存至 {replay_buffer_file}")

            # 保存模型
            if iteration % SAVE_MODEL_EVERY == 0:
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'policy_net_state_dict': policy_net.state_dict(),
                    'value_net_state_dict': value_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'models/mcts_model_iter_{iteration}.pth')
                logger.info(f"模型已保存至 models/mcts_model_iter_{iteration}.pth")

            # 打印部分 Replay Buffer 样本进行验证
            if len(replay_buffer) > 0:
                logger.info("Replay Buffer Samples:")
                for idx in range(min(5, len(replay_buffer))):
                    state, action_probs_list, value, expression = replay_buffer.buffer[idx]
                    logger.info(f"Sample {idx}: State={state}, Action Probs List={action_probs_list}, Value={value}, Expression={expression}")
            else:
                logger.info("Replay Buffer is empty.")

        logger.info("训练完成。")

        # 训练结束后的操作...
        # ----------------------------------------------
        # 训练结束后，执行一次 MCTS 搜索并输出搜索树
        # ----------------------------------------------

        # 创建并重置一个新的环境实例
        final_env = AlphaEnv()
        final_env.reset()
        logger.debug("Final 环境重置完成。")

        # 重新初始化 MCTS 用于最终搜索
        final_mcts = MCTS(
            policy_net=policy_net,
            value_net=value_net,
            c_puct=0,
            n_simulations=N_SIMULATIONS,
            n_threads=N_THREADS,
            device=device
        )
        logger.debug("Final MCTS 初始化完成")

        # 使用训练期间的 MCTS 对象，不重新初始化
        best_action = final_mcts.search(final_env)
        logger.info(f"最佳动作: {best_action}")

        # 获取最佳路径
        best_path = final_mcts.get_best_path()
        logger.info("最佳路径的动作序列:")
        for action in best_path:
            logger.info(f" - {action}")

        # 根据最佳路径构建因子表达式
        current_state = copy.deepcopy(final_env)
        for action_category, action in best_path:
            current_state.apply_action(action_category, action)

        final_expression = current_state.state_description()
        logger.info(f"最佳路径对应的因子表达式: {final_expression}")

        # 导出搜索树
        final_mcts.export_search_tree(filename_dot="mcts_tree_final.dot", filename_pdf="mcts_tree_final.pdf")
        logger.info("最终搜索树已导出")

    except Exception as e:
        logger.exception("训练过程中发生未处理的异常")

if __name__ == "__main__":
    setup_logging()  # 首先配置日志
    main()



