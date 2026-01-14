# train.py
import math
import logging
from mimetypes import init  # 首先导入 logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import copy
import random
import json
import multiprocessing
import numpy
from models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from networks.feature_ import FeatureExtractor
from train_preparation_pool import *  # 在配置日志之后再导入
import numpy as np
import random

import torch.nn.init as init

import warnings
warnings.filterwarnings("ignore", message=".*does not contain valid edges.*")

def set_seed(seed):
    """设置随机数种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次卷积算法的选择是确定的
    torch.backends.cudnn.benchmark = False     # 如果网络输入数据维度或类型上变化不大，设置为True可能更合适

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

# 设置多进程启动方式为 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def main():
    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting main training script.")

    try:
        # 设备配置，修改这里，确保使用正确的 GPU
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # ------------------ 超参数设置 ------------------ #
        NUM_ITERATIONS = 100
        NUM = 2
        NUM_BATCHES = 8
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        REPLAY_BUFFER_SIZE = 20000
        LEN = 20
        MCTS_SIM = 64
        MCTS_PARALLEL = 8
        BATCH_SIZE_EVAL = 2
        
        # 初始化数据和目标
        train_data = StockData(
            instrument="csi300",
            start_time="2010-01-01",
            end_time="2017-12-31",
            device=device
        )
        valid_data = StockData(
            instrument="csi300",
            start_time="2018-01-01",
            end_time="2019-12-31",
            device=device
        )
        test_data = StockData(
            instrument="csi300",
            start_time="2021-01-01",
            end_time="2024-12-31",
            device=device
        )
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -20) / close - 1
        calculator_train = QLibStockDataCalculator(data=train_data, target=target)
        calculator_valid = QLibStockDataCalculator(data=valid_data, target=target)
        calculator_test = QLibStockDataCalculator(data=test_data, target=target)

        set_seed(seed=1)
        
        # 初始化环境
        env = AlphaEnv()
        env.reset()
        logger.debug("环境初始化完成。")

        # 创建词汇表
        token_map = create_token_map()
        vocab_size = len(token_map) + 1
        logger.debug(f"词汇表大小: {vocab_size}")

        embed_dim = 128
        h_size = 128
        dropout = 0.1
        

        # 初始化特征提取器
        feature_extractor = FeatureExtractor(
        embed_dim=embed_dim,
        h_size=h_size,
        dropout = dropout
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
        # 转为列表并创建优化器
        optimizer = optim.Adam(list(all_parameters), lr=LEARNING_RATE)
        logger.debug("优化器初始化完成。")

        # 初始化经验回放缓冲区
        replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        logger.debug("经验回放缓冲区初始化完成。")

        # 创建日志目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(script_dir, 'logs_single')
        os.makedirs(log_dir, exist_ok=True)

        # 定义 CSV 文件路径
        training_metrics_file = os.path.join(log_dir, 'training_metrics.csv')
        rebuffer_play_file = os.path.join(log_dir, 'rebuffer_play.csv')

        if not os.path.exists(training_metrics_file):
            with open(training_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'average_policy_loss', 'average_value_loss'])
            logger.debug(f"创建文件并写入表头: {training_metrics_file}")

        if not os.path.exists(rebuffer_play_file):
            with open(rebuffer_play_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['state', 'action_probs', 'value', 'current_expression', 'final_expression'])
            logger.debug(f"创建文件并写入表头: {rebuffer_play_file}")


        # =============== 新增：保存 pool_dict 的 JSON 文件路径 ===============
        pool_file =os.path.join(log_dir, 'pool_dicts.json')
        # 如果想每次都新建文件，可用 'w' 模式，这里示例直接追加
        if not os.path.exists(pool_file):
            with open(pool_file, 'w', encoding='utf-8') as f:
                pass  # 写个空文件，防止报错
        # ============================================================

         # 每次创建一个新的池
        pool = MseAlphaPool(
             capacity=1,
             calculator=calculator_train,
             ic_lower_bound=None,
             l1_alpha=5e-3,
             device=device
        )
        
        # 在进入迭代循环前，初始化早停所需变量
        best_ic_valid = -float('inf')
        no_improve_counter = 0
        # 开始训练（外层循环）
        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"=== 训练轮数: {iteration} ===")
            
            # 内层训练迭代
            for num in range(1, NUM + 1):
                 # 并行生成数据
                logger.info(f"训练轮数: {iteration},批次{num}")
                all_game_data = parallel_self_play(
                    policy_net=policy_net, 
                    value_net=value_net, 
                    LEN=LEN, 
                    num_games=50,
                    MCTS_SIM = MCTS_SIM , 
                    MCTS_PARALLEL = MCTS_PARALLEL, 
                    BATCH_SIZE_EVAL = BATCH_SIZE_EVAL,
                    device=device,
                    num_workers=mp.cpu_count()//2  # 使用半数CPU核心 
                    )
                
            # ============================================================
            process_game_data_and_calculate_single_rewards(all_game_data, replay_buffer, file = rebuffer_play_file, pool=pool,logo=False)
            torch.cuda.empty_cache()
            
            # 训练策略网络
            avg_policy_loss = train_policy_net(policy_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"轮数 {iteration} 的平均策略损失: {avg_policy_loss}")

            # 训练价值网络
            avg_value_loss = train_value_net(value_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"轮数 {iteration} 的平均价值损失: {avg_value_loss}")

            # 记录训练指标到 CSV 文件
            if avg_policy_loss is not None and avg_value_loss is not None:
                with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, iteration, avg_policy_loss, avg_value_loss])
                logger.debug(f"已保存训练指标到 {training_metrics_file}")
            else:
                if avg_policy_loss is not None:
                    with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([iteration, iteration, avg_policy_loss, "N/A"])
                    logger.debug("已保存训练指标（价值网络 N/A）到文件")
            
            if pool.size >0:
                    # =============== 这里获取 pool 的字典并保存到文件 ===============
                    pool_dict = pool.to_json_dict()   # {'exprs': [...], 'weights': [...]}
                    ic_value_train, ric_value_train = pool.test_ensemble(calculator_train)
                    ic_value_valid, ric_value_valid = pool.test_ensemble(calculator_valid)
                    ic_value_test, ric_value_test = pool.test_ensemble(calculator_test)
                    # 将结果与 iteration 一起写入 logs/pool_dicts.json 中
                    with open(pool_file, 'a', encoding='utf-8') as f:
                        data_to_save = {
                            "iteration": iteration,
                            "num":num,
                            "pool_dict": pool_dict,
                            "ic_value_train": ic_value_train,
                            "ric_value_train": ric_value_train,
                            "ic_value_valid": ic_value_valid,
                            "ric_value_valid": ric_value_valid,
                            "ic_value_test": ic_value_test,
                            "ric_value_test": ric_value_test,
                        }
                        f.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")
                    
                    # =========== 早停逻辑 ===========
                    if ic_value_valid > best_ic_valid:
                        best_ic_valid = ic_value_valid
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
        
            # 清空 replay_buffer，以便下次自对弈产生新的数据
            if no_improve_counter >= 0.2* NUM_ITERATIONS:
                logger.info(f"在迭代 {iteration} 中 ic_value_valid 已连续{0.2* NUM_ITERATIONS}次未提升，提前停止训练。")
                break
            replay_buffer.buffer.clear()
            torch.cuda.empty_cache()


        logger.info("搜索完成。")

    except Exception as e:
        logger.exception("训练过程中发生未处理的异常")


if __name__ == "__main__":
    setup_logging()
    main()
