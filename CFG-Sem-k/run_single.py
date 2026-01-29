# train.py
import math
import logging
from mimetypes import init
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
from train_preparation_pool import *
import numpy as np
import random

import torch.nn.init as init

import warnings
warnings.filterwarnings("ignore", message=".*does not contain valid edges.*")

def set_seed(seed):
    """设置随机数种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

def setup_logging():

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    return logger

multiprocessing.set_start_method('spawn', force=True)

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting main training script.")

    try:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # ------------------ Hyperparameters ------------------ #
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
        
        # Initializing datasets and calculators
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
        
        # Initialize environment
        env = AlphaEnv()
        env.reset()
        logger.debug("环境初始化完成。")

        # Create token map and determine vocabulary size
        token_map = create_token_map()
        vocab_size = len(token_map) + 1
        logger.debug(f"词汇表大小: {vocab_size}")

        embed_dim = 128
        h_size = 128
        dropout = 0.1
        

        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
        embed_dim=embed_dim,
        h_size=h_size,
        dropout = dropout
        ).to(device)
        logger.debug("Feature extractor initialized.")

        # Initialize policy and value networks
        policy_net = PolicyNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)
        value_net = ValueNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)

        logger.debug("Policy and Value networks initialized.")


        all_parameters = set(policy_net.parameters()).union(set(value_net.parameters()))
        optimizer = optim.Adam(list(all_parameters), lr=LEARNING_RATE)
        logger.debug("优化器初始化完成。")

        replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        logger.debug("经验回放缓冲区初始化完成。")

        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(script_dir, 'logs_single')
        os.makedirs(log_dir, exist_ok=True)

        training_metrics_file = os.path.join(log_dir, 'training_metrics.csv')
        rebuffer_play_file = os.path.join(log_dir, 'rebuffer_play.csv')

        if not os.path.exists(training_metrics_file):
            with open(training_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'average_policy_loss', 'average_value_loss'])
            logger.debug(f"create file: {training_metrics_file}")

        if not os.path.exists(rebuffer_play_file):
            with open(rebuffer_play_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['state', 'action_probs', 'value', 'current_expression', 'final_expression'])
            logger.debug(f"create file: {rebuffer_play_file}")


        pool_file =os.path.join(log_dir, 'pool_dicts.json')
        if not os.path.exists(pool_file):
            with open(pool_file, 'w', encoding='utf-8') as f:
                pass 
        # ============================================================

        pool = MseAlphaPool(
             capacity=1,
             calculator=calculator_train,
             ic_lower_bound=None,
             l1_alpha=5e-3,
             device=device
        )

        best_ic_valid = -float('inf')
        no_improve_counter = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"=== training iteration: {iteration} ===")
            
            for num in range(1, NUM + 1):

                logger.info(f"training iteration: {iteration}, batch {num}")
                all_game_data = parallel_self_play(
                    policy_net=policy_net, 
                    value_net=value_net, 
                    LEN=LEN, 
                    num_games=50,
                    MCTS_SIM = MCTS_SIM , 
                    MCTS_PARALLEL = MCTS_PARALLEL, 
                    BATCH_SIZE_EVAL = BATCH_SIZE_EVAL,
                    device=device,
                    num_workers=mp.cpu_count()//2
                    )
                
            # ============================================================
            process_game_data_and_calculate_single_rewards(all_game_data, replay_buffer, file = rebuffer_play_file, pool=pool,logo=False)
            torch.cuda.empty_cache()

            avg_policy_loss = train_policy_net(policy_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"Iteration {iteration} average policy loss: {avg_policy_loss}")

            avg_value_loss = train_value_net(value_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"Iteration {iteration} average policy loss: {avg_value_loss}")

            if avg_policy_loss is not None and avg_value_loss is not None:
                with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, iteration, avg_policy_loss, avg_value_loss])
                logger.debug(f"save metrics {training_metrics_file}")
            else:
                if avg_policy_loss is not None:
                    with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([iteration, iteration, avg_policy_loss, "N/A"])
                    logger.debug("metrics saved with N/A for value loss")
            
            if pool.size >0:

                    pool_dict = pool.to_json_dict()   # {'exprs': [...], 'weights': [...]}
                    ic_value_train, ric_value_train = pool.test_ensemble(calculator_train)
                    ic_value_valid, ric_value_valid = pool.test_ensemble(calculator_valid)
                    ic_value_test, ric_value_test = pool.test_ensemble(calculator_test)

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

                    if ic_value_valid > best_ic_valid:
                        best_ic_valid = ic_value_valid
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1

            if no_improve_counter >= 0.2* NUM_ITERATIONS:
                logger.info(f"Iteration {iteration} ic_value_valid not increasing {0.2* NUM_ITERATIONS}, early stopping triggered.")
                break
            replay_buffer.buffer.clear()
            torch.cuda.empty_cache()


        logger.info("Search completed.")

    except Exception as e:
        logger.exception("exception occurred during training:")


if __name__ == "__main__":
    setup_logging()
    main()
