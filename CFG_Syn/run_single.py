# train.py
import math
import logging
from mimetypes import init  # First import logging
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
from train_preparation_pool import *  # Import after logging configuration
import numpy as np
import random

import torch.nn.init as init

import warnings
warnings.filterwarnings("ignore", message=".*does not contain valid edges.*")

def set_seed(seed):
    """Set random seed to ensure reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic selection of convolution algorithm each time
    torch.backends.cudnn.benchmark = False     # Set to True if network input data dimensions or types don't vary much

def setup_logging():
    # Remove all existing handlers to prevent duplication or overwriting
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to get detailed logs
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()  # Get root logger
    return logger

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def main():
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting main training script.")

    try:
        # Device configuration, modify here to ensure using the correct GPU
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # ------------------ Hyperparameter settings ------------------ #
        NUM_ITERATIONS = 10000
        NUM = 2
        NUM_BATCHES = 8
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        REPLAY_BUFFER_SIZE = 20000
        LEN = 10
        MCTS_SIM = 64
        MCTS_PARALLEL = 8
        BATCH_SIZE_EVAL = 2
        
        # Initialize data and targets
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
        logger.debug("Environment initialization completed.")

        # Create vocabulary
        token_map = create_token_map()
        vocab_size = len(token_map) + 1
        logger.debug(f"Vocabulary size: {vocab_size}")

        embed_dim = 128
        h_size = 128
        dropout = 0.1
        

        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
        embed_dim=embed_dim,
        h_size=h_size,
        dropout = dropout
        ).to(device)
        logger.debug("Feature extractor initialization completed.")

        # Initialize policy network and value network
        policy_net = PolicyNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)
        value_net = ValueNetwork(
            feature_extractor=feature_extractor,
            embed_dim=embed_dim
        ).to(device)

        logger.debug("Policy network and value network initialization completed.")


        # Put all parameters into one set to automatically remove duplicates
        all_parameters = set(policy_net.parameters()).union(set(value_net.parameters()))
        # Convert to list and create optimizer
        optimizer = optim.Adam(list(all_parameters), lr=LEARNING_RATE)
        logger.debug("Optimizer initialization completed.")

        # Initialize experience replay buffer
        replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        logger.debug("Experience replay buffer initialization completed.")

        # Create log directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(script_dir, 'logs_single')
        os.makedirs(log_dir, exist_ok=True)

        # Define CSV file paths
        training_metrics_file = os.path.join(log_dir, 'training_metrics.csv')
        rebuffer_play_file = os.path.join(log_dir, 'rebuffer_play.csv')

        if not os.path.exists(training_metrics_file):
            with open(training_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'average_policy_loss', 'average_value_loss'])
            logger.debug(f"Created file and wrote header: {training_metrics_file}")

        if not os.path.exists(rebuffer_play_file):
            with open(rebuffer_play_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['state', 'action_probs', 'value', 'current_expression', 'final_expression'])
            logger.debug(f"Created file and wrote header: {rebuffer_play_file}")


        # =============== New: JSON file path for saving pool_dict ===============
        pool_file =os.path.join(log_dir, 'pool_dicts.json')
        # If you want to create a new file each time, use 'w' mode, here we just append as an example
        if not os.path.exists(pool_file):
            with open(pool_file, 'w', encoding='utf-8') as f:
                pass  # Write an empty file to prevent errors
        # ============================================================

         # Create a new pool each time
        pool = MseAlphaPool(
             capacity=1,
             calculator=calculator_train,
             ic_lower_bound=None,
             l1_alpha=5e-3,
             device=device
        )
        
        # Before entering the iteration loop, initialize variables needed for early stopping
        best_ic_valid = -float('inf')
        no_improve_counter = 0
        # Start training (outer loop)
        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"=== Training round: {iteration} ===")
            
            # Inner training iteration
            for num in range(1, NUM + 1):
                 # Parallel data generation
                logger.info(f"Training round: {iteration}, batch {num}")
                all_game_data = parallel_self_play(
                    policy_net=policy_net, 
                    value_net=value_net, 
                    LEN=LEN, 
                    num_games=50,
                    MCTS_SIM = MCTS_SIM , 
                    MCTS_PARALLEL = MCTS_PARALLEL, 
                    BATCH_SIZE_EVAL = BATCH_SIZE_EVAL,
                    device=device,
                    num_workers=mp.cpu_count()//2  # Use half of CPU cores 
                    )
                
                
            # ============================================================
            process_game_data_and_calculate_single_rewards(all_game_data, replay_buffer, file = rebuffer_play_file, pool=pool,logo=False)
            torch.cuda.empty_cache()
            
            # Train policy network
            avg_policy_loss = train_policy_net(policy_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"Round {iteration} average policy loss: {avg_policy_loss}")

            # Train value network
            avg_value_loss = train_value_net(value_net, optimizer, replay_buffer, NUM_BATCHES, BATCH_SIZE, device)
            logger.info(f"Round {iteration} average value loss: {avg_value_loss}")

            # Record training metrics to CSV file
            if avg_policy_loss is not None and avg_value_loss is not None:
                with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, iteration, avg_policy_loss, avg_value_loss])
                logger.debug(f"Saved training metrics to {training_metrics_file}")
            else:
                if avg_policy_loss is not None:
                    with open(training_metrics_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([iteration, iteration, avg_policy_loss, "N/A"])
                    logger.debug("Saved training metrics (value network N/A) to file")
            
            if pool.size >0:
                    # =============== Here get the pool's dictionary and save it to file ===============
                    pool_dict = pool.to_json_dict()   # {'exprs': [...], 'weights': [...]}
                    ic_value_train, ric_value_train = pool.test_ensemble(calculator_train)
                    ic_value_valid, ric_value_valid = pool.test_ensemble(calculator_valid)
                    ic_value_test, ric_value_test = pool.test_ensemble(calculator_test)
                    # Write the results together with iteration to logs/pool_dicts.json
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
                    
                    # =========== Early stopping logic ===========
                    if ic_value_valid > best_ic_valid:
                        best_ic_valid = ic_value_valid
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
        
            # Clear replay_buffer to generate new data in the next self-play
            if no_improve_counter >= 0.2* NUM_ITERATIONS:
                logger.info(f"In iteration {iteration}, ic_value_valid has not improved for {0.2* NUM_ITERATIONS} consecutive times, early stopping training.")
                break
            replay_buffer.buffer.clear()
            torch.cuda.empty_cache()


        logger.info("Search completed.")

    except Exception as e:
        logger.exception("An unhandled exception occurred during training")


if __name__ == "__main__":
    setup_logging()
    main()