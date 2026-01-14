import copy
import numpy as np
import torch
import threading
import time
import random
from collections import OrderedDict
from numba import jit
from env.alpha_env import AlphaEnv
from networks.policy_ import evaluate_policy_network
from networks.value_ import evaluate_expression_value
import multiprocessing

def apply_dirichlet_noise(root, epsilon=0.25, alpha=0.03):
    """
    Apply Dirichlet noise to the child nodes of the root node to adjust prior probabilities.

    Parameters:
    root: The current node, containing child nodes and original prior probabilities
    epsilon: The noise ratio, controlling the combination ratio of original priors and noise
    alpha: The parameter of the Dirichlet distribution, controlling the distribution of the noise
    """
    # Get the number of child nodes
    num_children = len(root.childrens)
    
    # Generate Dirichlet noise
    noise = np.random.dirichlet([alpha] * num_children)
    
    # Iterate through each child node and modify the prior probability
    for idx, (action, child_node) in enumerate(root.childrens.items()):
        # Modify the prior probability: combine the original prior with Dirichlet noise
        original_P = child_node.p
        child_node.p = (1 - epsilon) * original_P + epsilon * noise[idx]
        


class Node:

    def __init__(self, state = AlphaEnv(), parent=None, proba=1, action = None):
        """
        p: The probability of this node, given by the policy network
        n: The number of times this node has been visited during simulations
        w: The total action value of this node, given by the value network
        q: The average action value (w / n)
        """
        self.state = state  # Environment state
        self.p = proba
        self.n = 0
        self.w = 0
        self.q = 0
        self.action = action  # The action from the parent node to this node
        self.childrens = {}  # Child nodes, with action as key and node as value
        self.parent = parent
    

    def update(self, v):
        """ Update the node's statistics after a simulation """

        self.w = self.w + v
        self.q = self.w / self.n if self.n > 0 else 0

    def update2(self, v):
        self.w += v
        q_avg = self.w / self.n if self.n > 0 else 0
        
        a=0.1
        if self.childrens:
            q_max = max(child.q for child in self.childrens.values())
        else:
            q_max = q_avg  # If no child nodes, q_max defaults to q_avg

        self.q = a * q_avg + (1 - a) * q_max

    def is_leaf(self):
        """ Check if the node is a leaf node """

        return len(self.childrens) == 0

    def expand(self, probas):

        # Create corresponding child nodes
        for action_output in probas:
            action_category = action_output.get('category')
            action_detail = action_output.get('action')
            prior_prob = action_output.get('prob')

            # Only create child nodes for actions with probability greater than 0
            if prior_prob > 0:
                action_tuple = (action_category, action_detail)

                if action_tuple in self.childrens:
                    continue  # Skip already created child nodes

                # Apply the action to generate a new state
                new_state = copy.deepcopy(self.state)
                new_state.apply_action(action_category, action_detail)

                # Create a child node
                child_node = Node(
                    state=new_state,
                    parent=self,
                    action=action_tuple,
                    proba=prior_prob,
                )
                self.childrens[action_tuple] = child_node



class EvaluatorThread(threading.Thread):
    def __init__(self, mcts, value_net, policy_net, eval_queue, result_queue, condition_search, condition_eval, BATCH_SIZE_EVAL):
        """ Used to batch evaluate positions during the tree search process """

        threading.Thread.__init__(self)
        self.mcts = mcts
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.value_net = value_net
        self.policy_net = policy_net
        self.condition_search = condition_search
        self.condition_eval = condition_eval
        self.BATCH_SIZE_EVAL = BATCH_SIZE_EVAL


    def run(self):
        for sim in range(self.mcts.MCTS_SIM // self.mcts.MCTS_PARALLEL):

            ## Wait for eval_queue to be filled with new positions to evaluate
            self.condition_search.acquire()
            while len(self.eval_queue) < self.mcts.MCTS_PARALLEL:
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < self.mcts.MCTS_PARALLEL:
                keys = list(self.eval_queue.keys())
                max_len = self.BATCH_SIZE_EVAL if len(keys) > self.BATCH_SIZE_EVAL else len(keys)

                expressions = list(self.eval_queue.values())[0:max_len]  # Get positions to evaluate

                # Get the value (v) and policy probabilities (probas) for each expression
                v = [evaluate_expression_value(expression, value_net=self.value_net) for expression in expressions]
                probas = [evaluate_policy_network(expression, policy_net=self.policy_net)
                          for expression in expressions]

                ## Replace the states in eval_queue with results and notify all threads that results are available
                for idx, i in zip(keys, range(max_len)):
                    del self.eval_queue[idx]
                    self.result_queue[idx] = (probas[i], v[i])

                self.condition_eval.notifyAll()
            self.condition_eval.release()
    



class SearchThread(threading.Thread):

    def __init__(self, mcts, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval, c_puct):
        """ Run a single simulation """

        threading.Thread.__init__(self)
        self.mcts = mcts
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search
        self.c_puct = c_puct
    

    def run(self):
        ##2. This needs to be changed to our environment and state
        current_node = self.mcts.root

        #current_node = current_node.select(path = [],c_puct = self.c_puct)
        c_puct = self.c_puct
        current_node = self.mcts.root
    
        # Continue searching while the current node is expanded and not in a terminal state
        while  current_node.childrens and not current_node.state.is_terminal_state():
            # If the current node has no children, treat it as a leaf node and stop searching
            best_score = -float('inf')
            best_child = None
            total_cout=0
            branch=0
            for child in current_node.childrens.values():
                total_cout = child.n + total_cout
                branch = branch + 1

            # Adaptive exploration coefficient (square root scaling)
            # Bref is a reference branch number, e.g., 20, can be adjusted according to your search tree size
            Bref = 40
            c_dynamic = c_puct * np.sqrt(branch / Bref)
                
            # Select the child node with the highest score according to the PUCT formula
            for child in current_node.childrens.values():
                score = child.q + c_dynamic * child.p * np.sqrt(total_cout) / (1 + child.n)
                if score > best_score:
                    best_score = score
                    best_child = child
        
            # If a best child node is found, update the current node and path
            if best_child is not None:
                current_node = best_child
            else:
                break  # If no suitable child node is found, exit the loop

            ## Virtual loss (due to multithreading)
            self.lock.acquire()
            current_node.n += 1
            self.lock.release()
        
        expression = current_node.state.state_description()
        if not current_node.state.is_terminal_state():

            ## Add the current leaf node's state with random rotation transformations to the evaluation queue
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = expression
            self.condition_search.notify()
            self.condition_search.release()

            ## Wait for the evaluation thread to complete
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            ## To avoid GPU memory leaks, copy the result
            result = self.result_queue.pop(self.thread_id)
            probas = result[0]
            v = float(result[1])
            self.condition_eval.release()

            self.lock.acquire()    
            current_node.expand(probas)
            
            ## Add noise to the root node
            if not current_node.parent:
                apply_dirichlet_noise(current_node, epsilon=0, alpha=0.03)

            ## Backpropagate the simulation result
            while current_node.parent:
                current_node.update(v)
                current_node = current_node.parent
            self.lock.release()
        else:

            ## Add the current leaf node's state with random rotation transformations to the evaluation queue
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = expression
            self.condition_search.notify()
            self.condition_search.release()

            ## Wait for the evaluation thread to complete
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            ## To avoid GPU memory leaks, copy the result
            result = self.result_queue.pop(self.thread_id)
            v = float(result[1])
            self.condition_eval.release()
            
            self.lock.acquire()
            ## Backpropagate the simulation result
            while current_node.parent:
                current_node.update(v)
                current_node = current_node.parent
            self.lock.release()




class MCTS:
    def __init__(self, policy_net, value_net, MCTS_SIM, MCTS_PARALLEL, BATCH_SIZE_EVAL, device=torch.device('cpu')):
        self.root = Node()
        self.policy_net = policy_net
        self.value_net = value_net
        self.MCTS_SIM = MCTS_SIM
        self.MCTS_PARALLEL = MCTS_PARALLEL
        self.BATCH_SIZE_EVAL = BATCH_SIZE_EVAL
        self.device = device  # New device attribute
    
    def calculate_action_probs_from_node2(self):
        """
        Calculate action probabilities from a given node.

        Parameters:
        - node: The current node object, containing child nodes and visit counts
        - temperature: The temperature parameter, used to smooth the policy

        Returns:
        - action_probs: The calculated action probability dictionary
        """
        # Get the visit counts of the current node's child nodes
        action_counts = {action: child.n for action, child in self.root.childrens.items()}

        if not action_counts:
            return {}

        # Calculate action probabilities π(a) ∝ N(s, a)^(1/τ)
        action_probs = {action: count   for action, count in action_counts.items()}

        # Normalize action probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {action: prob / total for action, prob in action_probs.items()}
        else:
            # If the total is 0, use a uniform distribution
            num_actions = len(action_counts)
            action_probs = {action: 1.0 / num_actions for action in action_counts}

        return action_probs
    
    def calculate_action_probs_from_node(self, temperature=1):
        """
        Calculate action probabilities from a given node.

        Parameters:
        - node: The current node object, containing child nodes and visit counts
        - temperature: The temperature parameter, used to smooth the policy

        Returns:
        - action_probs: The calculated action probability dictionary
        """
        # Get the visit counts of the current node's child nodes
        action_counts = {action: child.n for action, child in self.root.childrens.items()}

        if not action_counts:
            return {}

        # Calculate action probabilities π(a) ∝ N(s, a)^(1/τ)
        action_probs = {action: (count ** (1.0 / temperature)) for action, count in action_counts.items()}

        # Normalize action probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {action: prob / total for action, prob in action_probs.items()}
        else:
            # If the total is 0, use a uniform distribution
            num_actions = len(action_counts)
            action_probs = {action: 1.0 / num_actions for action in action_counts}

        return action_probs
    
    def advance(self, move):
        """ Manually advance the tree, used for GTP """
    
        # Use move as the key to access the child node, instead of using index idx
        if move in self.root.childrens:
            self.root = self.root.childrens[move]  # Update the root node to the corresponding child node
        else:
            raise KeyError(f"Move {move} not found in childrens")

    def search(self, competitive,c_puct,temperature):
        """
        Search for the best action through the game tree, updating node statistics with the policy network and value network
        """

        ## Lock to ensure thread synchronization
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        ## Create a single thread for the evaluator (currently)
        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluatorThread(self, self.value_net, self.policy_net, eval_queue, result_queue, condition_search, condition_eval, BATCH_SIZE_EVAL = self.BATCH_SIZE_EVAL)
        evaluator.start()

        threads = []
        ## Each thread performs an exact number of simulations
        for sim in range(self.MCTS_SIM // self.MCTS_PARALLEL):
            for thread_id in range(self.MCTS_PARALLEL):
                threads.append(SearchThread(self, eval_queue, result_queue, thread_id, 
                                        lock, condition_search, condition_eval,c_puct=c_puct))
                threads[-1].start()
            for thread in threads:
                thread.join()
        evaluator.join()

        ## Select the best action
        if competitive:
            final_probas = self.calculate_action_probs_from_node2()
        else:
            final_probas = self.calculate_action_probs_from_node(temperature=temperature)

        # Convert action_probs to a list
        actions = list(final_probas.keys())
        probabilities = list(final_probas.values())

        ## Determine action selection method based on whether it's competitive
        if competitive:
            # Select the most likely action (deterministic selection)
            final_move = max(zip(actions, probabilities), key=lambda x: x[1])[0]
        else:
            final_move = None
            cumulative_probabilities = []
            cumulative_sum = 0
            for p in probabilities:
                cumulative_sum += p
                cumulative_probabilities.append(cumulative_sum)

            random_value = random.random()

            for i, cumulative_prob in enumerate(cumulative_probabilities):
                if random_value < cumulative_prob:
                    final_move = actions[i]
                    break
            
            # If final_move is still not assigned after the loop, select the last action as a fallback
            if final_move is None:
                final_move = actions[-1]

        self.advance(final_move)

        return final_probas, final_move
    
if __name__ == "__main__":
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
        from models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
        from train_preparation_pool import *  # Import after logging configuration
        import numpy as np
        import random

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        token_map = create_token_map()
        vocab_size = len(token_map) + 1
        embed_dim = 128
        nhead = 8
        nhid = 128
        nlayers = 2
        dropout = 0.1
        
        #seed = 3  # Can choose any number you like
        #set_seed(seed)

        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout
        ).to(device)

        #feature_extractor.apply(initialize_weights)

        # Initialize policy network and value network
        policy_net = PolicyNetwork(feature_extractor=feature_extractor, embed_dim=embed_dim).to(device)
        value_net = ValueNetwork(feature_extractor=feature_extractor, embed_dim=embed_dim).to(device)

        env = AlphaEnv()
        root = Node(state=copy.deepcopy(env))
        mcts = MCTS(policy_net, value_net, device)
        print(mcts.root.state.state_description())
        selected_action = mcts.search(competitive=False)
        print(mcts.root.state.state_description())
        selected_action = mcts.search(competitive=False)
        print(mcts.root.state.state_description())
        selected_action = mcts.search(competitive=False)
        print(mcts.root.state.state_description())
        selected_action = mcts.search(competitive=False)
        print(mcts.root.state.state_description())



