import copy
import numpy as np
import torch
import threading
import time
import random
from collections import OrderedDict
from numba import jit
from env.alpha_env import AlphaEnv
from env.token_v import convert_state_tokens_to_ids
from networks.policy_ import evaluate_policy_network
from networks.value_ import evaluate_expression_value
import multiprocessing

def apply_dirichlet_noise(root, epsilon=0.25, alpha=0.03):
    """
    Apply Dirichlet noise to the child nodes of the root node to adjust the prior probabilities.

    Parameters:
    root: The current node containing child nodes and their original prior probabilities
    epsilon: The noise ratio controlling the combination of the original prior and the noise
    alpha: The parameter of the Dirichlet distribution, controlling the distribution of the noise
    """
    # Get the number of child nodes
    num_children = len(root.childrens)
    
    # Generate Dirichlet noise
    noise = np.random.dirichlet([alpha] * num_children)
    
    # Modify the prior probability for each child node by blending the original prior and the Dirichlet noise
    for idx, (action, child_node) in enumerate(root.childrens.items()):
        original_P = child_node.p
        child_node.p = (1 - epsilon) * original_P + epsilon * noise[idx]


class Node:

    def __init__(self, state = AlphaEnv(), parent=None, proba=1, action=None):
        """
        p: The probability of the node, given by the policy network
        n: The number of times this node has been visited during simulation
        w: The total action value of this node, given by the value network
        q: The average action value (w / n)
        """
        self.state = state  # The state of the environment
        self.p = proba
        self.n = 0
        self.w = 0
        self.q = 0
        self.action = action  # The action taken from the parent node to this node
        self.childrens = {}  # The child nodes, with action as key and node as value
        self.parent = parent
    

    def update(self, v):
        """ Update the node's statistics after a simulation """
        self.w = self.w + v
        self.q = self.w / self.n if self.n > 0 else 0

    def is_leaf(self):
        """ Check if the node is a leaf node """
        return len(self.childrens) == 0

    def expand(self, probas):
        """ Expand the node by creating child nodes based on the given probabilities """

        for action_output in probas:
            action_category = action_output.get('category')
            action_detail = action_output.get('action')
            prior_prob = action_output.get('prob')
            
            # Only create child nodes for actions with a probability greater than 0
            if prior_prob > 0:
                action_tuple = (action_category, action_detail)

                if action_tuple in self.childrens:
                    continue  # Skip already existing child nodes

                # Apply the action to generate a new state
                new_state = copy.deepcopy(self.state)
                new_state.apply_action(action_category, action_detail)

                # Create the child node
                child_node = Node(
                    state=new_state,
                    parent=self,
                    action=action_tuple,
                    proba=prior_prob,
                )
                self.childrens[action_tuple] = child_node


class EvaluatorThread(threading.Thread):
    def __init__(self, mcts, value_net, policy_net, eval_queue, result_queue, condition_search, condition_eval, BATCH_SIZE_EVAL):
        """ Thread for batch evaluating states during tree search """

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

            # Wait until the eval_queue is populated with new states to evaluate
            self.condition_search.acquire()
            while len(self.eval_queue) < self.mcts.MCTS_PARALLEL:
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < self.mcts.MCTS_PARALLEL:
                keys = list(self.eval_queue.keys())
                max_len = self.BATCH_SIZE_EVAL if len(keys) > self.BATCH_SIZE_EVAL else len(keys)

                expressions = list(self.eval_queue.values())[0:max_len]  # Get the states to evaluate

                # Get value (v) and policy probabilities (probas) for each state
                v = [evaluate_expression_value(expression_tokens=convert_state_tokens_to_ids(expression.state), value_net=self.value_net) for expression in expressions]
                probas = [evaluate_policy_network(expression_tokens=convert_state_tokens_to_ids(expression.state), possible_actions=expression.get_possible_actions(), policy_net=self.policy_net)
                          for expression in expressions]

                # Replace the states in eval_queue with their evaluation results and notify threads
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
        current_node = self.mcts.root

        c_puct = self.c_puct
        current_node = self.mcts.root

        # Continue search if the current node has children and is not a terminal state
        while current_node.childrens and not current_node.state.is_terminal_state():
            best_score = -float('inf')
            best_child = None
            total_count = 0
            branch = 0
            for child in current_node.childrens.values():
                total_count += child.n
                branch += 1

            # Adaptive exploration coefficient (scaled by square root)
            Bref = 40  # A reference branch count, can be adjusted based on your tree size
            c_dynamic = c_puct * np.sqrt(branch / Bref)
                
            # Choose the child node with the highest score based on the PUCT formula
            for child in current_node.childrens.values():
                score = child.q + c_dynamic * child.p * np.sqrt(total_count) / (1 + child.n)
                if score > best_score:
                    best_score = score
                    best_child = child
        
            if best_child is not None:
                current_node = best_child
            else:
                break  # Exit loop if no suitable child node is found

            # Virtual loss to avoid race conditions in multithreading
            self.lock.acquire()
            current_node.n += 1
            self.lock.release()
        
        expression = current_node.state
        if not current_node.state.is_terminal_state():

            # Add the current leaf node's state with random rotations to the eval_queue
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = expression
            self.condition_search.notify()
            self.condition_search.release()

            # Wait for the evaluation thread to finish
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            # Copy the result to avoid GPU memory leak
            result = self.result_queue.pop(self.thread_id)
            probas = result[0]
            v = float(result[1])
            self.condition_eval.release()

            self.lock.acquire()    
            current_node.expand(probas)
            
            # Apply Dirichlet noise to the root node
            if not current_node.parent:
                apply_dirichlet_noise(current_node, epsilon=0, alpha=0.03)

            # Backpropagate the simulation result
            while current_node.parent:
                current_node.update(v)
                current_node = current_node.parent
            self.lock.release()
        else:

            # Add the current leaf node's state with random rotations to the eval_queue
            self.condition_search.acquire()
            self.eval_queue[self.thread_id] = expression
            self.condition_search.notify()
            self.condition_search.release()

            # Wait for the evaluation thread to finish
            self.condition_eval.acquire()
            while self.thread_id not in self.result_queue.keys():
                self.condition_eval.wait()

            # Copy the result to avoid GPU memory leak
            result = self.result_queue.pop(self.thread_id)
            v = float(result[1])
            self.condition_eval.release()
            
            self.lock.acquire()
            # Backpropagate the simulation result
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
        self.device = device  # Added device attribute
    
    def calculate_action_probs_from_node2(self):
        """
        Calculate the action probabilities from a given node.

        Parameters:
        - node: The current node object containing children and visit count
        - temperature: The temperature parameter used for smoothing the policy

        Returns:
        - action_probs: A dictionary of calculated action probabilities
        """
        # Get the visit count for the child nodes
        action_counts = {action: child.n for action, child in self.root.childrens.items()}

        if not action_counts:
            return {}

        # Calculate action probabilities π(a) ∝ N(s, a)^(1/τ)
        action_probs = {action: count for action, count in action_counts.items()}

        # Normalize the action probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {action: prob / total for action, prob in action_probs.items()}
        else:
            # If total is 0, use a uniform distribution
            num_actions = len(action_counts)
            action_probs = {action: 1.0 / num_actions for action in action_counts}

        return action_probs
    
    def calculate_action_probs_from_node(self, temperature=1):
        """
        Calculate the action probabilities from a given node.

        Parameters:
        - node: The current node object containing children and visit count
        - temperature: The temperature parameter used for smoothing the policy

        Returns:
        - action_probs: A dictionary of calculated action probabilities
        """
        # Get the visit count for the child nodes
        action_counts = {action: child.n for action, child in self.root.childrens.items()}

        if not action_counts:
            return {}

        # Calculate action probabilities π(a) ∝ N(s, a)^(1/τ)
        action_probs = {action: (count ** (1.0 / temperature)) for action, count in action_counts.items()}

        # Normalize the action probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {action: prob / total for action, prob in action_probs.items()}
        else:
            # If total is 0, use a uniform distribution
            num_actions = len(action_counts)
            action_probs = {action: 1.0 / num_actions for action in action_counts}

        return action_probs
    
    def advance(self, move):
        """ Manually advance the tree using the move as the key """
        if move in self.root.childrens:
            self.root = self.root.childrens[move]  # Update the root node to the corresponding child node
        else:
            raise KeyError(f"Move {move} not found in childrens")

    def search(self, competitive, c_puct, temperature):
        """
        Perform a tree search to find the best action based on the policy and value networks, updating node statistics.

        Parameters:
        competitive: Whether to perform a competitive (greedy) search
        c_puct: Exploration parameter for the PUCT formula
        temperature: The temperature parameter for action selection
        """
        # Locking to ensure thread synchronization
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        # Create a single evaluator thread (for now)
        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluatorThread(self, self.value_net, self.policy_net, eval_queue, result_queue, condition_search, condition_eval, BATCH_SIZE_EVAL=self.BATCH_SIZE_EVAL)
        evaluator.start()

        threads = []
        # Each thread performs a specified number of simulations
        for sim in range(self.MCTS_SIM // self.MCTS_PARALLEL):
            for thread_id in range(self.MCTS_PARALLEL):
                threads.append(SearchThread(self, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval, c_puct=c_puct))
                threads[-1].start()
            for thread in threads:
                thread.join()
        evaluator.join()

        # Select the best action based on whether it's competitive or not
        if competitive:
            final_probas = self.calculate_action_probs_from_node2()
        else:
            final_probas = self.calculate_action_probs_from_node(temperature=temperature)

        actions = list(final_probas.keys())
        probabilities = list(final_probas.values())

        # Choose the action based on the action probabilities
        if competitive:
            # Choose the action with the highest probability (greedy)
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
            
            # If no action was selected, choose the last one as a fallback
            if final_move is None:
                final_move = actions[-1]

        self.advance(final_move)

        return final_probas, final_move
