import copy
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import torch


Action = Tuple[str, Any]


def apply_dirichlet_noise(root, epsilon: float = 0.25, alpha: float = 0.03) -> None:
    """Add Dirichlet noise to root actions to encourage self-play exploration."""
    if not root.childrens:
        return
    noise = np.random.dirichlet([alpha] * len(root.childrens))
    for idx, child_node in enumerate(root.childrens.values()):
        child_node.p = (1.0 - epsilon) * child_node.p + epsilon * noise[idx]


@dataclass
class EvaluationItem:
    """Common container passed to the policy and value networks."""

    state: Any
    continue_out_game: Optional[int] = None


class MCTSMethods:
    """Interface implemented by each grammar and state-transition variant."""

    max_expression_length = 10
    default_continue_out_game = 3

    @classmethod
    def initial_state(cls):
        raise NotImplementedError

    @classmethod
    def terminal(cls, state) -> bool:
        return state.is_terminal_state()

    @classmethod
    def clone_state(cls, state):
        return copy.deepcopy(state)

    @classmethod
    def apply_action(cls, state, category: str, action: Any) -> None:
        state.apply_action(category, action)

    @classmethod
    def child_continue_out_game(cls, state, parent_continue_out_game: int) -> int:
        return parent_continue_out_game

    @classmethod
    def evaluation_item(cls, node) -> EvaluationItem:
        return EvaluationItem(node.state)

    @classmethod
    def evaluate_batch(cls, items: List[EvaluationItem], policy_net, value_net):
        raise NotImplementedError


class Node:
    """Grammar-independent MCTS tree node."""

    methods: Type[MCTSMethods] = MCTSMethods

    def __init__(
        self,
        state=None,
        parent=None,
        proba: float = 1.0,
        action: Optional[Action] = None,
        continue_out_game: Optional[int] = None,
    ):
        self.state = state if state is not None else self.methods.initial_state()
        self.p = float(proba)
        self.n = 0
        self.w = 0.0
        self.q = 0.0
        self.action = action
        self.childrens: Dict[Action, "Node"] = {}
        self.parent = parent
        if continue_out_game is None:
            continue_out_game = self.methods.default_continue_out_game
        self.continue_out_game = continue_out_game

    def update(self, value: float) -> None:
        """Backpropagate one simulation result and update the mean value."""
        self.n += 1
        self.w += float(value)
        self.q = self.w / self.n

    def is_leaf(self) -> bool:
        return len(self.childrens) == 0

    def expand(self, probas: Optional[Iterable[Dict[str, Any]]]) -> None:
        """Expand legal child nodes from the policy network output."""
        if not probas:
            return
        for action_output in probas:
            category = action_output.get("category")
            action = action_output.get("action")
            prior_prob = action_output.get("prob", 0.0)
            try:
                prior_prob = float(prior_prob)
            except (TypeError, ValueError):
                continue
            if prior_prob <= 0:
                continue

            action_tuple = (category, action)
            if action_tuple in self.childrens:
                continue

            new_state = self.methods.clone_state(self.state)
            self.methods.apply_action(new_state, category, action)
            child_continue = self.methods.child_continue_out_game(
                new_state, self.continue_out_game
            )
            self.childrens[action_tuple] = self.__class__(
                state=new_state,
                parent=self,
                action=action_tuple,
                proba=prior_prob,
                continue_out_game=child_continue,
            )


class EvaluatorThread(threading.Thread):
    """Batch leaf nodes and invoke the variant-specific network evaluator."""

    def __init__(
        self,
        mcts,
        eval_queue,
        result_queue,
        condition_search,
        condition_eval,
        batch_size_eval: int,
    ):
        super().__init__()
        self.mcts = mcts
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.condition_search = condition_search
        self.condition_eval = condition_eval
        self.batch_size_eval = batch_size_eval

    def run(self):
        rounds = max(1, self.mcts.MCTS_SIM // self.mcts.MCTS_PARALLEL)
        for _ in range(rounds):
            self.condition_search.acquire()
            while len(self.eval_queue) < self.mcts.MCTS_PARALLEL:
                self.condition_search.wait()
            self.condition_search.release()

            self.condition_eval.acquire()
            while len(self.result_queue) < self.mcts.MCTS_PARALLEL:
                keys = list(self.eval_queue.keys())
                max_len = min(self.batch_size_eval, len(keys))
                selected_keys = keys[:max_len]
                items = [self.eval_queue[key] for key in selected_keys]
                with torch.no_grad():
                    probas, values = self.mcts.methods.evaluate_batch(
                        items, self.mcts.policy_net, self.mcts.value_net
                    )
                for key, prob, value in zip(selected_keys, probas, values):
                    del self.eval_queue[key]
                    self.result_queue[key] = (prob, value)
                self.condition_eval.notify_all()
            self.condition_eval.release()


class SearchThread(threading.Thread):
    """Run one MCTS simulation."""

    def __init__(
        self,
        mcts,
        eval_queue,
        result_queue,
        thread_id,
        lock,
        condition_search,
        condition_eval,
        c_puct: float,
    ):
        super().__init__()
        self.mcts = mcts
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.thread_id = thread_id
        self.lock = lock
        self.condition_search = condition_search
        self.condition_eval = condition_eval
        self.c_puct = c_puct

    def _select_child(self, node):
        total_count = sum(child.n for child in node.childrens.values())
        branch = max(1, len(node.childrens))
        c_dynamic = self.c_puct * np.sqrt(branch / 40.0)
        best_score = -float("inf")
        best_child = None
        for child in node.childrens.values():
            score = child.q + c_dynamic * child.p * np.sqrt(total_count + 1.0) / (
                1 + child.n
            )
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def run(self):
        current_node = self.mcts.root
        path = [current_node]

        while current_node.childrens and not self.mcts.methods.terminal(current_node.state):
            best_child = self._select_child(current_node)
            if best_child is None:
                break
            current_node = best_child
            path.append(current_node)

        item = self.mcts.methods.evaluation_item(current_node)

        self.condition_search.acquire()
        self.eval_queue[self.thread_id] = item
        self.condition_search.notify()
        self.condition_search.release()

        self.condition_eval.acquire()
        while self.thread_id not in self.result_queue:
            self.condition_eval.wait()
        probas, value = self.result_queue.pop(self.thread_id)
        self.condition_eval.release()

        self.lock.acquire()
        if not self.mcts.methods.terminal(current_node.state):
            current_node.expand(probas)
            if current_node.parent is None:
                apply_dirichlet_noise(
                    current_node,
                    epsilon=self.mcts.root_dirichlet_epsilon,
                    alpha=self.mcts.root_dirichlet_alpha,
                )
        for node in path:
            node.update(float(value))
        self.lock.release()


class MCTS:
    """Shared MCTS engine configured through an injected methods adapter."""

    NodeClass = Node
    methods: Type[MCTSMethods] = MCTSMethods

    def __init__(
        self,
        policy_net,
        value_net,
        MCTS_SIM,
        MCTS_PARALLEL,
        BATCH_SIZE_EVAL,
        device=torch.device("cpu"),
        root_dirichlet_epsilon: float = 0.0,
        root_dirichlet_alpha: float = 0.03,
    ):
        self.root = self.NodeClass()
        self.policy_net = policy_net
        self.value_net = value_net
        self.MCTS_SIM = MCTS_SIM
        self.MCTS_PARALLEL = max(1, MCTS_PARALLEL)
        self.BATCH_SIZE_EVAL = max(1, BATCH_SIZE_EVAL)
        self.device = device
        self.root_dirichlet_epsilon = float(root_dirichlet_epsilon)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)

    def calculate_action_probs_from_node2(self):
        return self._calculate_action_probs(temperature=None)

    def calculate_action_probs_from_node(self, temperature: float = 1.0):
        return self._calculate_action_probs(temperature=temperature)

    def _calculate_action_probs(self, temperature: Optional[float]):
        action_counts = {action: child.n for action, child in self.root.childrens.items()}
        if not action_counts:
            return {}
        if temperature is None:
            action_probs = {action: float(count) for action, count in action_counts.items()}
        else:
            temperature = max(float(temperature), 1e-8)
            action_probs = {
                action: float(count) ** (1.0 / temperature)
                for action, count in action_counts.items()
            }
        total = sum(action_probs.values())
        if total > 0:
            return {action: prob / total for action, prob in action_probs.items()}
        uniform = 1.0 / len(action_counts)
        return {action: uniform for action in action_counts}

    def advance(self, move):
        if move not in self.root.childrens:
            raise KeyError(f"Move {move} not found in childrens")
        self.root = self.root.childrens[move]

    def search(self, competitive, c_puct, temperature):
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()
        eval_queue = OrderedDict()
        result_queue = {}

        evaluator = EvaluatorThread(
            self,
            eval_queue,
            result_queue,
            condition_search,
            condition_eval,
            batch_size_eval=self.BATCH_SIZE_EVAL,
        )
        evaluator.start()

        rounds = max(1, self.MCTS_SIM // self.MCTS_PARALLEL)
        for _ in range(rounds):
            threads = []
            for thread_id in range(self.MCTS_PARALLEL):
                thread = SearchThread(
                    self,
                    eval_queue,
                    result_queue,
                    thread_id,
                    lock,
                    condition_search,
                    condition_eval,
                    c_puct=c_puct,
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        evaluator.join()

        if competitive:
            final_probas = self.calculate_action_probs_from_node2()
        else:
            final_probas = self.calculate_action_probs_from_node(temperature=temperature)
        if not final_probas:
            return {}, None

        actions = list(final_probas.keys())
        probabilities = list(final_probas.values())
        if competitive:
            final_move = max(zip(actions, probabilities), key=lambda x: x[1])[0]
        else:
            final_move = random.choices(actions, weights=probabilities, k=1)[0]

        self.advance(final_move)
        return final_probas, final_move


def build_mcts_classes(methods: Type[MCTSMethods]):
    """Build variant-specific Node and MCTS classes for legacy entry points."""

    class VariantNode(Node):
        pass

    VariantNode.methods = methods

    class VariantMCTS(MCTS):
        pass

    VariantMCTS.methods = methods
    VariantMCTS.NodeClass = VariantNode

    return VariantNode, VariantMCTS
