# mcts.py

import torch
import numpy as np
from alpha_env import AlphaEnv
from policy_ import PolicyNetwork, evaluate_policy_network
from value_ import ValueNetwork, evaluate_expression_value
from feature_ import FeatureExtractor
from token_v import create_token_map, parse_expression_to_tokens
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import pydot  # 新增导入 pydot
import sys

# 确保 evaluator_ 中有 parse_expression_from_string 和 evaluate_expression_ic
from evaluator_ import parse_expression_from_string, evaluate_expression_ic

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置全局日志级别为 INFO
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=1.0, row=0, col=0):
        self.state = state  # AlphaEnv 实例
        self.parent = parent  # 父节点
        self.action = action  # 从父节点到此节点的动作
        self.children = {}  # 子节点，键为动作，值为节点
        self.visit_count = 0  # 访问计数 N(s)
        self.total_value = 0.0  # 累计价值 W(s)
        self.Q = 0.0  # 平均价值 Q(s)
        self.P = prior_prob  # 先验概率 P(s, a)
        self.is_expanded = False  # 节点是否已扩展

        # 分配行和列索引
        self.row = row
        self.col = col

        # 初始化锁
        self.lock = threading.Lock()

        # 跟踪输出
        if parent and action:
            logger.debug(f"创建子节点: 父节点标号 ({parent.row}, {parent.col}) 动作 {action}，新节点标号 ({self.row}, {self.col})")
        else:
            logger.debug(f"创建根节点，初始标号 ({self.row}, {self.col})")

    def expand(self, policy_net, depth_counts):
        """扩展节点：使用策略网络获取先验概率，生成子节点"""
        with self.lock:
            if self.is_expanded:
                logger.debug(f"节点标号 ({self.row}, {self.col}) 已扩展，跳过扩展步骤。")
                return

            # 获取可能的动作类别
            possible_actions = self.state.get_possible_actions()
            logger.info(f"扩展节点 ({self.row}, {self.col})，可能的动作类别: {possible_actions}")

            if not possible_actions:
                # 无可用动作，标记为已扩展
                logger.info(f"节点 ({self.row}, {self.col}) 无可用动作，标记为已扩展。")
                self.is_expanded = True
                return

            # 获取当前状态的表达式
            expression = self.state.state_description()

            # 使用策略网络获取先验概率和动作
            action_outputs = evaluate_policy_network(
                expression,
                policy_net=policy_net,
                possible_actions=possible_actions
            )

            if not action_outputs:
                logger.warning(f"策略网络未返回动作，表达式: {expression}，节点 ({self.row}, {self.col})")
                self.is_expanded = True
                return

            logger.debug(f"策略网络返回动作数: {len(action_outputs)}")

            # 用于存储扩展的动作信息
            expanded_actions = []

            for action_output in action_outputs:
                action_category = action_output.get('category')
                action = action_output.get('action')
                prior_prob = action_output.get('prob')

                # 检查动作类别是否在 possible_actions 中
                if action_category not in possible_actions:
                    logger.warning(f"动作类别 {action_category} 不在可能的动作类别中，跳过。")
                    continue  # 跳过不合法的动作类别

                # 应用动作，生成新状态
                new_state = copy.deepcopy(self.state)
                new_state.apply_action(action_category, action)

                # 获取新的列索引
                child_depth = self.row + 1
                with self.parent.lock if self.parent else threading.Lock():
                    child_col = depth_counts.get(child_depth, 0)
                    depth_counts[child_depth] = child_col + 1

                # 创建子节点
                child_node = MCTSNode(
                    state=new_state,
                    parent=self,
                    action=(action_category, action),
                    prior_prob=prior_prob,
                    row=child_depth,
                    col=child_col
                )
                self.children[(action_category, action)] = child_node

                # 记录扩展的动作
                expanded_actions.append({
                    'action_category': action_category,
                    'action': action,
                    'prior_prob': prior_prob,
                    'new_state_description': new_state.state_description()
                })

            if not self.children:
                logger.info(f"节点 ({self.row}, {self.col}) 下没有生成子节点。")
            else:
                self.is_expanded = True
                logger.info(f"节点 ({self.row}, {self.col}) 成功扩展，生成 {len(self.children)} 个子节点。")
                # 输出扩展后的状态和动作
                for idx, act in enumerate(expanded_actions, 1):
                    logger.info(f"  扩展动作 {idx}: 分类={act['action_category']}, 动作={act['action']}, 先验概率={act['prior_prob']:.2f}")
                    logger.info(f"    新状态: {act['new_state_description']}")

    def select(self, c_puct):
        """使用 PUCT 公式选择子节点"""
        if not self.children:
            logger.debug(f"节点 ({self.row}, {self.col}) 无子节点可选择。")
            return None

        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            Q = child.Q
            U = c_puct * child.P * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = Q + U

            if score > best_score:
                best_score = score
                best_child = child

        if best_child:
            logger.debug(f"节点 ({self.row}, {self.col}) 选择子节点 ({best_child.row}, {best_child.col})，分数: {best_score:.2f}")

        return best_child

    def update(self, leaf_value):
        """反向传播：更新节点的访问计数和价值，并递归更新父节点"""
        with self.lock:
            self.visit_count += 1
            self.total_value += leaf_value
            self.Q = self.total_value / self.visit_count  # 更新平均价值

            logger.debug(f"更新节点 ({self.row}, {self.col})，访问次数: {self.visit_count}，总价值: {self.total_value:.2f}，平均价值 Q: {self.Q:.2f}")

        # 在非零和博弈中，不需要取反
        if self.parent:
            self.parent.update(leaf_value)

    def is_terminal(self):
        """判断是否为终止状态"""
        terminal = self.state.is_terminal_state()
        logger.debug(f"检查终止状态，节点 ({self.row}, {self.col})，是否终止: {terminal}")
        return terminal

    def __hash__(self):
        # 为了在 networkx 中使用节点，需要定义 __hash__ 方法
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

class MCTS:
    def __init__(self, policy_net, value_net, c_puct=5, n_simulations=100, n_threads=4, device=torch.device('cpu')):
        self.root = None
        self.policy_net = policy_net
        self.value_net = value_net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.depth_counts = {}  # 跟踪每一层的节点数量
        self.lock = threading.Lock()  # 全局锁，用于同步访问depth_counts
        self.device = device  # 新增设备属性

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=n_threads)

    def simulate(self, node, sim_id):
        """模拟并评估叶子节点的价值，考虑策略网络输出的绝对值"""
        expression = node.state.state_description()
        logger.info(f"模拟 {sim_id}: 评估表达式: {expression}")
        leaf_value = evaluate_expression_value(
            expression,
            value_net=self.value_net
        )
        leaf_value = abs(leaf_value)  # 取绝对值
        logger.info(f"模拟 {sim_id}: 叶子价值 (绝对值): {leaf_value:.2f}")
        return leaf_value

    def search(self, initial_state):
        """进行 MCTS 搜索，返回最优动作"""
        self.root = MCTSNode(state=copy.deepcopy(initial_state), row=0, col=0)
        logger.info("开始 MCTS 搜索...")

        futures = []
        for sim in range(1, self.n_simulations + 1):
            futures.append(self.executor.submit(self.run_simulation, sim))

        # 等待所有模拟完成
        for future in futures:
            future.result()

        logger.info("MCTS 搜索结束。")

        # 在搜索结束后，选择访问次数最多的动作
        if self.root.children:
            best_action = max(
                self.root.children.items(),
                key=lambda item: item[1].visit_count
            )[0]
            logger.info(f"最佳动作: {best_action}，访问次数: {self.root.children[best_action].visit_count}")
            return best_action
        else:
            logger.warning("未能找到最佳动作。")
            return None

    def run_simulation(self, sim_id):
        """单次模拟过程，供线程调用"""
        try:
            node = self.root
            search_path = [node]

            # Selection 阶段
            while True:
                if node.is_terminal():
                    logger.debug(f"模拟 {sim_id}: 节点 ({node.row}, {node.col}) 是终止状态，进入 Simulation 阶段。")
                    break
                if not node.is_expanded:
                    logger.debug(f"模拟 {sim_id}: 节点 ({node.row}, {node.col}) 未扩展，进入 Expansion 阶段。")
                    break
                selected_child = node.select(self.c_puct)
                if selected_child is None:
                    logger.debug(f"模拟 {sim_id}: 节点为 None，结束当前模拟。")
                    break  # 无可选择的子节点，结束当前模拟
                node = selected_child
                search_path.append(node)

            # 如果节点为 None，跳过后续步骤
            if node is None:
                return

            # Expansion 阶段
            if not node.is_expanded:
                with self.lock:
                    node.expand(self.policy_net, self.depth_counts)

            # Simulation 阶段
            leaf_value = 0
            if not node.is_terminal():
                leaf_value = self.simulate(node, sim_id)
            else:
                # 终止状态的价值，根据实际情况设定
                leaf_value = 0
                logger.debug(f"模拟 {sim_id}: 终止状态的叶子价值设为 0。")

            # Backpropagation 阶段
            for n in reversed(search_path):
                n.update(leaf_value)
                # 在非零和博弈中，不需要取反
                # leaf_value = leaf_value  # 这行可以省略
        except Exception as e:
            logger.exception(f"模拟 {sim_id} 中发生异常: {e}")

    def get_best_path(self):
        """获取从根节点到最优叶子节点的最佳路径，并返回对应的因子表达式"""
        node = self.root
        best_path = []
        logger.info("获取最佳路径...")
        while node:
            if node.action:
                best_path.append(node.action)
                logger.debug(f"路径动作: {node.action}，节点 ({node.row}, {node.col})")
            if node.children:
                # 选择访问次数最多的子节点
                node = max(node.children.values(), key=lambda n: n.visit_count)
            else:
                break
        return best_path

    def export_search_tree(self, filename_dot="mcts_tree.dot", filename_pdf="mcts_tree.pdf"):
        """导出搜索树为 DOT 和 PDF 文件，包含所有访问过的节点 (N >=1)，并在 PDF 中标注完整表达式的 IC"""
        G = pydot.Dot(graph_type='digraph')

        node_id_map = {}  # Mapping from node object to unique ID
        current_id = 0
        stack = [self.root]

        while stack:
            node = stack.pop()
            if node.visit_count < 1:
                logger.debug(f"节点 ({node.row}, {node.col}) 未被访问，跳过。")
                continue
            if node in node_id_map:
                continue
            node_id_map[node] = current_id
            logger.debug(f"添加节点 ({node.row}, {node.col}) 到搜索树，ID: {current_id}")
            current_id += 1
            for action, child in node.children.items():
                if child.visit_count >= 1:
                    stack.append(child)

        # 添加所有边
        for node in node_id_map:
            for action, child in node.children.items():
                if child.visit_count >=1 and child in node_id_map:
                    edge = pydot.Edge(str(node_id_map[node]), str(node_id_map[child]))
                    G.add_edge(edge)
                    logger.debug(f"添加边从节点 {node_id_map[node]} 到节点 {node_id_map[child]}")

        # 为节点创建标签，显示动作、访问次数和平均价值，以及 IC（如果是终止状态）
        for node, node_id in node_id_map.items():
            if node.action:
                action_category, action = node.action
                if node.state.is_terminal_state():
                    try:
                        expression_str = node.state.state_description()
                        expression_obj = parse_expression_from_string(expression_str)
                        ic = evaluate_expression_ic(expression_obj)
                        ic_label = f", IC={ic:.2f}"
                    except Exception as e:
                        ic_label = ", IC=Error"
                        logger.warning(f"计算节点 ({node.row}, {node.col}) 的 IC 时出错: {e}")
                else:
                    ic_label = ""
                label = f"{action_category}:{action}\\nN={node.visit_count}, Q={node.Q:.2f}{ic_label}\\n({node.row}, {node.col})"
            else:
                label = f"Root\\nN={node.visit_count}\\n({node.row}, {node.col})"
            # 创建 pydot Node，确保标签被正确引用
            node_pydot = pydot.Node(
                str(node_id),
                label=label,
                shape='box',
                style='filled',
                fillcolor='lightblue',
                fontsize='12',
                fontname='Arial'
            )
            G.add_node(node_pydot)

        # 将图形保存为 DOT 文件
        try:
            G.write_dot(filename_dot)
            logger.info(f"DOT 文件已保存为 {filename_dot}")
        except Exception as e:
            logger.error(f"保存 DOT 文件失败: {e}")
            return

        # 使用 Graphviz 将 DOT 文件转换为 PDF
        try:
            # 直接使用 pydot 的 write_pdf 方法
            G.write_pdf(filename_pdf)
            logger.info(f"搜索树已保存为 {filename_pdf}，请使用 PDF 查看器打开查看。")
        except Exception as e:
            logger.error(f"保存 PDF 文件失败: {e}")