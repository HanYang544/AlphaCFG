# policy_.py
from collections import OrderedDict  # 确保导入 OrderedDict
import torch
import torch.nn as nn
from token_v import create_token_map, parse_expression_to_tokens, get_first_token
from feature_ import FeatureExtractor

# 类别名称映射
CATEGORY_MAP = {
    0: 'Vari',
    1: 'constant',
    2: 'unaryop',
    3: 'binaryop',
    4: 'rollingop',
    5: 'pairedrollingop',
    # 6: 'num'  移除 num 类
}

# 动作映射
ACTION_MAP = {
    'unaryop': ['Abs', 'Sign', 'Log', 'CSRank'],
    'binaryop': ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Pow'],
    'rollingop': ['Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min', 'Med', 'Mad', 'Rank', 'Delta', 'WMA', 'EMA'],
    'pairedrollingop': ['Cov', 'Corr'],
    'Vari': ['high', 'low', 'volume', 'open', 'close', 'vwap'],
    'num': ['Mapped Integer'],  # num 类的动作
    'constant': ['Mapped Constant']  # constant 类的动作，占位符
}

class PolicyNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim, constant_range=(0, 100), num_range=(1, 100)):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        # 各个类别的输出维度
        self.category_head = nn.Linear(embed_dim, 6)  # Vari 和 constant 类别选择概率
        # 各类别的具体动作映射，使用 nn.ModuleDict
        self.action_heads = nn.ModuleDict({
            'unaryop': nn.Linear(embed_dim, 4),  # 4个动作概率分布
            'binaryop': nn.Linear(embed_dim, 7),  # 7个动作概率分布
            'rollingop': nn.Linear(embed_dim, 15),  # 15个动作概率分布
            'pairedrollingop': nn.Linear(embed_dim, 2),  # 2个动作概率分布
            'Vari': nn.Linear(embed_dim, 6),  # Vari的6个动作
            'constant': nn.Linear(embed_dim, 1),  # constant映射为浮点数
        })
        self.num_head = nn.Linear(embed_dim, 1)  # num 映射为正整数
        self.softmax = nn.Softmax(dim=-1)
        self.constant_range = constant_range
        self.num_range = num_range

    def forward(self, src, src_key_padding_mask=None, first_token=None, possible_actions=None):
        features = self.feature_extractor(src, src_key_padding_mask)
        action_outputs = []

        if first_token == 'E':
            # 获取类别概率
            category_logits = self.category_head(features)
            category_probs = self.softmax(category_logits).squeeze(0)

            for i, category_prob in enumerate(category_probs):
                if i >= len(CATEGORY_MAP):
                    continue  # 避免超出 CATEGORY_MAP 的范围

                category_name = CATEGORY_MAP[i]

                # 如果提供了 possible_actions，则只保留可能的动作类别
                if possible_actions and category_name not in possible_actions:
                    continue

                if category_name == 'constant':
                    # 特殊处理 'constant' 类别
                    constant_value = torch.sigmoid(self.action_heads['constant'](features)) * \
                                     (self.constant_range[1] - self.constant_range[0]) + self.constant_range[0]
                    prob = category_prob.item()
                    action_outputs.append({
                        'category': 'constant',
                        'action': constant_value.item(),
                        'prob': prob
                    })
                elif category_name in self.action_heads:
                    action_head = self.action_heads[category_name]
                    action_logits = action_head(features)
                    action_probs = self.softmax(action_logits).squeeze(0)

                    for j, action_prob in enumerate(action_probs):
                        action = ACTION_MAP[category_name][j]
                        prob = category_prob.item() * action_prob.item()
                        action_outputs.append({
                            'category': category_name,
                            'action': action,
                            'prob': prob
                        })
                else:
                    print(f"未知的类别: {category_name}")
                    # 可以选择忽略或处理未知类别

        elif first_token == 'num':
            # 特殊处理 'num' 类别
            num_value = torch.relu(self.num_head(features)).long() % \
                        (self.num_range[1] - self.num_range[0] + 1) + self.num_range[0]
            prob = 1.0  # 假设只有一个可能的 num 值
            action_outputs.append({
                'category': 'num',
                'action': num_value.item(),
                'prob': prob
            })
        else:
            print(f"无法识别的第一个符号：{first_token}")
            return None

        return action_outputs  # 返回动作字典列表

def evaluate_policy_network(expression, policy_net, possible_actions):
    # 检查表达式是否已经完成
    if 'E' not in expression and 'num' not in expression:
        print(f"检测到终止状态: {expression}")
        return None  # 终止状态，返回 None

    try:
        # 创建词汇表
        token_map = create_token_map()
        tokens = parse_expression_to_tokens(expression, token_map)
        vocab_size = len(token_map) + 1  # 词汇表大小
        sequence_length = len(tokens)  # 序列长度
        batch_size = 1

        # 获取模型所在的设备
        device = next(policy_net.parameters()).device

        # 将解析的 token 序列转换为 tensor 格式并移动到设备
        src = torch.tensor([tokens], dtype=torch.long, device=device)  # (batch_size, sequence_length)

        # 生成键填充掩码
        src_key_padding_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)

        first_token = get_first_token(tokens, token_map)

        if first_token is None:
            print(f"无法识别的第一个符号，表达式: {expression}")
            return None

        # 调用传入的策略网络
        action_outputs = policy_net(
            src,
            src_key_padding_mask=src_key_padding_mask,
            first_token=first_token,
            possible_actions=possible_actions
        )

        return action_outputs

    except Exception as e:
        print(f"evaluate_policy_network 出现异常: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的异常堆栈信息
        return None

# 测试代码
if __name__ == "__main__":
    # 定义参数
    vocab_size = 100
    embed_dim = 64
    nhead = 8
    nhid = 128
    nlayers = 2
    dropout = 0.1

    # 初始化特征提取器
    feature_extractor = FeatureExtractor(
        vocab_size, embed_dim, nhead, nhid, nlayers, dropout
    )

    # 初始化策略网络
    policy_net = PolicyNetwork(
        feature_extractor=feature_extractor,
        embed_dim=embed_dim
    )

    # 将模型移动到 CUDA 设备（例如 cuda:0）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)

    expression = "Cov(1.1,1.1,num)"
    possible_actions = ['Vari', 'constant', 'unaryop', 'binaryop']
    action_outputs = evaluate_policy_network(expression, policy_net, possible_actions)
    print("Action Outputs:", action_outputs)
