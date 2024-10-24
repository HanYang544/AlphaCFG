# value_.py

import torch
import torch.nn as nn
from token_v import create_token_map, parse_expression_to_tokens
from feature_ import FeatureExtractor

class ValueNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim):
        super(ValueNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()  # 将输出映射到 (-1, 1)
        )

    def forward(self, src, src_key_padding_mask=None):
        features = self.feature_extractor(src, src_key_padding_mask)
        value = self.value_head(features)
        return value

def evaluate_expression_value(expression, value_net):
    # 创建词汇表
    token_map = create_token_map()

    # 解析表达式为token序列
    tokens = parse_expression_to_tokens(expression, token_map)

    # 获取模型所在的设备
    device = next(value_net.parameters()).device

    # 定义参数
    vocab_size = len(token_map) + 1  # 词汇表大小，包括0作为未定义的token
    sequence_length = len(tokens)  # 序列长度与表达式中token数量一致
    batch_size = 1  # 单一表达式作为输入

    # 将解析的token序列转换为tensor格式并移动到设备
    src = torch.tensor([tokens], dtype=torch.long, device=device)  # (batch_size, sequence_length)

    # 生成键填充掩码，确保它的类型为布尔类型并移动到设备
    src_key_padding_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)

    # 使用传入的价值网络获取输出
    value_output = value_net(src, src_key_padding_mask=src_key_padding_mask)

    return value_output.item()

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

    # 初始化价值网络
    value_net = ValueNetwork(
        feature_extractor=feature_extractor,
        embed_dim=embed_dim
    )

    # 将模型移动到 CUDA 设备（例如 cuda:0）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value_net.to(device)

    expression = "Add($high, $low)"
    value = evaluate_expression_value(expression, value_net)
    print("Expression Value:", value)





