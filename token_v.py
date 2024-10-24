import re

def create_token_map():
    token_map = {
        'Add': 1, 'Sub': 2, 'Mul': 3, 'Div': 4, 'Abs': 5, 'Sign': 6, 'Log': 7, 'CSRank': 8,
        'Greater': 9, 'Less': 10, 'Pow': 11, 'Ref': 12, 'Mean': 13, 'Sum': 14, 'Std': 15,
        'Var': 16, 'Skew': 17, 'Kurt': 18, 'Max': 19, 'Min': 20, 'Med': 21, 'Mad': 22,
        'Rank': 23, 'Delta': 24, 'WMA': 25, 'EMA': 26, 'Cov': 27, 'Corr': 28,
        '$high': 29, '$low': 30, '$volume': 31, '$open': 32, '$close': 33, '$vwap': 34,
        'E': 35, 'num': 36, 'constant': 37
    }
    return token_map

def parse_expression_to_tokens(expression, token_map):
    # 使用正则表达式拆分表达式，忽略括号和逗号
    tokens = re.findall(r'\$?\b\w+\b|\d+\.\d+|\d+', expression)
    token_sequence = []
    for token in tokens:
        if token in token_map:
            token_sequence.append(token_map[token])
        elif token == 'num':
            token_sequence.append(token_map['num'])
        elif re.match(r'\$[a-zA-Z]+', token):  # 变量，如 $high
            token_sequence.append(token_map.get('$Vari', token_map.get('Vari', 0)))
        elif re.match(r'\d+(\.\d+)?', token):  # 数字
            # Only mark 'num' if explicitly marked as such
            continue
        else:
            print(f"未识别的标记: {token}")
            token_sequence.append(0)  # 使用0表示未定义的标记
    return token_sequence



def get_first_token(expression_tokens, token_map):
    inverse_token_map = {v: k for k, v in token_map.items()}
    has_E_or_num = False  # Flag to indicate the presence of 'E' or 'num'

    for token_id in expression_tokens:
        token = inverse_token_map.get(token_id)
        if token == 'E':
            return 'E'  # Return 'E' immediately if found
        elif token == 'num':
            has_E_or_num = True  # Set flag if 'num' is found

    return 'num' if has_E_or_num else None  # Return 'num' only if found, otherwise None

import unittest
class TestSpecificExpressions(unittest.TestCase):
    def setUp(self):
        self.token_map = create_token_map()

    def test_expression_log_e(self):
        # 测试表达式 "Log E"
        expression = "Log E"
        tokens = parse_expression_to_tokens(expression, self.token_map)
        first_token = get_first_token(tokens, self.token_map)
        self.assertEqual(first_token, 'E', f"测试失败：'{expression}' 应该返回 'E'，而返回了 {first_token}")

    def test_expression_low(self):
        # 测试表达式 "$low"
        expression = "$low"
        tokens = parse_expression_to_tokens(expression, self.token_map)
        first_token = get_first_token(tokens, self.token_map)
        self.assertIsNone(first_token, f"测试失败：'{expression}' 应该返回 None，而返回了 {first_token}")

# 运行测试
if __name__ == '__main__':
    unittest.main()