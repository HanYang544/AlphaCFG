import torch
from stock_data import *
from calculator__ import *
from expression import *


# expression.py

def parse_expression_from_string(expression_str: str) -> Expression:
    """
    解析前缀表示法的表达式字符串，返回 Expression 对象。

    参数:
    - expression_str: str, 前缀表示法的表达式，例如 "Add $high $low"

    返回:
    - Expression 对象
    """
    tokens = expression_str.split()
    tokens = tokens[::-1]  # 反转列表，方便使用 pop() 从末尾获取

    def parse() -> Expression:
        if not tokens:
            raise ValueError("无效的表达式：缺少令牌。")
        token = tokens.pop()

        # 二元操作符
        if token in ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Pow']:
            left = parse()
            right = parse()
            operator_map = {
                'Add': Add,
                'Sub': Sub,
                'Mul': Mul,
                'Div': Div,
                'Greater': Greater,
                'Less': Less,
                'Pow': Pow
            }
            return operator_map[token](left, right)

        # 一元操作符
        elif token in ['Abs', 'Sign', 'Log', 'CSRank']:
            operand = parse()
            operator_map = {
                'Abs': Abs,
                'Sign': Sign,
                'Log': Log,
                'CSRank': CSRank
            }
            return operator_map[token](operand)

        # 滚动操作符
        elif token in ['Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min', 'Med', 'Mad', 'Rank', 'Delta',
                       'WMA', 'EMA']:
            operand = parse()
            delta_time_token = tokens.pop()
            try:
                delta_time = int(delta_time_token)
            except ValueError:
                raise ValueError(f"无效的 delta_time 值: {delta_time_token}")
            operator_map = {
                'Ref': Ref,
                'Mean': Mean,
                'Sum': Sum,
                'Std': Std,
                'Var': Var,
                'Skew': Skew,
                'Kurt': Kurt,
                'Max': Max,
                'Min': Min,
                'Med': Med,
                'Mad': Mad,
                'Rank': Rank,
                'Delta': Delta,
                'WMA': WMA,
                'EMA': EMA
            }
            return operator_map[token](operand, delta_time)

        # 成对滚动操作符
        elif token in ['Cov', 'Corr']:
            lhs = parse()
            rhs = parse()
            delta_time_token = tokens.pop()
            try:
                delta_time = int(delta_time_token)
            except ValueError:
                raise ValueError(f"无效的 delta_time 值: {delta_time_token}")
            operator_map = {
                'Cov': Cov,
                'Corr': Corr
            }
            return operator_map[token](lhs, rhs, delta_time)

        # 特征变量
        elif token.startswith('$'):
            feature_name = token[1:].upper()  # 假设 FeatureType 的名称为大写
            try:
                feature = FeatureType[feature_name]
            except KeyError:
                raise ValueError(f"未知的特征变量: {feature_name}")
            return Feature(feature)

        # 常量
        else:
            try:
                value = float(token)
                return Constant(value)
            except ValueError:
                raise ValueError(f"未知的令牌或无效的常量: {token}")

    expression = parse()
    if tokens:
        raise ValueError("无效的表达式：存在多余的令牌。")
    return expression


def evaluate_expression_ic(expr: Expression) -> float:
    """
    计算给定表达式的 IC 值。

    参数:
    - expr: Expression 对象，表示需要评估的因子表达式

    返回:
    - IC 值（float）
    """
    # 初始化 StockData，使用一个更长时间范围
    stock_data = StockData(
        instrument="csi100",
        start_time="2015-01-01",  # 扩大时间范围
        end_time="2023-01-01",
        features=[FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH, FeatureType.LOW, FeatureType.VOLUME, FeatureType.VWAP],
        device=torch.device('cpu'),
        max_backtrack_days=600,
        max_future_days=30
    )

    # 定义目标因子
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    # 使用 QLibStockDataCalculator 计算表达式的 IC
    calculator = QLibStockDataCalculator(data=stock_data, target=target)

    # 打印表达式和目标的值（可选）
    value_expr = expr.evaluate(stock_data)
    value_target = target.evaluate(stock_data)

    # print("Expression value (before IC calculation):")
    # print(value_expr)

    # print("Target value (before IC calculation):")
    # print(value_target)

    # 计算 IC 值
    ic_value = calculator.calc_single_IC_ret(expr)

    # 输出 IC 值
    # print(f"The IC value for the expression is: {ic_value:.5f}")

    return ic_value

# 示例：使用封装的函数计算表达式的 IC 值
volume_feature = Feature(FeatureType.VOLUME)
mean_volume = Mean(volume_feature, 20)
expression = Add(
    Mul(Constant(1.0), mean_volume),  # 使用 1.0 常量，避免过大的数值
    Constant(-5.0)
)

# 调用函数，计算 IC
ic_result = evaluate_expression_ic(expression)
# print(f"Final IC result: {ic_result:.5f}")




