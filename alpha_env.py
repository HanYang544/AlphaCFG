import random

class AlphaEnv:
    def __init__(self, seed=None):
        # 各类别动作定义
        self.unary_operators = ['Abs', 'Sign', 'Log', 'CSRank']
        self.binary_operators = ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Pow']
        self.rolling_operators = ['Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min', 'Med', 'Mad',
                                'Rank', 'Delta', 'WMA', 'EMA']
        self.paired_rolling_operators = ['Cov', 'Corr']
        self.features = ['high', 'low', 'volume', 'open', 'close', 'vwap']
        self.constants = [random.uniform(1, 100) for _ in range(10)]
        self.num_range = (1, 30)  # num 范围
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.state = ['E']  # 初始状态
        self.expansion_depth = 0

    def reset(self):
        """重置环境"""
        self.state = ['E']
        self.expansion_depth = 0
        return self.state

    def get_possible_actions(self):
        """根据当前状态返回可能的动作类别"""
        for element in self.state:
            if 'E' in element:
                # 如果存在未扩展的 E，返回可扩展的动作类别
                return ['Vari', 'constant', 'unaryop', 'binaryop', 'rollingop', 'pairedrollingop']
            elif 'num' in element:
                # 如果存在未扩展的 num，返回 num 类别
                return ['num']
        return []

    def apply_action(self, action_category, action):
        """根据类别和动作修改当前状态"""
        new_state, reward, done, info = self.step(action_category, action)
        return self, reward, done, info  # 返回更新后的环境对象

    def step(self, action_category, action):
        """根据动作类别和具体动作扩展"""
        new_state = []
        expanded = False

        # 左到右处理扩展
        for element in self.state:
            if not expanded:
                if 'E' in element:
                    # 如果遇到 E，扩展为对应的动作
                    expanded = True
                    new_elements = self.expand_element(element, action_category, action)
                    new_state.extend(new_elements)
                    self.expansion_depth += 1
                    continue
                elif 'num' in element:
                    # 如果遇到 num，扩展为具体数字
                    expanded = True
                    new_element = element.replace('num', str(action), 1)
                    new_state.append(new_element)
                    self.expansion_depth += 1
                    continue
            # 未处理的元素直接添加
            new_state.append(element)

        self.state = new_state

        # 判断表达式是否完成（没有 E 和 num）
        done = all('E' not in elem and 'num' not in elem for elem in self.state)

        return self.state, 0, done, {'depth': self.expansion_depth}

    def is_terminal_state(self):
        """判断当前表达式是否为终止状态"""
        return all('E' not in elem and 'num' not in elem for elem in self.state)

    def expand_element(self, element, action_category, action):
        """根据类别和动作扩展 E 或 num"""
        if action_category == 'unaryop':
            # 一元操作符，格式：操作符 E
            return [element.replace('E', f"{action} E", 1)]
        elif action_category == 'binaryop':
            # 二元操作符，格式：操作符 E E
            return [element.replace('E', f"{action} E E", 1)]
        elif action_category == 'rollingop':
            # 滚动操作符，格式：操作符 E num
            return [element.replace('E', f"{action} E num", 1)]
        elif action_category == 'pairedrollingop':
            # 成对滚动操作符，格式：操作符 E E num
            return [element.replace('E', f"{action} E E num", 1)]
        elif action_category == 'Vari':
            # 特征变量，格式：$变量名
            return [element.replace('E', f"${action}", 1)]
        elif action_category == 'constant':
            # 常量，直接替换 E
            return [element.replace('E', str(action), 1)]
        elif action_category == 'num':
            # 替换 num
            return [element.replace('num', str(action), 1)]
        else:
            # 未知的动作类别，忽略
            return [element]

    def render(self):
        """输出当前表达式"""
        print("当前表达式：", ' '.join(self.state))

    def state_description(self):
        """返回当前状态的描述"""
        return ' '.join(self.state)

# 测试代码
if __name__ == "__main__":
    env = AlphaEnv()
    env.reset()
    env.apply_action('binaryop', 'Add')
    env.render()
    env.apply_action('Vari', 'high')
    env.render()
    env.apply_action('Vari', 'low')
    env.render()
