import random

class AlphaEnv:
    def __init__(self, seed=None):
        # === 1) Operator categories (similar to your original writing), only split Sub1/Sub2, Div1/Div2, Pow1/Pow2 ===
        self.unary_operators = ['Log','Abs','Sign','CSRank'] 
        self.binary_operators = [
            'Add1', 'Add2', 
            'Sub1','Sub2','Sub3',
            'Mul1','Mul2',  
            'Div1','Div2','Div3',
            'Pow1','Pow2','Pow3',
            'Greater1', 'Greater2', 
            'Less1', 'Less2'
        ]
        self.rolling_operators = [
            'Ref','Skew','Kurt','Mean','Sum','Std','Var',
            'Max','Min','Med','Rank','Mad','Delta','WMA','EMA'
        ]
        self.paired_rolling_operators = ['Cov','Corr']
        
        # === 2) Feature list (Q->high/low/...) ===
        self.features = ['high','low','volume','open','close','vwap']

        # === 3) Constants/numeric ranges, same as your original ===
        self.constants = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
        self.num_range = [20, 30, 40]

        # === 4) Random seed ===
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # === 5) Initial state only contains one E (start symbol) ===
        self.state = ['Q']
        self.expansion_depth = 1

    def reset(self):
        """Reset environment: restore to only containing the start symbol E"""
        self.state = ['Q']
        self.expansion_depth = 1
        return self.state

    def get_possible_actions(self):
        """
        Determine available action categories based on the first placeholder (E/Q/num) in the current state (self.state).
        
        - If 'J' is encountered (e.g., "Add Q J"), available category: constant
        - If 'Q' is encountered, available categories: unaryop, binaryop, rollingop, pairedrollingop, Vari
        - If 'num' is encountered, available category: ['num'] (for replacing with specific integer)
        - If none, the expression is completed
        """
        for element in self.state:
            if 'J' in element:
                # If the string contains 'J' (e.g., "Add Q J"), can expand J
                return ['constant']
            
            elif 'Q' in element:
                # If contains 'Q', can expand Q
                return ['unaryop', 'binaryop', 'rollingop', 'pairedrollingop', 'Vari']
            
            elif 'num' in element:
                # If contains 'num', can replace with specific number
                return ['num']
        
        # If none of E/Q/num, expression is completed
        return []

    def apply_action(self, action_category, action):
        """
        Interact in the same way as your original: pass in (action_category, action),
        for example ('binaryop', 'Add') or ('rollingop', 'Ref') etc.
        """
        new_state, reward, done, info = self.step(action_category, action)
        return self, reward, done, info

    def step(self, action_category, action):
        new_state = []
        expanded = False

        # Check from left to right
        for element in self.state:
            if not expanded:
                pos_J = element.find('J') if 'J' in element else float('inf')
                pos_Q = element.find('Q') if 'Q' in element else float('inf')
                pos_num = element.find('num') if 'num' in element else float('inf')

                # Find the leftmost non-terminal symbol
                min_pos = min(pos_J, pos_Q, pos_num)

                if min_pos == float('inf'):
                    # Current element has no non-terminal
                    new_state.append(element)
                    continue

                # Replace based on the leftmost non-terminal
                if min_pos == pos_J:
                    symbol = 'J'
                elif min_pos == pos_Q:
                    symbol = 'Q'
                else:
                    symbol = 'num'

                expanded = True
                if symbol in ('J', 'Q'):
                    new_elements = self._expand_element(element, symbol=symbol,
                                                        action_category=action_category,
                                                        action=action)
                    new_state.extend(new_elements)
                else:  # num
                    replaced = element.replace('num', str(action), 1)
                    new_state.append(replaced)
                
                # Update expansion depth based on operator type
                if action_category == 'binaryop':
                    self.expansion_depth += 2  # Add more depth for binary operators
                elif action_category == 'rollingop':
                    self.expansion_depth += 2  # Keep adding 1 for other cases
                elif action_category == 'pairedrollingop':
                    self.expansion_depth += 3  # Add 1 for paired rolling operators
                elif action_category == 'unaryop':
                    self.expansion_depth += 1
            else:
                new_state.append(element)

        self.state = new_state

        done = all(('J' not in elem and 'Q' not in elem and 'num' not in elem) for elem in self.state)

        return self.state, 0, done, {'depth': self.expansion_depth}


    def _expand_element(self, element, symbol, action_category, action):
        """
        In the given element(string), replace the **first** occurrence of 'E' or 'Q' 
        with the corresponding production right-hand side.
        
        - symbol='J' means to execute "J -> ..." of a certain rule
        - symbol='Q' means "Q -> ..." of a certain rule
        """
        if symbol == 'J':
            expansion_str = self._expand_for_J(action_category, action)
            # Only replace the first 'J'
            replaced = element.replace('J', expansion_str, 1)
            return [replaced]

        elif symbol == 'Q':
            expansion_str = self._expand_for_Q(action_category, action)
            replaced = element.replace('Q', expansion_str, 1)
            return [replaced]
        
        # Normally won't reach here
        return [element]

    # ============== Expansion logic for different symbols (E/Q) ==============

    def _expand_for_J(self, action_category, action):
        """
        For the grammar of J, expand to constant
        """
        if action_category == 'constant':
            # J->constant(float)
            # Originally you directly use str(action) to replace
            return str(action)

        # Unknown category
        return f"{action} Q J"

    def _expand_for_Q(self, action_category, action):
        """
        For the grammar of Q, such as:
          Q->Abs(Q), Q->Add(Q,J), Q->Ref(Q,num), Q->Delta(Q,Num), Q->Cov(Q,Q,Num), Q->high, ...
        """
        if action_category == 'unaryop':
            # Q->Abs(Q), Q->Sign(Q), Q->Log(Q), Q->CSRank(Q)
            if action == 'Abs':
                return "Abs Q"
            elif action == 'Sign':
                return "Sign Q"
            elif action == 'Log':
                return "Log Abs Q"
            elif action == 'CSRank':
                return "CSRank Q"
            else:
                return f"{action} Q"

        elif action_category == 'binaryop':
            # Q->Add(Q,J), Q->Sub1(Q,J), Q->Sub2(J,Q), ...
            if action == 'Add1':
                return "Add1 Q Q"
            elif action == 'Mul1':
                return "Mul1 Q Q"
            elif action == 'Greater1':
                return "Greater1 Q Q"
            elif action == 'Less1':
                return "Less1 Q Q"
            elif action == 'Sub1':
                return "Sub1 Q Q"
            elif action == 'Div':
                return "Div1 Q Q"
            elif action == 'Pow1':
                return "Pow1 Q Q"
            
            if action == 'Add2':
                return "Add2 J Q"
            elif action == 'Mul2':
                return "Mul2 J Q"
            elif action == 'Greater2':
                return "Greater2 J Q"
            elif action == 'Less2':
                return "Less2 J Q"
            elif action == 'Sub2':
                return "Sub2 J Q"
            elif action == 'Div2':
                return "Div2 J Q"
            elif action == 'Pow2':
                return "Pow2 J Q"
            
            elif action == 'Sub3':
                return "Sub3 Q J"
            elif action == 'Div3':
                return "Div3 Q J"
            elif action == 'Pow3':
                return "Pow3 Q J"
          
            else:
                return f"{action} Q J"

        elif action_category == 'rollingop':
            # Q->Ref(Q,Num), Q->Mean(Q,Num), ..., Q->Delta(Q,Num)
            if action == 'Ref':
                return "Ref Q num"
            elif action == 'Mean':
                return "Mean Q num"
            elif action == 'Sum':
                return "Sum Q num"
            elif action == 'Std':
                return "Std Q num"
            elif action == 'Var':
                return "Var Q num"
            elif action == 'Skew':
                return "Skew Q num"
            elif action == 'Kurt':
                return "Kurt Q num"
            elif action == 'Max':
                return "Max Q num"
            elif action == 'Min':
                return "Min Q num"
            elif action == 'Med':
                return "Med Q num"
            elif action == 'Mad':
                return "Mad Q num"
            elif action == 'Rank':
                return "Rank Q num"
            elif action == 'Delta':
                return "Delta Q num"
            elif action == 'WMA':
                return "WMA Q num"
            elif action == 'EMA':
                return "EMA Q num"
            else:
                return f"{action} Q num"

        elif action_category == 'pairedrollingop':
            # Q->Cov(Q,Q,Num), Q->Corr(Q,Q,Num)
            if action == 'Cov':
                return "Cov Q Q num"
            elif action == 'Corr':
                return "Corr Q Q num"
            else:
                return f"{action} Q Q num"

        elif action_category == 'Vari':
            # Q-> high / low / ...
            return f"${action}"

        # Unknown category
        return f"{action} Q J"

    # ============== Other helper methods same as your original ==============

    def render(self):
        """Output the current expression"""
        print("Current expression:", ' '.join(self.state))

    def state_description(self):
        """Return the description of the current state"""
        return ' '.join(self.state)
    
    def is_terminal_state(self):
        """When there is no E/Q/num, it is a terminal state."""
        return all(
            ('J' not in elem) and 
            ('Q' not in elem) and
            ('num' not in elem)
            for elem in self.state
        )


# ===================== Test examples =====================
if __name__ == "__main__":
    env = AlphaEnv()
    env.reset()
    print("[Initialization] state:", env.state_description())

    # (1) Apply a binaryop to E: 'Add' => E->Add(Q,E)
    env.apply_action('binaryop', 'Add')
    print("[E->Add(Q,E)] state:", env.state_description())

    # (2) Now state = ["Add Q E"]
    #     First placeholder is Q => choose unary operator 'Log' => Q->Log(Q)
    env.apply_action('unaryop', 'Log')
    print("[Q->Log(Q)] state:", env.state_description())

    # (3) state = ["Add Log Q E"]
    #     First placeholder is Q => become 'close' => Q->close
    env.apply_action('Vari', 'close')
    print("[Q->close] state:", env.state_description())

    # (4) state = ["Add Log close E"]
    #     Now first placeholder is E => rollingop 'Mean' => E->Mean(Q,num)
    env.apply_action('rollingop', 'Mean')
    print("[E->Mean(Q,num)] state:", env.state_description())

    # (5) state = ["Add Log close Mean Q num"]
    #     First placeholder is Q => use 'Vari' => 'high'
    env.apply_action('Vari', 'high')
    print("[Q->high] state:", env.state_description())

    # (6) state = ["Add Log close Mean high num"]
    #     Last placeholder is 'num' => replace with 30
    env.apply_action('num', 30)
    print("[num->30] state:", env.state_description())

    # state = ["Add Log close Mean high 30"] without E/Q/num => done
    print("Expression completed:", env.state_description())
    print("is_terminal_state?", env.is_terminal_state())