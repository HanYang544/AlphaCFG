import re

def create_token_map():
    token_map = {}

    # ========== 1) Unary operators ========== 
    token_map['Log'] = 1
    token_map['Abs'] = 2
    token_map['Sign'] = 3
    token_map['CSRank'] = 4

    # ========== 2) Binary operators ========== 
    token_map['Add1'] = 5
    token_map['Add2'] = 6
    token_map['Sub1'] = 7
    token_map['Sub2'] = 8
    token_map['Sub3'] = 9
    token_map['Mul1'] = 10
    token_map['Mul2'] = 11
    token_map['Div1'] = 12
    token_map['Div2'] = 13
    token_map['Div3'] = 14
    token_map['Pow1'] = 15
    token_map['Pow2'] = 16
    token_map['Pow3'] = 17
    token_map['Greater1'] = 18
    token_map['Greater2'] = 19
    token_map['Less1'] = 20
    token_map['Less2'] = 21

    # ========== 3) Rolling operators ========== 
    token_map['Ref'] = 22
    token_map['Skew'] = 23
    token_map['Kurt'] = 24
    token_map['Mean'] = 25
    token_map['Sum'] = 26
    token_map['Std'] = 27
    token_map['Var'] = 28
    token_map['Max'] = 29
    token_map['Min'] = 30
    token_map['Med'] = 31
    token_map['Rank'] = 32
    token_map['Mad'] = 33
    token_map['Delta'] = 34
    token_map['WMA'] = 35
    token_map['EMA'] = 36  # Added EMA

    # ========== 4) Paired rolling operators ========== 
    token_map['Cov'] = 37
    token_map['Corr'] = 38

    # ========== 5) Features ========== 
    token_map['$high'] = 39
    token_map['$low'] = 40
    token_map['$volume'] = 41
    token_map['$open'] = 42
    token_map['$close'] = 43
    token_map['$vwap'] = 44

    # ========== 6) Special tokens ========== 
    token_map['J'] = 45         # Non-terminal symbol E
    token_map['Q'] = 46         # Non-terminal symbol Q
    token_map['num'] = 47       # Placeholder for numbers

    # ========== 7) Constants ========== 
    token_map['-0.1'] = 48
    token_map['-0.05'] = 49
    token_map['-0.01'] = 50
    token_map['0.01'] = 51
    token_map['0.05'] = 52
    token_map['0.1'] = 53

    # ========== 8) Numbers (num_range) ========== 
    token_map['40'] = 54
    token_map['30'] = 55
    token_map['20'] = 56
    token_map['Greater3'] = 57
    token_map['Less3'] = 58

    return token_map



def parse_expression_to_tokens(expression, token_map):
    # Updated regex to capture negative floats and predefined nums
    tokens = re.findall(r'[-]?\d+\.\d+|\$?\b\w+\b|[-]?\d+', expression)
    token_sequence = []
    for token in tokens:
        if token in token_map:
            token_sequence.append(token_map[token])
        elif re.match(r'\$[a-zA-Z]+', token):  # Variable, e.g., $high
            token_sequence.append(token_map.get(token, token_map.get('Vari', 0)))
        elif re.match(r'[-]?\d+(\.\d+)?', token):  # Numbers, including predefined
            # Check if the number is in the predefined constants or num
            if token in token_map:
                token_sequence.append(token_map[token])
            else:
                print(f"Unrecognized number: {token}")
                token_sequence.append(token_map['constant'])
        else:
            print(f"Unrecognized token: {token}")
            token_sequence.append(0)  # Use 0 for undefined tokens
    return token_sequence


def get_first_token(expression_tokens, token_map):
    inverse_token_map = {v: k for k, v in token_map.items()}
    for token_id in expression_tokens:
        token = inverse_token_map.get(token_id)
        if token == 'J':
            return 'J'
        elif token == 'Q':
            return 'Q'
        elif token == 'num':
            return 'num'
    return None


# ===== Test examples =====
if __name__ == "__main__":
    token_map = create_token_map()

    test_expressions = [
        "Add close Log Mean high 30",
        "Sub1 open Abs E",
        "Ref volume Delta E 20",
        "Corr low Var Q",
        "Pow1 close CSRank Q"
    ]

    for i, expr in enumerate(test_expressions, 1):
        tokens = parse_expression_to_tokens(expr, token_map)
        first_token = get_first_token(tokens, token_map)
        print(f"【Test Example {i}】")
        print("Expression:", expr)
        print("Token sequence:", tokens)
        print("First placeholder:", first_token)
        print("-" * 40)
