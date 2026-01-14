import re

def create_token_map():
    token_map = {}

    # ========== 1) Unary operators ========== 
    token_map['Log'] = 1
    token_map['Abs'] = 2
    token_map['Sign'] = 3
    token_map['CSRank'] = 4

    # ========== 2) Binary operators ========== 
    token_map['Add'] = 5
    token_map['Sub'] = 6
    token_map['Mul'] = 7
    token_map['Div'] = 8
    token_map['Pow'] = 9
    token_map['Greater'] = 10
    token_map['Less'] = 11

    # ========== 3) Rolling operators ========== 
    token_map['Ref'] = 12
    token_map['Skew'] = 13
    token_map['Kurt'] = 14
    token_map['Mean'] = 15
    token_map['Sum'] = 16
    token_map['Std'] = 17
    token_map['Var'] = 18
    token_map['Max'] = 19
    token_map['Min'] = 20
    token_map['Med'] = 21
    token_map['Rank'] = 22
    token_map['Mad'] = 23
    token_map['Delta'] = 24
    token_map['WMA'] = 25
    token_map['EMA'] = 26  # Added EMA

    # ========== 4) Paired rolling operators ========== 
    token_map['Cov'] = 27
    token_map['Corr'] = 28

    # ========== 5) Features ========== 
    token_map['$high'] = 29
    token_map['$low'] = 30
    token_map['$volume'] = 31
    token_map['$open'] = 32
    token_map['$close'] = 33
    token_map['$vwap'] = 34

    # ========== 6) Special tokens ========== 
    token_map['Q'] = 35         # Non-terminal symbol Q
    token_map['num'] = 36       # Used for placeholder numbers

    # ========== 7) Constants ========== 
    token_map['-0.1'] = 37
    token_map['-0.05'] = 38
    token_map['-0.01'] = 39
    token_map['0.01'] = 40
    token_map['0.05'] = 41
    token_map['0.1'] = 42

    # ========== 8) Numbers (num_range) ========== 
    token_map['40'] = 43
    token_map['30'] = 44
    token_map['20'] = 45

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
        if token == 'Q':
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
        print(f"【Test instance {i}】")
        print("Expression:", expr)
        print("Token sequence:", tokens)
        print("First placeholder:", first_token)
        print("-" * 40)