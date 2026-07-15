import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph

# Step 1: Define vocabulary and operator parameter counts
class TokenMap:
    def __init__(self):
        self.token_map = {
            # ========== 1) Unary operators ========== 
            'Log': 1,
            'Abs': 2,
            'Sign': 3,
            'CSRank': 4,

            # ========== 2) Binary operators ========== 
            'Add1': 5,
            'Add2': 6,
            'Sub1': 7,
            'Sub2': 8,
            'Sub3': 9,
            'Mul1': 10,
            'Mul2': 11,
            'Div1': 12,
            'Div2': 13,
            'Div3': 14,
            'Pow1': 15,
            'Pow2': 16,
            'Pow3': 17,
            'Greater1': 18,
            'Greater2': 19,
            'Less1': 20,
            'Less2': 21,

            # ========== 3) Rolling operators ========== 
            'Ref': 22,
            'Skew': 23,
            'Kurt': 24,
            'Mean': 25,
            'Sum': 26,
            'Std': 27,
            'Var': 28,
            'Max': 29,
            'Min': 30,
            'Med': 31,
            'Rank': 32,
            'Mad': 33,
            'Delta': 34,
            'WMA': 35,
            'EMA': 36,  # Added EMA

            # ========== 4) Paired rolling operators ========== 
            'Cov': 37,
            'Corr': 38,

            # ========== 5) Features ========== 
            '$high': 39,
            '$low': 40,
            '$volume': 41,
            '$open': 42,
            '$close': 43,
            '$vwap': 44,

            # ========== 6) Special tokens ========== 
            'J': 45,         # Non-terminal symbol E
            'Q': 46,         # Non-terminal symbol Q
            'num': 47,       # Used for placeholder numbers (num_range)

            # ========== 7) Constants ========== 
            '-0.1': 48,
            '-0.05': 49,
            '-0.01': 50,
            '0.01': 51,
            '0.05': 52,
            '0.1': 53,

            # ========== 8) Numbers (num_range) ========== 
            '40': 54,
            '30': 55,
            '20': 56,
            'Greater3': 57,
            'Less3': 58,
        }

        # ========== Operator arity (number of parameters) ========== 
        self.operator_arities = {
            # Unary operators
            'Log': 1,
            'Abs': 1,
            'Sign': 1,
            'CSRank': 1,

            # Binary operators
            'Add1': 2,
            'Add2': 2,
            'Sub1': 2,
            'Sub2': 2,
            'Sub3': 2,
            'Mul1': 2,
            'Mul2': 2,
            'Div1': 2,
            'Div2': 2,
            'Div3': 2,
            'Pow1': 2,
            'Pow2': 2,
            'Pow3': 2,
            'Greater1': 2,
            'Greater2': 2,
            'Greater3': 2,
            'Less1': 2,
            'Less2': 2,
            'Less3': 2,

            # Rolling operators
            'Ref': 2,
            'Skew': 2,
            'Kurt': 2,
            'Mean': 2,
            'Sum': 2,
            'Std': 2,
            'Var': 2,
            'Max': 2,
            'Min': 2,
            'Med': 2,
            'Rank': 2,
            'Mad': 2,
            'Delta': 2,
            'WMA': 2,
            'EMA': 2,

            # Paired rolling operators
            'Cov': 3,
            'Corr': 3
        }

    def get_word_id(self, token):
        """
        Return the ID of the token, defaulting to 0.
        """
        return self.token_map.get(token, 0)

    def get_arity(self, operator):
        """
        Return the number of parameters for the operator, defaulting to 0 (undefined arity).
        """
        return self.operator_arities.get(operator, 0)


# Step 2: Define expression tree node
class ExpressionTree:
    def __init__(self, value=None, children=None, is_operator=False):
        """
        Initialize an expression tree node
        :param value: The value of the node, e.g., 'Add', '$high', '0.05', etc.
        :param children: List of child nodes
        :param is_operator: Whether the node is an operator node
        """
        self.value = value
        self.children = children if children is not None else []
        self.is_operator = is_operator

    def is_leaf(self):
        """Check if the node is a leaf node"""
        return len(self.children) == 0

    def __repr__(self):
        """Return the string representation of the node"""
        if self.is_leaf():
            return f"Leaf({self.value})"
        else:
            children_repr = ", ".join([str(child) for child in self.children])
            return f"({self.value}({children_repr}))"


# Step 3: Implement expression parser
def parse_expression(expression_str, token_map):
    """
    Parse an expression string without parentheses and build a syntax tree
    :param expression_str: Expression string, e.g., 'Rank Corr $high 0.05 40 40'
    :param token_map: TokenMap object
    :return: ExpressionTree object
    """
    tokens = expression_str.strip().split()
    tokens = tokens[::-1]  # Reverse the list for efficient popping from the end

    def helper():
        if not tokens:
            return None
        token = tokens.pop()
        if token in token_map.operator_arities:
            arity = token_map.get_arity(token)
            children = []
            for _ in range(arity):
                child = helper()
                if child is not None:
                    children.append(child)
            return ExpressionTree(value=token, children=children, is_operator=True)
        else:
            # Variable or constant/number
            return ExpressionTree(value=token, is_operator=False)

    tree = helper()
    return tree


# Step 4: Convert syntax tree to DGL graph
def tree_to_dgl_graph(tree):
    """
    Convert an expression tree to a DGL graph.
    :param tree: ExpressionTree object
    :return: DGL graph object and list of nodes
    """
    nodes = []
    edges_src = []
    edges_dst = []

    def traverse(node):
        """
        Recursively traverse the tree and build nodes and edges.
        :param node: Current node
        :return: Index of the current node
        """
        idx = len(nodes)
        nodes.append(node)
        for child in node.children:
            child_idx = traverse(child)
            edges_src.append(child_idx)  # Child node
            edges_dst.append(idx)        # Parent node
        return idx

    root_idx = traverse(tree)
    # Create graph using the latest DGL API
    src = torch.tensor(edges_src, dtype=torch.int64)
    dst = torch.tensor(edges_dst, dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=len(nodes))
    return g, nodes, root_idx


# Step 5: Define TreeLSTMCell
class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, max_children=3):
        """
        Initialize TreeLSTM cell
        :param x_size: Input feature dimension
        :param h_size: Hidden state dimension
        :param max_children: Maximum number of child nodes (default is 3)
        """
        super(TreeLSTMCell, self).__init__()
        self.h_size = h_size
        self.max_children = max_children

        # Weights for input, output, and update gates
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=True)
        self.U_iou = nn.Linear(max_children * h_size, 3 * h_size, bias=False)

        # Weights for forget gate
        self.U_f = nn.Linear(max_children * h_size, max_children * h_size, bias=False)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weight parameters"""
        nn.init.xavier_uniform_(self.W_iou.weight)
        nn.init.xavier_uniform_(self.U_iou.weight)
        nn.init.xavier_uniform_(self.U_f.weight)
        self.W_iou.bias.data.fill_(0)

    def message_func(self, edges):
        """
        Message passing function, passing hidden states and cell states of child nodes
        :param edges: DGL edges
        :return: Dictionary containing child nodes' hidden states 'h' and cell states 'c'
        """
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_child = nodes.mailbox['h']  # (batch_size, num_children, h_size)
        c_child = nodes.mailbox['c']  # (batch_size, num_children, h_size)
        num_children = h_child.shape[1]
        batch_size = h_child.shape[0]

        h_cat = h_child.reshape(batch_size, -1)  # default: concat all
        c_padded = c_child

        # Identify the operator token id of the current node
        op_ids = nodes.data['op']  # shape: (batch_size,)

        # Define special aggregation operator id sets
        mean_ops = {5, 10}  # Add1, Mul1
        cov_corr_ops = {37, 38}     # Cov, Corr

        h_custom = []
        c_custom = []

        for i in range(batch_size):
            op_id = op_ids[i].item()
            h_i = h_child[i]  # (num_children, h_size)
            c_i = c_child[i]

            if op_id in mean_ops:
                h_i_agg = h_i.mean(dim=0).unsqueeze(0)  # (1, h_size)
                c_i_agg = c_i.mean(dim=0).unsqueeze(0)
            elif op_id in cov_corr_ops and h_i.shape[0] == 3:
                # Average the first two child nodes
                h_pair = h_i[:2].mean(dim=0)  # (h_size,)
                c_pair = c_i[:2].mean(dim=0)
                # The third child node
                h_last = h_i[2]
                c_last = c_i[2]
                # Concatenate
                h_i_agg = torch.cat([h_pair.unsqueeze(0), h_last.unsqueeze(0)], dim=0)
                c_i_agg = torch.cat([c_pair.unsqueeze(0), c_last.unsqueeze(0)], dim=0)
            else:
                h_i_agg = h_i
                c_i_agg = c_i

            # Pad to max_children
            pad_len = self.max_children - h_i_agg.shape[0]
            if pad_len > 0:
                h_pad = torch.zeros(pad_len, self.h_size, device=h_i.device)
                c_pad = torch.zeros(pad_len, self.h_size, device=c_i.device)
                h_i_agg = torch.cat([h_i_agg, h_pad], dim=0)
                c_i_agg = torch.cat([c_i_agg, c_pad], dim=0)

            h_custom.append(h_i_agg.reshape(-1))  # flatten
            c_custom.append(c_i_agg)

        h_cat = torch.stack(h_custom, dim=0)  # (batch_size, max_children * h_size)
        c_padded = torch.stack(c_custom, dim=0)  # (batch_size, max_children, h_size)

        # Forget gate and update
        f = torch.sigmoid(self.U_f(h_cat))
        f = f.view(batch_size, self.max_children, self.h_size)
        c = torch.sum(f * c_padded, dim=1)
        iou_child = self.U_iou(h_cat)

        return {'iou_child': iou_child, 'c': c}

    def apply_node_func(self, nodes):
        """
        Apply function, calculate input, output, and update gates, and update hidden and cell states
        :param nodes: DGL nodes
        :return: Dictionary containing updated 'h' and 'c'
        """
        # If the node data doesn't have 'iou_child', it's a leaf node, set default to 0
        if 'iou_child' in nodes.data:
            iou_child = nodes.data['iou_child']
        else:
            # Construct a zero tensor based on x's shape, shape: (batch_size, 3 * h_size)
            x = nodes.data['x']
            iou_child = torch.zeros(x.size(0), 3 * self.h_size, device=x.device)

        # Get the node's embedding vector x
        x = nodes.data['x']  # (batch_size, x_size)

        # Calculate W_iou(x)
        iou_node = self.W_iou(x)  # (batch_size, 3 * h_size)

        # Calculate final iou
        iou = iou_node + iou_child  # (batch_size, 3 * h_size)

        # Split i, o, u
        i, o, u = torch.chunk(iou, 3, dim=1)  # (batch_size, h_size) each

        # Calculate gates
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # Update cell state and hidden state
        c = i * u + nodes.data['c']          # (batch_size, h_size)
        h = o * torch.tanh(c)                # (batch_size, h_size)

        return {'h': h, 'c': c}


# Step 6: Define TreeLSTM model
class TreeLSTMModel(nn.Module):
    def __init__(self, num_vocabs, x_size, h_size, dropout=0.5, pretrained_emb=None, max_children=3):
        """
        Initialize TreeLSTM model
        :param num_vocabs: Vocabulary size
        :param x_size: Input embedding dimension
        :param h_size: Hidden state dimension
        :param dropout: Dropout probability
        :param pretrained_emb: Pre-trained embeddings (optional)
        :param max_children: Maximum number of child nodes
        """
        super(TreeLSTMModel, self).__init__()
        self.h_size = h_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using pretrained embeddings')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(x_size, h_size, max_children=max_children)

    def forward(self, graph, wordid, mask, h, c):
        """
        Forward pass
        :param graph: DGL graph
        :param wordid: Vocabulary IDs of nodes
        :param mask: Node mask (not used currently)
        :param h: Initial hidden state
        :param c: Initial cell state
        :return: h_out (hidden state feature vector)
        """
        # Embedding layer
        embeds = self.embedding(wordid)  # (num_nodes, x_size)
        embeds = self.dropout(embeds)

        # Initialize nodes' h and c
        graph.ndata['h'] = h
        graph.ndata['c'] = c

        # Set 'x' as the embedding vector
        graph.ndata['x'] = embeds
        graph.ndata['op'] = wordid  # Store the token id here

        # Propagate TreeLSTM
        dgl.prop_nodes_topo(
            graph,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func
        )


        # Get hidden state
        h_out = self.dropout(graph.ndata['h'])
        return h_out


# ========== Aligned FeatureExtractor ==========
class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim=256, h_size=256, dropout=0.5):
        super().__init__()
        self.token_map = TokenMap()
        self.model = TreeLSTMModel(
            num_vocabs=max(self.token_map.token_map.values()) + 1,
            x_size=embed_dim,
            h_size=h_size,
            dropout=dropout
        )
        self.fc = nn.Linear(h_size, h_size)

    def forward(self, expression: str) -> torch.Tensor:
        # Expression -> Tree -> DGL graph
        tree = parse_expression(expression, self.token_map)
        graph, nodes, root_idx = tree_to_dgl_graph(tree)
        graph = graph.to(next(self.parameters()).device)

        # Node features
        wordid = torch.tensor(
            [self.token_map.get_word_id(node.value) for node in nodes],
            dtype=torch.long
        ).to(graph.device)

        # Forward pass
        h = torch.zeros(graph.number_of_nodes(), self.model.h_size).to(graph.device)
        c = torch.zeros(graph.number_of_nodes(), self.model.h_size).to(graph.device)
        feature = self.model(graph, wordid, None, h, c)
        root_h = feature[root_idx].unsqueeze(0)  # [1, h_size]
        return self.fc(root_h)
