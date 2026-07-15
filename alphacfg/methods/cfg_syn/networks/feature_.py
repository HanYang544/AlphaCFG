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
            'Add': 5,
            'Sub': 6,
            'Mul': 7,
            'Div': 8,
            'Pow': 9,
            'Greater': 10,
            'Less': 11,

            # ========== 3) Rolling operators ========== 
            'Ref': 12,
            'Skew': 13,
            'Kurt': 14,
            'Mean': 15,
            'Sum': 16,
            'Std': 17,
            'Var': 18,
            'Max': 19,
            'Min': 20,
            'Med': 21,
            'Rank': 22,
            'Mad': 23,
            'Delta': 24,
            'WMA': 25,
            'EMA': 26,  # Added EMA

            # ========== 4) Paired rolling operators ========== 
            'Cov': 27,
            'Corr': 28,

            # ========== 5) Features ========== 
            '$high': 29,
            '$low': 30,
            '$volume': 31,
            '$open': 32,
            '$close': 33,
            '$vwap': 34,

            # ========== 6) Special tokens ========== 
            'J': 35,         # Non-terminal symbol E
            'Q': 36,         # Non-terminal symbol Q
            'num': 37,       # Used for placeholder numbers (num_range)

            # ========== 7) Constants ========== 
            '-0.1': 38,
            '-0.05': 39,
            '-0.01': 40,
            '0.01': 41,
            '0.05': 42,
            '0.1': 43,

            # ========== 8) Numbers (num_range) ========== 
            '40': 44,
            '30': 45,
            '20': 46
        }

        # ========== Operator parameter counts (arity) ========== 
        self.operator_arities = {
            # Unary operators
            'Log': 1,
            'Abs': 1,
            'Sign': 1,
            'CSRank': 1,

            # Binary operators
            'Add': 2,
            'Sub': 2,
            'Mul': 2,
            'Div': 2,
            'Pow': 2,
            'Greater': 2,
            'Less': 2,

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
        Return the ID of the token, default is 0.
        """
        return self.token_map.get(token, 0)

    def get_arity(self, operator):
        """
        Return the parameter count of the operator, default is 0 (undefined arity).
        """
        return self.operator_arities.get(operator, 0)



# Step 2: Define expression tree node
class ExpressionTree:
    def __init__(self, value=None, children=None, is_operator=False):
        """
        Initialize expression tree node
        :param value: The value of the node, e.g., 'Add', '$high', '0.05' etc.
        :param children: List of child nodes
        :param is_operator: Whether it is an operator node
        """
        self.value = value
        self.children = children if children is not None else []
        self.is_operator = is_operator

    def is_leaf(self):
        """Check if it is a leaf node"""
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
        Recursively traverse the tree to build nodes and edges.
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

        # Weights for input gate, output gate, and update gate
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

        h_custom = []
        c_custom = []

        for i in range(batch_size):
            op_id = op_ids[i].item()
            h_i = h_child[i]  # (num_children, h_size)
            c_i = c_child[i]

            
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
        Apply function, calculate input gate, output gate, and update gate, and update hidden state and cell state
        :param nodes: DGL nodes
        :return: Dictionary containing updated 'h' and 'c'
        """
        # If the node data doesn't have 'iou_child', it means the node is a leaf node, set default value to 0
        if 'iou_child' in nodes.data:
            iou_child = nodes.data['iou_child']
        else:
            # Here construct a zero tensor based on the shape of x, shape: (batch_size, 3 * h_size)
            x = nodes.data['x']
            iou_child = torch.zeros(x.size(0), 3 * self.h_size, device=x.device)

        # Get the embedding vector x of the node
        x = nodes.data['x']  # (batch_size, x_size)

        # Calculate W_iou(x)
        iou_node = self.W_iou(x)  # (batch_size, 3 * h_size)

        # Calculate the final iou
        iou = iou_node + iou_child  # (batch_size, 3 * h_size)

        # Split i, o, u
        i, o, u = torch.chunk(iou, 3, dim=1)  # (batch_size, h_size) each

        # Calculate gating
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
        Forward propagation
        :param graph: DGL graph
        :param wordid: Vocabulary IDs of nodes
        :param mask: Mask of nodes (not used currently)
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

        # Forward propagation
        h = torch.zeros(graph.number_of_nodes(), self.model.h_size).to(graph.device)
        c = torch.zeros(graph.number_of_nodes(), self.model.h_size).to(graph.device)
        feature = self.model(graph, wordid, None, h, c)
        root_h = feature[root_idx].unsqueeze(0)  # [1, h_size]
        return self.fc(root_h)
