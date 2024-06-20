"""
    finetune EQL model using prune strategy
"""

import torch
from SRNet.functions import Var, ide


class Node:
    def __init__(self, type, in_weights, in_bias, layer_idx, node_idx, children=None) -> None:
        self.type = type
        self.in_weights = in_weights
        self.in_bias = in_bias
        self.children = children
        self.layer_idx = layer_idx
        self.node_idx = node_idx

    def sort_children(self):
        self.children.sort(key=lambda x: x.in_weights.abs().mean(dim=1))
        _, sort_indices = torch.sort(self.in_weights.abs().mean(dim=1))
        self.in_weights = self.in_weights[sort_indices]


def construct_graph(model):
    def dfs(root, layer_idx, node_idx, in_idx):
        weights = layers[layer_idx - 1].weight.transpose(0, 1)
        bias = layers[layer_idx - 1].bias
        root_node = Node(
            root,
            weights[:, in_idx:in_idx + root.arity],
            bias[in_idx:in_idx + root.arity],
            layer_idx,
            node_idx
        )
        children = []
        if layer_idx > 1:
            next_in_idx = 0
            for i, child in enumerate(function_set):
                child_node = dfs(child, layer_idx - 1, i, next_in_idx)
                next_in_idx += child.arity
                children.append(child_node)
        children.extend(var_nodes)

        root_node.children = children
        return root_node

    layers = model.layers
    params = model.sr_param
    function_set = params.function_set
    vars = [Var(f"x{i}") for i in range(layers[0].in_features)]
    var_nodes = [Node(var, None, None, 0, i) for i, var in enumerate(vars)]

    root = dfs(ide, len(layers), 0, 0)  # 这里为什么要用ide这个算子，直接写root可以吗？
    return root


def pruning_finetune(root: Node):
    children = root.children
    if children is None:
        pass
    else:
        for child in children:
            pruning_finetune(child)
