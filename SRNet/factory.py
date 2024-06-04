import sys
sys.path.append("/Users/lihaoyang/Projects/srn-m/SRNet")
import random
from parameters import CGPParameter


# CGP中结点的定义
class Node:
    def __init__(self, no, func, arity, inputs=None, start_gidx=None):
        """
        :param no: index of node
        :param func: node's function, None when node is input or output
        :param arity: func's arity
        :param inputs: node's input genes.
        :param start_gidx: start position in the genes.
        """
        if inputs is None:
            inputs = []
        self.no = no
        self.func = func
        self.arity = arity
        self.inputs = inputs
        self.value = None
        self.is_input = False
        self.is_output = False
        if func is None:
            if len(self.inputs) == 0:
                self.is_input = True
            else:
                self.is_output = True

        self.start_gidx = start_gidx

    def __repr__(self):
        return f'Node({self.no}, {self.func}, {self.inputs})'


# 定义CGP
class CGPFactory:

    def __init__(self, params):
        self.params = params
        self.n_inputs = params.n_inputs
        self.n_outputs = params.n_outputs
        self.n_rows = params.n_rows
        self.n_cols = params.n_cols
        self.max_arity = params.max_arity
        self.levels_back = params.levels_back
        self.function_set = params.function_set
        self.diff_function_set = params.diff_function_set
        self.n_eph = params.n_eph
        self.n_f = len(self.function_set)
        self.has_diff = params.has_diff
        self.n_f_node = self.n_rows * self.n_cols
        self.n_diff_cols = params.n_diff_cols

    def create_genes_and_bounds(self):
        genes = []
        uppers, lowers = [], []
        func_start = self.n_inputs + self.n_eph + int(self.has_diff)
        func_end = func_start + self.n_f_node
        for i in range(func_start, func_end):
            col = (i - func_start) // self.n_rows

            # first bit is node function
            if col < self.n_diff_cols:
                # diff layers
                f_gene = random.randint(0, len(self.diff_function_set) - 1)
                up = len(self.diff_function_set) - 1
            else:
                f_gene = random.randint(0, self.n_f - 1)
                up = self.n_f - 1
            lowers.append(0)
            uppers.append(up)
            genes.append(f_gene)

            # next bits are input of the node function.
            up = func_start + col * self.n_rows - 1
            low = max(0, up - self.levels_back)
            if col < self.n_diff_cols:
                # first arity should be u or previous layer d(u)
                # second arity should be input variables
                ups = [func_start + col * self.n_rows - 1, self.n_inputs - 1] + [up] * (self.max_arity - 2)
                lows = [max(func_start - 1, up - self.levels_back), 0] + [low] * (self.max_arity - 2)
                for u, l in zip(ups, lows):
                    lowers.append(l)
                    uppers.append(u)
                    in_gene = random.randint(l, u)
                    genes.append(in_gene)
            else:
                for i_arity in range(self.max_arity):
                    lowers.append(low)
                    uppers.append(up)
                    in_gene = random.randint(low, up)
                    genes.append(in_gene)

        # output genes
        up = func_start + self.n_f_node - 1
        low = max(0, up - self.levels_back)
        for i in range(self.n_outputs):
            lowers.append(low)
            uppers.append(up)
            out_gene = random.randint(low, up)
            genes.append(out_gene)

        return genes, (lowers, uppers)

    def create_bounds(self):
        """when provide genes, create bounds by genes"""
        uppers, lowers = [], []

        func_start = self.n_inputs + self.n_eph + int(self.has_diff)
        func_end = func_start + self.n_f_node
        for i in range(func_start, func_end):
            col = (i - func_start) // self.n_rows

            lowers.append(0)
            uppers.append(self.n_f - 1 if col >= self.n_diff_cols else len(self.diff_function_set) - 1)

            up = func_start + col * self.n_rows - 1
            low = max(0, up - self.levels_back)
            if col < self.n_diff_cols:
                # first arity should be u or previous layer d(u)
                # second arity should be input variables
                ups = [func_start + col * self.n_rows - 1, self.n_inputs - 1] + [up] * (self.max_arity - 2)
                lows = [max(func_start - 1, up - self.levels_back), 0] + [low] * (self.max_arity - 2)
                for u, l in zip(ups, lows):
                    lowers.append(l)
                    uppers.append(u)
            else:
                for i_arity in range(self.max_arity):
                    lowers.append(low)
                    uppers.append(up)

        up = func_start + self.n_f_node - 1
        low = max(0, up - self.levels_back)
        for i in range(self.n_outputs):
            lowers.append(low)
            uppers.append(up)

        return lowers, uppers

    def create_nodes(self, genes):
        nodes = []
        func_start = self.n_inputs + self.n_eph + int(self.has_diff)
        for i in range(func_start):
            nodes.append(Node(i, None, 0, None))

        f_pos = 0
        for i in range(self.n_f_node):
            col = i // self.n_rows

            f_gene = genes[f_pos]
            if col < self.n_diff_cols:
                f = self.diff_function_set[f_gene]
            else:
                f = self.function_set[f_gene]

            input_genes = genes[f_pos + 1: f_pos + f.arity + 1]
            nodes.append(Node(i + func_start, f, f.arity, input_genes, start_gidx=f_pos))
            f_pos += self.max_arity + 1

        idx_output_node = func_start + self.n_f_node
        for gene in genes[-self.n_outputs:]:
            nodes.append(Node(idx_output_node, None, 0, [gene], start_gidx=f_pos))
            f_pos += 1
            idx_output_node += 1

        return nodes


if __name__ == '__main__':
    class Args:
        n_rows = 3
        n_cols = 3
        levels_back = 3

    args = Args()
    function_set = ["add", "sub", "mul"]
    cgp_params = CGPParameter(2, 1, 1, args=args, function_set=function_set, one_in_one_out=False)
    cgp_model = CGPFactory(cgp_params)
    genes,bounds=cgp_model.create_genes_and_bounds()
    nodes=cgp_model.create_nodes(genes)
    print(f'genes:{genes}\nbounds:{bounds}\nnodes:{nodes}')
