import torch
import random
import sympy as sp
import torch.nn.functional as F
from torch import nn
from functools import reduce
from parameters import BaseSRParameter, CGPParameter
from factory import CGPFactory


class BaseSRModel(nn.Module):
    def __init__(self, sr_param: BaseSRParameter) -> None:
        super().__init__()
        self.sr_param = sr_param

    def initialization(self):
        pass

    def diff_forward(self, X):
        """ calculate forward difference u """
        return X

    def forward(self, X, expr=False, symbol_constant=False):
        """ forward calculation """
        pass

    def expr(self, input_vars=None, symbol_constant=False):
        pass

    def mutate_(self, probability):
        # self mutate
        pass

    def mutate(self, probability):
        # return mutated genes
        pass

    def assign_genes(self, genes):
        pass


class CGPModel(BaseSRModel):
    def __init__(self, sr_param: CGPParameter, genes=None, ephs=None) -> None:
        super().__init__(sr_param)

        self.factory = CGPFactory(sr_param)
        if genes is None:
            genes, bounds = self.factory.create_genes_and_bounds()
        else:
            bounds = self.factory.create_bounds()

        self.genes = genes
        self.bounds = bounds

        if ephs is None:
            if sr_param.n_eph == 0:
                self.ephs = None
            else:
                self.ephs = nn.Parameter(torch.rand(size=(sr_param.n_eph,)))
        else:
            assert len(ephs) == sr_param.n_eph
            self.ephs = nn.Parameter(ephs)
        # CGP特有的三个属性：结点，活跃路径和活跃结点
        self.nodes = None
        self.active_paths = None
        self.active_nodes = None
        self.initialization()

    @property
    def device(self):
        return self.ephs.device

    def copy_self(self):
        return CGPModel(
            self.sr_param,
            self.genes[:],
            self.ephs.detach().clone()
        ).to(self.device)

    def create(self):
        return CGPModel(
            self.sr_param
        ).to(self.device)

    def initialization(self):
        # whenever `self.genes` is rebuilt, we should call this method
        self.nodes = self.factory.create_nodes(self.genes)
        self.active_paths = self._get_active_paths()
        self.active_nodes = set(reduce(lambda l1, l2: l1 + l2, self.active_paths))

    # 把CGP的获取符号表达式的方法中的一部分拆出来
    def select_features(self, X, idx, expr=False, symbol_constant=False):
        """
            X: shape=(B, D+int(has_diff)), where D is the #var
        """
        n_var_const = self.sr_param.n_inputs + self.sr_param.n_eph  # 计算输入变量和常数的数量和
        if expr:
            if self.sr_param.n_inputs <= idx < n_var_const:
                c = f'c{idx - self.sr_param.n_inputs}' if symbol_constant \
                    else self.ephs[idx - self.sr_param.n_inputs].item()
            elif idx < self.sr_param.n_inputs:
                c = X[idx]
            else:
                # is u
                c = X[-1]
            return c
        else:
            if self.sr_param.n_inputs <= idx < n_var_const:
                return self.ephs[idx - self.sr_param.n_inputs]
            elif idx < self.sr_param.n_inputs:
                return X[idx] if isinstance(X, list) else X[:, idx]
            return X[-1] if isinstance(X, list) else X[:, -1]

    # 定义模型的前向传播操作，得到输出
    def forward(self, X, expr=False, symbol_constant=False):
        """
            if `one_in_one_out` is True, treat X as one single variable
            else: treat X as X[:, 0], X[:, 1], ...

            For regular expression:
                X: shape=(B, D) or D x (B,), where D is the #var
                X[-1] can be u_hat for PDE finding
                return y: shape=(B, O), where O is the #out
            For image expression:
                X: shape=(B, C, H, W),
                return y: shape=(B, h, w)
        """

        outputs = []
        # 遍历活跃路径中的路径中的基因型，判断结点是输入结点、输出结点还是内部结点，来得到不同的输出值
        for path in self.active_paths:
            for gene in path:
                node = self.nodes[gene]
                if node.is_input:
                    if self.sr_param.one_in_one_out:
                        # no matter how much # of variables, we put X as input
                        node.value = X
                    else:
                        node.value = self.select_features(X, node.no)
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                    if len(node.value.shape) == 0:
                        node.value = node.value.repeat(X.shape[0])
                    outputs.append(node.value)
                else:
                    f = node.func
                    operants = [self.nodes[i].value for i in node.inputs]
                    node.value = f(*operants)

        if self.sr_param.one_in_one_out:
            return outputs[0]
        return torch.stack(outputs, dim=1)

    # 获取表达式的方法
    def expr(self, input_vars=None, symbol_constant=False):
        """
            if `one_in_one_out` is True, treat X as one single variable
            else: treat X as X[:, 0], X[:, 1], ...
            return a list of self.n_outputs formulas
        """
        if input_vars is not None and len(input_vars) != self.sr_param.n_inputs + int(self.sr_param.has_diff):
            raise ValueError(f'Expect len(input_vars)={self.sr_param.n_inputs}, but got {len(input_vars)}')

        if input_vars is None:
            if self.sr_param.n_inputs == 1:
                input_vars = ["x"]
            else:
                input_vars = [f'x{i}' for i in range(self.sr_param.n_inputs)]
            input_vars = [sp.Symbol(x) if isinstance(x, str) else x for x in input_vars]
            u = sp.Function("u")
            u = u(*input_vars)
            input_vars.append(u)

        results = []
        for path in self.active_paths:
            for i_node in path:
                node = self.nodes[i_node]
                if node.is_input:
                    c = self.select_features(input_vars, i_node, True, symbol_constant)
                    node.value = c
                elif node.is_output:
                    results.append(self.nodes[node.inputs[0]].value)
                else:
                    f = node.func
                    operants = [self.nodes[i].value for i in node.inputs]
                    # get a sympy symbolic expression.
                    node.value = f.expr(*operants)

        return results

    # 自己变异
    def mutate_(self, probability):
        mutant_genes = self.mutate(probability)

        self.genes = mutant_genes
        self.initialization()

    # 其他变异，和CGP中的变异操作相同
    def mutate(self, probability):
        mutant_genes = self.genes[:]
        low, up = self.bounds[0], self.bounds[1]

        # diff_end_idx = self.sr_param.n_diff_cols * self.sr_param.n_rows * (self.sr_param.max_arity + 1)
        for gidx in range(len(self.genes)):
            chance = random.random()
            if chance < probability:
                candidates = [
                    gene for gene in range(low[gidx], up[gidx] + 1)
                    if gene != self.genes[gidx]
                ]
                if len(candidates) == 0:
                    continue
                mutant_genes[gidx] = random.choice(candidates)
        return mutant_genes

    def get_genes(self):
        return self.genes[:]

    def assign_genes(self, genes):
        self.genes = genes[:]
        self.initialization()

    # 对应原始CGP中的获取活跃路径
    def _get_active_paths(self):
        stack = []
        active_path, active_paths = [], []
        for node in reversed(self.nodes):
            if node.is_output:
                stack.append(node)
            else:
                break

        while len(stack) > 0:
            node = stack.pop()

            if len(active_path) > 0 and node.is_output:
                active_paths.append(list(reversed(active_path)))
                active_path = []

            active_path.append(node.no)

            for input in node.inputs:
                stack.append(self.nodes[input])

        if len(active_path) > 0:
            active_paths.append(list(reversed(active_path)))

        return active_paths


# 处理图像的CGP模型
class ImageCGPModel(CGPModel):
    kernel_size: tuple
    input_channels: int

    def __init__(self, sr_param: CGPParameter, genes=None, ephs=None, kernel_size=(3, 3)) -> None:
        super().__init__(sr_param, genes, ephs)
        self.kernel_size = kernel_size
        self.input_channels = self.sr_param.input_channels

    def select_features(self, X, idx, expr=False, symbol_constant=False):
        """
          X: shape=(B, C, H, W)
          out: selected `X_sub`: shape=(B, H-kh+1, W-kw+1)
        """

        if expr:
            if idx >= self.sr_param.n_inputs:
                eph_idx = idx - self.sr_param.n_inputs
                c = f"c{eph_idx}" if symbol_constant else self.ephs[eph_idx].item()
            else:
                c = X[idx]
            return c

        B, C, H, W = X.shape
        kh, kw = self.kernel_size

        Hout, Wout = H - kh + 1, W - kw + 1
        i_row = idx // kw
        i_col = idx - i_row * kw

        if idx >= self.sr_param.n_inputs:
            return torch.empty(
                size=(B, Hout, Wout), dtype=X.dtype, device=X.device
            ).fill_(self.ephs[idx - self.sr_param.n_inputs])

        # trick operator
        weight = torch.zeros(
            size=(kh, kw),
            dtype=X.dtype,
            device=X.device
        )
        weight[i_row, i_col] = 1.
        weight = weight.reshape(1, 1, kh, kw).expand(1, C, -1, -1)
        selected = F.conv2d(
            X, weight, stride=1, padding=0, dilation=1
        ).squeeze(dim=1)
        return selected

    def forward(self, X, expr=False, symbol_constant=False):
        # output is B x C_out x (H-kh+1) x (W-kw+1)
        return super().forward(X, expr, symbol_constant)

    def expr(self, input_vars=None, symbol_constant=False):
        # let's adjust input_vars
        kh, kw = self.kernel_size
        input_vars = []
        for idx in range(self.sr_param.n_inputs):
            i_row = idx // kw
            i_col = idx - i_row * kw
            var = f"Pixel_{i_row}-{i_col}"
            input_vars.append(var)
        input_vars = [sp.Symbol(pv) for pv in input_vars]
        pixel_expressions = super().expr(input_vars, symbol_constant)
        return pixel_expressions


if __name__ == "__main__":
    # 测试CGPModel
    # class Args:
    #     n_rows = 3
    #     n_cols = 3
    #     levels_back = 3
    #
    #
    # X = torch.randn(5, 2)
    # args = Args()
    # function_set = ["add", "sub", "mul"]
    # cgp_params = CGPParameter(1, 1, 1, args=args, function_set=function_set, one_in_one_out=True)
    # cgp_model = CGPModel(cgp_params)
    # features = cgp_model.select_features(X, 0)
    # print(features.shape)
    # out = cgp_model(X)
    # print(out.shape)
    # print(cgp_model.active_paths)
    # print(cgp_model.expr())

    # 测试ImageCGPModel
    class Args:
        n_rows = 3
        n_cols = 3
        levels_back = 3


    imgs = torch.randn(1, 1, 32, 32)
    args = Args()
    imagecgp_params = CGPParameter(1, 1, n_eph=1, args=args)
    imagecgp_model = ImageCGPModel(imagecgp_params)
    features = imagecgp_model.select_features(imgs,expr=False,idx=1)
    print(features.shape)
    out = imagecgp_model(imgs)
    print(out.shape)
    print(imagecgp_model.expr())
