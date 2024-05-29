import sys
from functions import function_map, default_functions

sys.path.append("/Users/lihaoyang/Projects/srn-m/SRNet")


class BaseSRParameter:
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_eph,
            function_set,
            one_in_one_out  # if True, ignore `n_inputs` and `n_outputs`, treat them both as 1 during inference,
    ) -> None:
        self.input_channels = None
        if one_in_one_out:
            n_inputs, n_outputs, n_eph = 1, 1, 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_eph = n_eph
        self.one_in_one_out = one_in_one_out

        if function_set is None or len(function_set) == 0:
            function_set = default_functions
        # 选择默认的运算符，放到function_map中
        self.function_set = [
            function_map[f] if isinstance(f, str) else f for f in function_set
        ]
        self.max_arity = max(f.arity for f in self.function_set)

    # 转化成图像卷积的运算公式ic_xxx
    def convert_to_ic_function_(self, *kernel_size):
        orig_function_set = [f.func_name for f in self.function_set]
        ic_function_set = [f"ic_{func}" for func in orig_function_set]

        self.function_set = [function_map[f] for f in ic_function_set]
        # 变为卷积运算公式之后，还要给每个运算指定卷积核大小，通过调用function.py中的assign_kernel_size()函数
        for i in range(len(self.function_set)):
            self.function_set[i].assign_kernel_size(*kernel_size)

    # 删除关于图像卷积的运算公式中除sgx,sgy之外的运算符
    def remove_img_functions_(self):
        new_functions = [f for f in self.function_set if "sg" not in f.func_name]
        self.function_set = new_functions


# CGP参数
class CGPParameter(BaseSRParameter):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_eph,
            args,
            function_set=None,
            one_in_one_out=False
    ) -> None:
        super().__init__(n_inputs, n_outputs, n_eph, function_set, one_in_one_out)

        # CGP独有的参数，放到args里去，单独初始化和调用
        self.n_rows = args.n_rows
        self.n_cols = args.n_cols
        if args.levels_back is None:
            self.levels_back = args.n_rows * args.n_cols + n_inputs + 1
        self.levels_back = args.levels_back

        # see if there are diff functions
        # 看有没有微分算子
        self.has_diff = any("diff" in f.func_name for f in self.function_set)
        self.n_diff_cols = 0
        self.diff_function_set = [
            f for f in self.function_set if "diff" in f.func_name
        ]
        self.function_set = [
            f for f in self.function_set if "diff" not in f.func_name
        ]
        if self.has_diff:
            # add layers diff operators
            # 有的话就把微分算子加到CGP图的列中去
            self.n_diff_cols = args.n_diff_cols if hasattr(args, "n_diff_cols") else 2
            self.n_cols += self.n_diff_cols


# EQL参数
class EQLParameter(BaseSRParameter):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_eph,
            args,
            function_set=None,
            one_in_one_out=False,
    ) -> None:
        super().__init__(n_inputs, n_outputs, n_eph, function_set, one_in_one_out)
        self.n_layers = args.n_layers
        self.hidden_out_size = len(self.function_set)
        n_arities = 0
        for f in self.function_set:
            n_arities += f.arity
        self.hidden_in_size = n_arities


class KANParameter(BaseSRParameter):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_eph,
            args,
            function_set=None,
            one_in_one_out=False,
    ) -> None:
        super().__init__(n_inputs, n_outputs, n_eph, function_set, one_in_one_out)

        self.layers_hidden = args.layers_hidden
        self.grid_size = args.grid_size
        self.spline_order = args.spline_order
        self.scale_noise = args.scale_noise
        self.scale_base = args.scale_base
        self.scale_spline = args.scale_spline
        self.base_activation = args.base_activation
        self.grid_eps = args.grid_eps
        self.grid_range = args.grid_range
