import sympy as sp
import torch
import kornia as K
import math
import torch.nn.functional as F
from typing import Tuple
from functools import partial

threshold = 1e-3


# 运算符的形式
class Function:
    def __init__(self, func_name, arity, pt_fn, sp_fn) -> None:
        self.func_name = func_name
        self.arity = arity  # 参数的数量
        self.pt_fn = pt_fn  # pytorch版的公式
        self.sp_fn = sp_fn  # sympy版的公式

    # 允许一个类的实例像函数一样被调用。这意味着可以创建一个对象，然后像调用普通函数一样调用这个对象
    def __call__(self, *args):
        return self.pt_fn(*args)

    def expr(self, *sym_args):
        return self.sp_fn(*sym_args)

    # 和__repr__方法差不多
    def __str__(self) -> str:
        return self.func_name

    def __repr__(self) -> str:
        return self.func_name


# 变量的形式
class Var(Function):
    def __init__(self, name) -> None:
        super().__init__(name, 0, None, None)

    def __call__(self, *args):
        return args[0]

    def expr(self):
        return sp.Symbol(self.func_name)


# 在图像卷积中应用运算符
class ImageConvFunction(Function):
    """
        Input image(s): H x W
        Convolutional apply symbolic operation on each image
        f~conv~image(s)
        e.g.
        pt_fn = sin -> return [sin(img[1:9]), sin(img[9:17]), sin(img[17:25])...]
        pt_fn = * -> return [mul(img[1:9]), mul(img[1:9])...]
    """

    kernel_size: Tuple[int, int]

    def __init__(
            self, func_name, arity, pt_fn, sp_fn,
            kernel_size=None, padding=0, stride=1, dilation=1
    ) -> None:
        super().__init__(func_name, arity, pt_fn, sp_fn)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def assign_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _apply_conv(self, *imgs):

        # 计算卷积后图像的宽度和高度
        def _conv_out_size(i, k):
            return math.ceil(
                (i + 2 * self.padding - self.dilation * (k - 1) - 1) / self.stride + 1
            )

        B, H, W = imgs[0].shape  # 这里的imgs是一个元组，只包含一个tensor，所以写成imgs[0]
        out_height = _conv_out_size(H, self.kernel_size[0])
        out_width = _conv_out_size(W, self.kernel_size[1])

        out_img = self.pt_fn(*imgs)
        # tricky operation
        # 调整输出图像的大小
        # 先创建一个全1的权重矩阵，其大小与输出图像的大小相同
        weight = torch.ones(
            size=(1, 1, out_height, out_width),
            device=imgs[0].device,
            dtype=imgs[0].dtype
        )
        # 然后将输出图像重塑为一个4维张量，使用刚刚创建的全1矩阵和其他卷积参数进行卷积操作
        out_img = F.conv2d(
            out_img.unsqueeze(1),  # reshape to (B, 1, H, W)
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
        # 最后返回第一个通道的输出。结果是一个二维张量，其形状与输入图像的形状相同。
        return out_img[:, 0]

    def __call__(self, *args):
        return self._apply_conv(*args)

    def expr(self, *sym_args):
        return super().expr(*sym_args)


# 定义二元运算符
def binary_func(x1, x2, type, lib="sp"):
    """ since lambda function cannot be pickle """
    if type == "add":
        return x1 + x2
    elif type == "sub":
        return x1 - x2
    elif type == "mul":
        return x1 * x2
    elif type == "div":
        if lib == "torch":
            return torch.div(x1, torch.abs(x2) + threshold)
        else:
            return x1 / (sp.Abs(x2) + threshold)
    elif "diff" in type:
        if lib == "torch":
            # 计算梯度。x1是输入，x2是输出。这里只计算x1的梯度
            dif = torch.autograd.grad(
                x1, x2, torch.ones_like(x1),
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]

            if type == "diff2":
                # 计算二阶导数。计算dif相对于x2的二阶导数
                dif = torch.autograd.grad(
                    dif, x2, torch.ones_like(dif),
                    retain_graph=True, create_graph=True, only_inputs=True
                )[0]
            return dif
        else:
            # 如果不用pytorch，用sympy的话，直接计算一阶导和二阶导
            diff = x1.diff(x2)
            if type == "diff2":
                diff = diff.diff(x2)
            return diff


# 定义一元运算符
def unary_func(x, type, lib="sp"):
    if type == "id":
        return x
    elif type == "log":
        return torch.log(torch.abs(x) + threshold) if lib == "torch" else sp.log(sp.Abs(x) + threshold)
    elif type == "sin":
        return sp.sin(x)
    elif type == "cos":
        return sp.cos(x)
    elif type == "sqrt":
        return torch.sqrt(torch.abs(x) + threshold) if lib == "torch" else sp.sqrt(sp.Abs(x) + threshold)
    elif type == "square":
        return x ** 2
    elif type == "cube":
        return x ** 3
    elif type == "zero1":
        return torch.zeros_like(x) if lib == "torch" else 0.
    elif type == "image_dx":
        if lib == "torch":
            if len(x.shape) < 4:
                x = x.unsqueeze(1)
            out = K.filters.spatial_gradient(x)[:, :, 0]  # 使用sobel算子。因为sobel算子会在H和W之前新增两个维度，一个是x，一个是y。这里[:,:,0]取第一个，为x
            if len(out.shape) < 4:
                out = out.unsqueeze(1)
            return out
        else:
            return sp.Symbol(f"sgx({x})")
    elif type == "image_dy":
        if lib == "torch":
            if len(x.shape) < 4:
                x = x.unsqueeze(1)
            out = K.filters.spatial_gradient(x)[:, :, 1]
            if len(out.shape) < 4:
                out = out.unsqueeze(1)
            return out
        else:
            return sp.Symbol(f"sgy({x})")


add = Function("add", 2, torch.add, partial(binary_func, type="add"))
sub = Function("sub", 2, torch.sub, partial(binary_func, type="sub"))
mul = Function("mul", 2, torch.mul, partial(binary_func, type="mul"))
div = Function("div", 2, partial(binary_func, type="div", lib="torch"), partial(binary_func, type="div"))
diff = Function("diff", 2, partial(binary_func, type="diff", lib="torch"), partial(binary_func, type="diff"))
diff2 = Function("diff2", 2, partial(binary_func, type="diff2", lib="torch"), partial(binary_func, type="diff2"))

ide = Function("id", 1, partial(unary_func, type="id"), partial(unary_func, type="id"))
log = Function("log", 1, partial(unary_func, type="log", lib="torch"), partial(unary_func, type="log"))
sin = Function("sin", 1, torch.sin, sp.sin)
cos = Function("cos", 1, torch.cos, sp.cos)
sqrt = Function("sqrt", 1, partial(unary_func, type="sqrt", lib="torch"), partial(unary_func, type="sqrt"))
square = Function("square", 1, torch.square, partial(unary_func, type="square"))
cube = Function("cube", 1, partial(unary_func, type="cube", lib="torch"), partial(unary_func, type="cube"))
exp = Function("exp", 1, torch.exp, sp.exp)
zero = Function("zero1", 1, partial(unary_func, type="zero1", lib="torch"), partial(unary_func, type="zero1"))  # 没用到
img_dx = Function("sgx", 1, partial(unary_func, type="image_dx", lib="torch"), partial(unary_func, type="image_dx"))
img_dy = Function("sgy", 1, partial(unary_func, type="image_dy", lib="torch"), partial(unary_func, type="image_dy"))

""" Image Convolutional Function, e.g. conv2d(sin(img), weight=1) """
ic_add = ImageConvFunction("ic_add", 2, torch.add, partial(binary_func, type="add"))
ic_sub = ImageConvFunction("ic_sub", 2, torch.sub, partial(binary_func, type="sub"))
ic_mul = ImageConvFunction("ic_mul", 2, torch.mul, partial(binary_func, type="mul"))
ic_div = ImageConvFunction("ic_div", 2, partial(binary_func, type="div", lib="torch"), partial(binary_func, type="div"))

ic_ide = ImageConvFunction("ic_id", 1, partial(unary_func, type="id"), partial(unary_func, type="id"))
ic_log = ImageConvFunction("ic_log", 1, partial(unary_func, type="log", lib="torch"), partial(unary_func, type="log"))
ic_sin = ImageConvFunction("ic_sin", 1, torch.sin, sp.sin)
ic_cos = ImageConvFunction("ic_cos", 1, torch.cos, sp.cos)
ic_sqrt = ImageConvFunction("ic_sqrt", 1, partial(unary_func, type="sqrt", lib="torch"),
                            partial(unary_func, type="sqrt"))
ic_square = ImageConvFunction("ic_square", 1, torch.square, partial(unary_func, type="square"))

function_map = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "diff": diff,
    "diff2": diff2,

    "id": ide,
    "ide": ide,
    "log": log,
    "sin": sin,
    "cos": cos,
    "sqrt": sqrt,
    "square": square,
    "cube": cube,
    "exp": exp,

    "image_dx": img_dx,
    "image_dy": img_dy,

    "ic_add": ic_add,
    "ic_sub": ic_sub,
    "ic_mul": ic_mul,
    "ic_div": ic_div,
    "ic_id": ic_ide,
    "ic_ide": ic_ide,
    "ic_log": ic_log,
    "ic_sin": ic_sin,
    "ic_cos": ic_cos,
    "ic_sqrt": ic_sqrt,
    "ic_square": ic_square,
}

default_functions = ["add", "sub", "mul", "div", "ide", "log", "sin", "cos", "sqrt", "square"]
# 原始的EQL中没有除法运算
eql_add_sg_functions = [
    "add", "mul", "ide", "log", "sin", "cos", "sqrt", "square",
    "image_dx", "image_dy"
]

add_sg_functions = [
    "add", "sub", "mul", "div", "ide", "log", "sin", "cos", "sqrt", "square",
    # add sgx
    "image_dx", "image_dy"
]
add_diff_functions = [
    "add", "sub", "mul", "div", "ide", "log", "sin", "cos", "sqrt", "square",
    # add diff
    "diff", "diff2"
]
default_ic_functions = [
    "ic_add", "ic_sub", "ic_mul", "ic_div", "ic_ide", "ic_log", "ic_sin", "ic_cos", "ic_sqrt", "ic_square"
]


if __name__ == "__main__":
    img=torch.rand(1,32,32)
    ic_add.assign_kernel_size(kernel_size=(2,2))
    # ic_add._apply_conv(img)  # 报错
    img_dx = unary_func(img,"image_dx","torch")
    print(img_dx)