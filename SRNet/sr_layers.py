import sympy as sp
import math
import copy
from torch import nn
from parameters import BaseSRParameter
from nn_parser import ExplainedModule
from sr_models import BaseSRModel, ImageCGPModel
from usr_models import ImageEQL, EQLPDE


class LinearSRModule(nn.Module):
    def __init__(self, sr_model: BaseSRModel, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sr_model = sr_model
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, X):
        if len(X.shape) > 2:
            # may come from the output of image cgp
            B = X.shape[0]
            X = X.reshape(B, -1)
        out = self.sr_model(X)
        return self.proj(out)


class ImageSRModule(nn.Module):
    def __init__(
            self,
            sr_model: BaseSRModel,
            C_in: int,
            C_out: int
    ) -> None:
        super().__init__()
        self.sr_model = sr_model
        self.proj = None
        if C_in != C_out:
            self.proj = nn.Linear(C_in, C_out)

    def forward(self, X):
        # B, C_in, H, W
        out = self.sr_model(X)
        B, C_in, H, W = out.shape
        # B, H, W, C_out
        if self.proj is not None:
            out = self.proj(out.reshape(B, H, W, -1))
            out = out.reshape(B, -1, H, W)
        return out


# 根据AutoPUM模块划分算法，对划分出的不同功能的模块进行模型构建
class SRLayer(nn.Module):
    def __init__(
            self,
            sr_param: BaseSRParameter,
            sr_class,
            explained_module: ExplainedModule
    ) -> None:
        super().__init__()
        self.sr_param = sr_param
        self.sr_class = sr_class
        self.ex_module = explained_module
        self.model = self.construct_layer()

    @property
    def device(self):
        return next(self.parameters()).device

    # 构建一个卷积层
    def build_conv(self, in_shape, out_shape):
        C_in, H_in, W_in = in_shape
        C_out, H_out, W_out = out_shape

        # supporting padding=0, stride=1, dialtion=1
        # k = i + 2p + 1 - o
        kernel_size_H = H_in + 1 - H_out
        kernel_size_W = W_in + 1 - W_out
        kernel_size = (kernel_size_H, kernel_size_W)

        self.sr_param.n_inputs = kernel_size_H * kernel_size_W
        self.sr_param.input_channels = C_in
        self.sr_param.n_outputs = C_out

        # if isinstance(self.sr_param, CGPParameter):
        #     self.sr_class = ImageCGPModel
        # elif isinstance(self.sr_param, GPParameter):
        #     self.sr_class = ImageGPModel
        # elif isinstance(self.sr_param, EQLParameter):
        #     if self.sr_class != EQLPDE:
        #         self.sr_class = ImageEQL
        # else:
        #     raise ValueError(f"Not implement image model for {self.sr_param.__class__}")

        if str(self.sr_param.__class__) == "<class 'SRNet.parameters.CGPParameter'>":
            self.sr_class = ImageCGPModel
        elif str(self.sr_param.__class__) == "<class 'SRNet.parameters.EQLParameter'>":
            if self.sr_class != EQLPDE:
                self.sr_class = ImageEQL
        else:
            raise ValueError(f"Not implement image model for {self.sr_param.__class__}")

        sr_model = self.sr_class(self.sr_param, kernel_size=kernel_size)
        model = ImageSRModule(sr_model, self.sr_param.n_outputs, C_out)
        return model

    # 构建一个线性层
    def build_linear(self, in_shape, out_shape):
        # original linear builder
        in_features = math.prod(in_shape)
        out_features = math.prod(out_shape)
        self.sr_param.n_inputs = in_features
        self.sr_param.n_outputs = 1
        self.sr_param.remove_img_functions_()  # 线性层中没有关于图像的sober算子，需要移除

        sr_model = self.sr_class(self.sr_param)
        model = LinearSRModule(sr_model, self.sr_param.n_outputs, out_features)

        return model

    # 构建一个激活层
    def build_activation(self, in_shape, out_shape):
        sr_param = copy.deepcopy(self.sr_param)
        sr_param.n_inputs = 1
        sr_param.n_outputs = 1
        sr_param.n_eph = 0
        sr_param.one_in_one_out = True

        self.sr_param = sr_param
        self.sr_param.remove_img_functions_()
        sr_model = self.sr_class(sr_param)
        return sr_model

    # 根据之前的模块划分结果，对不同类型的模块（卷积，线性，激活函数）构建不同的层
    def construct_layer(self):
        in_shape = self.ex_module.in_shape
        out_shape = self.ex_module.out_shape
        em_type = self.ex_module.em_type
        if "Conv" in em_type:
            return self.build_conv(in_shape, out_shape)
        if "Linear" in em_type:
            return self.build_linear(in_shape, out_shape)
        if "Activation" in em_type:
            return self.build_activation(in_shape, out_shape)

    def forward(self, X):
        return self.model(X)

    def expr(self, vars=None, mul_w=True):
        if vars is not None and isinstance(vars, sp.Matrix):
            vars = vars.tolist()[0]

        if isinstance(self.model, BaseSRModel):
            return sp.Matrix(self.model.expr(vars)).T

        # list of sympy expressions
        if isinstance(self.model, ImageSRModule):
            cgp_expr = self.model.sr_model.expr(vars)
            # directly return expression for simplify
            return cgp_expr
        elif isinstance(self.model, LinearSRModule):
            cgp_expr = self.model.sr_model.expr(vars)
            proj: nn.Linear = self.model.proj
        else:
            cgp_expr = self.model[0].expr(vars)
            proj: nn.Linear = self.model[1]

        if not mul_w:
            return cgp_expr
        weight = proj.weight.detach().numpy().T
        bias = proj.bias.detach().numpy()

        expr = sp.Matrix(cgp_expr).T * sp.Matrix(weight) + sp.Matrix(bias).T
        return expr

    def copy_self(self):
        new_module = copy.deepcopy(self)
        return new_module.to(self.device)

    def get_srmodel(self):
        # 返回的self.model是根据模块划分结果构建出的不同的层
        # self.model.sr_model是线性SR模块还是图像SR模块
        if isinstance(self.model, BaseSRModel):
            return self.model
        elif isinstance(self.model, (LinearSRModule, ImageSRModule)):
            return self.model.sr_model
        else:
            return self.model[0]

    def mutate_(self, probability):
        self.get_srmodel().mutate_(probability)

    def mutate(self, probability):
        return self.get_srmodel().mutate(probability)

    def genes(self):
        return self.get_srmodel().genes

    def assign_genes(self, genes):
        self.get_srmodel().assign_genes(genes)

    def regularization(self):
        return self.get_srmodel().regularization()