import copy
import torch
import torch.nn.functional as F
import sympy as sp
import numpy as np
from torch import Tensor, nn
from parameters import CGPParameter, EQLParameter
from sr_models import CGPModel
from functions import img_dx, img_dy


# PINN
class DiffMLP(nn.Module):
    def __init__(self, in_features, n_layer=5, hidden_size=100) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_layer = n_layer
        self.hidden_size = hidden_size

        models = [nn.Linear(in_features, hidden_size), nn.ReLU()]
        for _ in range(n_layer):
            models.append(nn.Linear(hidden_size, hidden_size))
            models.append(nn.ReLU())
        models.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*models)

    def forward(self, X):
        if isinstance(X, list):
            X = torch.stack(X, dim=-1)
        return self.model(X).squeeze()  # 用于移除tensor中最后一个大小为1的维度


class LinearProj(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 4:
            # img tensor: (B, C, H, W)
            input = input.transpose(-1, 1)  # 这里为什么要转换维度？
        out = super().forward(input)

        return out


# EQL
class EQL(nn.Module):
    def __init__(self, sr_param: EQLParameter) -> None:
        super().__init__()
        self.sr_param = sr_param
        hidden_in_size = 0
        for f in sr_param.function_set:
            hidden_in_size += f.arity
        self.sr_param.hidden_in_size = hidden_in_size
        self.sr_param.hidden_out_size = len(sr_param.function_set)
        self.layers = nn.ModuleList(self.build_layers())

    # 构建EQL的每一层
    def build_layers(self):
        in_features = self.sr_param.n_inputs

        layers = [nn.Linear(in_features, self.sr_param.hidden_in_size)]
        layer_in_size = self.sr_param.hidden_out_size + in_features
        for i in range(self.sr_param.n_layers - 1):
            layers.append(nn.Linear(layer_in_size, self.sr_param.hidden_in_size))
        out_proj = nn.Linear(layer_in_size, self.sr_param.n_outputs)
        layers.append(out_proj)

        return layers

    def apply_activation(self, X, expr=False):
        in_idx = 0
        out = []
        for f in self.sr_param.function_set:
            arity = f.arity
            if expr:
                f = f.expr
            # 确保索引不会超出X的范围
            max_index = in_idx + arity
            if isinstance(X, torch.Tensor) and max_index > X.size(-1):
                raise ValueError(
                    f"尝试访问的索引超出了X的范围。最大允许索引: {X.size(-1) - 1}, 尝试访问的索引: {max_index - 1}")

            args = [
                X[in_idx + i] if not isinstance(X, torch.Tensor) else X[:, ..., in_idx + i] for i in range(arity)
            ]
            out.append(f(*args))
            in_idx += arity
        if expr:
            return out
        return torch.stack(out, dim=-1)

    # 前向传播操作，对于EQL中的第一层，隐层和最后一层的操作都不太一样
    def forward(self, X: list or torch.Tensor):
        input = X
        if isinstance(X, list):
            # (B, C, H_out, W_out, nvar)
            input = torch.stack(X, dim=-1)
        out = self.layers[0](input)
        out = self.apply_activation(out)

        for i in range(1, len(self.layers) - 1):
            out = self.layers[i](torch.cat([out, input], dim=-1))
            out = self.apply_activation(out)
        out = self.layers[-1](torch.cat([out, input], dim=-1))
        return out

    def expr(self, input_vars=None, sparse_filter=0.01):

        def get_layer_param(layer_idx):
            weight = self.layers[layer_idx].weight.detach().cpu().numpy().T
            bias = self.layers[layer_idx].bias.detach().cpu().numpy()
            if sparse_filter > 0:
                weight = np.where(np.abs(weight) <= sparse_filter, np.zeros_like(weight), weight)
                bias = np.where(np.abs(bias) <= sparse_filter, np.zeros_like(bias), bias)
            return weight, bias

        if input_vars is None:
            input_vars = [f"x{i}" for i in range(self.sr_param.n_inputs)]
        input_vars = [sp.Symbol(v) if isinstance(v, str) else v for v in input_vars]

        # 第一层
        weight, bias = get_layer_param(0)
        out = sp.Matrix(input_vars).T * sp.Matrix(weight) + sp.Matrix(bias).T
        out = self.apply_activation(out, expr=True)
        # 隐层
        for i in range(1, len(self.layers) - 1):
            weight, bias = get_layer_param(i)
            out = out + input_vars
            out = sp.Matrix(out).T * sp.Matrix(weight) + sp.Matrix(bias).T
            out = self.apply_activation(out, expr=True)
        # 最后一层
        weight = self.layers[-1].weight.detach().cpu().numpy().T
        bias = self.layers[-1].bias.detach().cpu().numpy()
        out = out + input_vars
        out = sp.Matrix(out).T * sp.Matrix(weight) + sp.Matrix(bias).T
        return out

    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg


class EQLPDE(nn.Module):
    def __init__(self, sr_param: EQLParameter, with_fu=False, pd_lib=['dx', 'dy', 'dxdy']) -> None:
        super().__init__()
        self.sr_param = sr_param
        self.u_model = None
        self.with_fu = with_fu
        if with_fu:
            self.u_model = EQL(sr_param)
        else:
            self.u_model = DiffMLP(
                sr_param.n_inputs + sr_param.n_eph,
                n_layer=sr_param.n_layers
            )

        self.pd_lib = pd_lib
        pde_param = copy.deepcopy(sr_param)
        pde_param.n_inputs = len(pd_lib)
        pde_param.n_outputs = 1
        # important: strict `n_layers` as 1
        pde_param.n_layers = 1
        self.pde_param = pde_param
        self.pde_model = EQL(pde_param)  # 之后进行替换的时候，估计得从这里改？

    # 求导
    def diff_item(self, u, x, item='x', real_u=False):
        # 如果u是神经网络的预测值，那么就求导
        if not real_u:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # 如果u是真实值，直接对图像进行求导操作
        # now, diff.shape should be (B, C, H, W)
        diff = img_dx(u) if item == 'x' else img_dy(u)
        return diff

    # 构建偏微分项
    def build_pde_lib(self, x, y, t, u, pd_item, real_u):
        if pd_item == 'dt':
            di = self.diff_item(u, t, real_u=real_u)

        elif pd_item == 'dx':
            di = self.diff_item(u, x, 'x', real_u=real_u)
        elif pd_item == 'dy':
            di = self.diff_item(u, y, 'y', real_u=real_u)

        elif pd_item == 'dx2':
            di = self.diff_item(self.diff_item(u, x, 'x', real_u=real_u), x, 'x', real_u=real_u)
        elif pd_item == 'dy2':
            di = self.diff_item(self.diff_item(u, x, 'y', real_u=real_u), x, 'y', real_u=real_u)
        elif pd_item == 'dxdy':
            di = self.diff_item(self.diff_item(u, x, 'x', real_u=real_u), x, 'y', real_u=real_u)
        else:
            raise ValueError("Invalid pd_item: %s" % pd_item)
        if real_u:
            di = di.reshape(-1)  # 将tensor的形状改为一串，没有行和列
        return di

    # 返回各个偏微分项，用于之后计算loss
    def forward(self, x, y, t, dx=None, dy=None, u_real=None):
        input_data = [x, y, t]
        if dx is not None:
            input_data.append(dx)
        if dy is not None:
            input_data.append(dy)

        # if self.with_fu:
        u_hat = self.u_model(input_data)
        # else:
        #     assert u_real is not None, "`u_real` should not be None when `with_fu=False`"

        pd_reals = []
        pd_hats = []  # 经过pinn后得到的偏微分项，之后要被输入进eql中
        for pd_item in self.pd_lib:
            pd_hat = self.build_pde_lib(x, y, t, u_hat, pd_item, real_u=False)
            pd_hats.append(pd_hat)
            if u_real is not None:
                pd_real = self.build_pde_lib(x, y, t, u_real, pd_item, real_u=True)
                pd_reals.append(pd_real)

        # if self.with_fu:
        pde_out = self.pde_model(pd_hats)  # 将pinn生成的偏微分项输入进eql中
        # else:
        #     pde_out = self.pde_model(pd_reals)
        return {
            "u_hat": u_hat,
            "pd_reals": pd_reals,
            "pd_hats": pd_hats,
            "pde_out": pde_out
        }

    def expr(self, input_vars=None, sparse_filter=0.01):
        return self.pde_model.expr(input_vars, sparse_filter=sparse_filter)

    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.pde_model.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg


class ImageEQL(EQL):
    kernel_size: tuple
    input_channels: int

    def __init__(self, sr_param: EQLParameter, kernel_size=(3, 3)) -> None:
        super().__init__(sr_param)
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

        i_row = idx // kw
        i_col = idx - i_row * kw

        # trick operator
        weight = torch.zeros(
            size=(kh, kw),
            dtype=X.dtype,
            device=X.device
        )
        weight[i_row, i_col] = 1.
        weight = weight.reshape(1, 1, kh, kw)

        selected = [
            nn.functional.conv2d(
                X[:, ic:ic + 1], weight, stride=1, padding=0, dilation=1
            ) for ic in range(C)
        ]
        return torch.cat(selected, dim=1)

    def forward(self, X):
        # X: shape=(B, C, H, W)
        # out: shape=(B, C_out, H_out, W_out)

        # nvar * (B, H_out, W_out)
        # nvar * (B, C, H_out, W_out)
        features = [
            self.select_features(X, i)
            for i in range(self.sr_param.n_inputs)
        ]
        out = super().forward(features)
        out = out.mean(dim=1)  # 相当于把out的dim=1那一维（C）给删掉了
        B, H_out, W_out, C_out = out.shape
        return out.reshape(B, C_out, H_out, W_out)

    def expr(self, input_vars=None):
        if input_vars is None:
            input_vars = [f"Pix{i}" for i in range(self.sr_param.n_inputs)]
        return super().expr(input_vars)

    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg


# CGP+PINN
class DiffCGPModel(CGPModel):
    def __init__(self, sr_param: CGPParameter, genes=None, ephs=None) -> None:
        super().__init__(sr_param, genes, ephs)
        self.diff = self.build_diff()

    def build_diff(self):

        hiddens = []
        for _ in range(3):
            hiddens.append(nn.Linear(20, 20))
            hiddens.append(nn.Sigmoid())

        return nn.Sequential(
            nn.Linear(self.sr_param.n_inputs, 20),
            nn.Sigmoid(),
            *hiddens,
            nn.Linear(20, 1)
        )

    def diff_forward(self, X, y=None):
        diff_out = self.diff(X).squeeze(dim=1)
        if y is None:
            return diff_out
        return diff_out, self.diff_loss(diff_out, y)

    def diff_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def side_loss(self, X):
        return torch.mean(self(X) ** 2)


# 带微分项的用于处理图像的CGP模型
class DiffImageCGPModel(DiffCGPModel):
    kernel_size: tuple
    input_channels: int

    def __init__(self, sr_param: CGPParameter, genes=None, ephs=None, kernel_size=(3, 3)) -> None:
        super().__init__(sr_param, genes, ephs)
        self.kernel_size = kernel_size
        self.input_channels = self.sr_param.input_channels

    def build_diff(self):
        return nn.Conv2d(self.input_channels, 1, kernel_size=self.kernel_size),

    def diff_forward(self, X):
        return self.diff(X).squeeze(dim=1)

    def select_features(self, X, idx, expr=False, symbol_constant=False):
        """
          X: list of C (B, H, W) and 1 (B, H-kh+1, W-kw+1) tensors
          out: selected `X_sub`: shape=(B, H-kh+1, W-kw+1)
        """

        var_and_const = self.sr_param.n_inputs + self.sr_param.n_eph
        if expr:
            if self.sr_param.n_inputs <= idx < var_and_const:
                eph_idx = idx - self.sr_param.n_inputs
                c = f"c{eph_idx}" if symbol_constant else self.ephs[eph_idx].item()
            elif idx < self.sr_param.n_inputs:
                c = X[idx]
            else:
                c = X[-1]
            return c

        B, H, W = X[0].shape
        C = len(X) - 1
        kh, kw = self.kernel_size

        Hout, Wout = H - kh + 1, W - kw + 1
        i_row = idx // kw
        i_col = idx - i_row * kw

        if self.sr_param.n_inputs <= idx < var_and_const:
            return torch.empty(
                size=(B, Hout, Wout), dtype=X[0].dtype, device=X[0].device
            ).fill_(self.ephs[idx - self.sr_param.n_inputs])
        elif idx >= var_and_const:
            return X[-1]

        # trick operator
        weight = torch.zeros(
            size=(kh, kw),
            dtype=X.dtype,
            device=X.device
        )
        weight[i_row, i_col] = 1.
        weight = weight.reshape(1, 1, kh, kw).expand(1, C, -1, -1)
        selected = nn.functional.conv2d(
            X, weight, stride=1, padding=0, dilation=1
        ).squeeze(dim=1)
        return selected
