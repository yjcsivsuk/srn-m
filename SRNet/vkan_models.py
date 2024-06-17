import torch
import copy
from torch import nn
from SRNet.functions import img_dx, img_dy
from SRNet.parameters import vKANParameter
from SRNet.usr_models import DiffMLP
from vKAN import KAN


class vKANPDE(nn.Module):
    def __init__(self, sr_param: vKANParameter, with_fu=False, pd_lib=['dx', 'dy', 'dxdy']) -> None:
        super().__init__()
        self.sr_param = sr_param
        self.u_model = None
        self.with_fu = with_fu
        if with_fu:
            self.u_model = KAN(sr_param)
        else:
            self.u_model = DiffMLP(
                sr_param.n_inputs + sr_param.n_eph,
                n_layer=sr_param.n_layer
            )

        self.pd_lib = pd_lib
        pde_param = copy.deepcopy(sr_param)
        pde_param.width = [3, 3, 1]
        self.pde_param = pde_param
        self.pde_model = KAN(pde_param)  # 之后进行替换的时候，估计得从这里改？

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
        u_hat = self.u_model(input_data)  # input_data(list:5)
        # else:
        #     assert u_real is not None, "`u_real` should not be None when `with_fu=False`"

        pd_reals = []
        pd_hats = []  # 经过pinn后得到的偏微分项，之后要被输入进kan中
        for pd_item in self.pd_lib:
            pd_hat = self.build_pde_lib(x, y, t, u_hat, pd_item, real_u=False)
            pd_hats.append(pd_hat)  # 这里的pd_hats是一个列表，直接输入进KAN中会报错
            if u_real is not None:
                pd_real = self.build_pde_lib(x, y, t, u_real, pd_item, real_u=True)
                pd_reals.append(pd_real)

        # if self.with_fu:
        pde_out = self.pde_model(pd_hats)  # 将pinn生成的偏微分项输入进kan中
        # else:
        #     pde_out = self.pde_model(pd_reals)
        return {
            "u_hat": u_hat,
            "pd_reals": pd_reals,
            "pd_hats": pd_hats,
            "pde_out": pde_out
        }

    # def expr(self, input_vars=None, sparse_filter=0.01):
    #     return self.pde_model.expr(input_vars, sparse_filter=sparse_filter)
    #
    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.pde_model.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg
