"""
不进行端到端训练，pinn和kan分开训练
"""
import sys

sys.path.append("/Users/lihaoyang/Projects/srn-m/SRNet")
import torch
from torch import nn
from parameters import KANParameter
from usr_models import KAN, KANLinear
from functions import img_dx, img_dy
from utils import pinn_loss, kan_loss  # 要修改loss的计算方式，把原先EQLPDE的loss函数拆成两部分


class PINN(nn.Module):
    def __init__(self,  n_layer, in_features=5, hidden_size=100, pd_lib=['dx', 'dy', 'dxdy']) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.pd_lib = pd_lib
        models = [nn.Linear(self.in_features, self.hidden_size), nn.ReLU()]
        for _ in range(self.n_layer):
            models.append(nn.Linear(self.hidden_size, self.hidden_size))
            models.append(nn.ReLU())
        models.append(nn.Linear(self.hidden_size, 1))
        self.model = nn.Sequential(*models)

    def diff_item(self, u, x, item='x', real_u=False):
        # 如果u是神经网络的预测值，那么就求导
        if not real_u:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # 如果u是真实值，直接对图像进行求导操作
        # now, diff.shape should be (B, C, H, W)
        diff = img_dx(u) if item == 'x' else img_dy(u)
        return diff

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

    def forward(self, x, y, t, dx, dy, u_real=None):
        input_data_list = [x, y, t, dx, dy]
        input_data = torch.stack(input_data_list, dim=-1)
        u_hat = self.model(input_data).squeeze()  # pinn的输出u^hat
        pd_reals = []  # 真实的ux和uy
        pd_hats = []  # 经过pinn后得到的偏微分项ux^hat和uy^hat，之后要被输入进kan中
        for pd_item in self.pd_lib:
            pd_hat = self.build_pde_lib(x, y, t, u_hat, pd_item, real_u=False)
            pd_hats.append(pd_hat)
            if u_real is not None:
                pd_real = self.build_pde_lib(x, y, t, u_real, pd_item, real_u=True)
                pd_reals.append(pd_real)
        # pde_out = self.pde_model(pd_hats)  # kan要做的事
        return {
            "u_hat": u_hat,
            "pd_reals": pd_reals,
            "pd_hats": pd_hats
        }

