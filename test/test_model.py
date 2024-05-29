import sys

import torch

sys.path.append("/Users/lihaoyang/Projects/srn-m")

import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
from SRNet.parameters import CGPParameter, EQLParameter, KANParameter
from SRNet.sr_models import CGPModel, ImageCGPModel
from SRNet.usr_models import EQL, ImageEQL, DiffCGPModel, DiffMLP, KAN, KANLinear
from SRNet.functions import *
from load_data import load_mnist_data

n_inputs = 3
X = torch.randn(size=(2, n_inputs))


class Args:
    n_rows = 2
    n_cols = 2
    levels_back = 4

    n_diff_cols = 2
    n_layers = 3

    layers_hidden = [3, 3, 1]
    grid_size = 5
    spline_order = 3
    scale_noise = 0.1
    scale_base = 1.0
    scale_spline = 1.0
    base_activation = nn.SiLU
    grid_eps = 0.02
    grid_range = [-1, 1]


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# 测试CGP模型，测试无误
def test_cgp_model():
    setup_seed(1)
    args = Args()
    cgp_param = CGPParameter(
        n_inputs=n_inputs,
        n_outputs=2,
        n_eph=1,
        args=args,
        function_set=None
    )
    cgp = CGPModel(cgp_param)

    print(f'CGP nodes:{cgp.nodes}')
    print(f'CGP active_paths:{cgp.active_paths}')
    for i in range(cgp_param.n_outputs):
        print(f'active paths {i}\'s function operator:{[cgp.nodes[gidx].func for gidx in cgp.active_paths[i]]}')
    print(f'CGP expression:{cgp.expr()}')
    # 经过手动计算器计算，CGP得到的表达式正确，符合输入x和输出y
    print("X:", X)
    y = cgp(X)
    print("y:", y)


# 测试处理图像的CGP模型，测试无误，但不知道怎么分析
def test_sg_cgd_model():
    setup_seed(1)
    train_set, val_set = load_mnist_data("../data/img")
    img = train_set[10][0].reshape(1, 1, 32, 32)
    kernel_size = (3, 3)
    in_channels = 1
    args = Args()
    args.n_cols = 5
    args.n_rows = 5
    args.levels_back = 10
    cgp_param = CGPParameter(
        n_inputs=math.prod(kernel_size),
        n_outputs=1,
        n_eph=1,
        args=args,
        function_set=add_sg_functions
    )
    cgp_param.input_channels = in_channels
    cgp = ImageCGPModel(cgp_param)

    print(f'CGP Expression:{cgp.expr()}')
    with torch.no_grad():
        cgp_out = cgp(img).cpu()
    print(f'CGP Out Shape:{cgp_out.shape}')

    out_dx = img_dx(img)
    out_dy = img_dy(img)

    figure, axis = plt.subplots(1, 4)
    axis[0].imshow(img.squeeze())
    axis[1].imshow(out_dx.squeeze())
    axis[2].imshow(out_dy.squeeze())
    axis[3].imshow(cgp_out.squeeze())
    # plt.savefig("test_sg_cgd_model.png")
    plt.show()
    plt.close()


# 测试带有PINN+CGP
def test_diff_cgp():
    def build_diff():
        hiddens = []
        for _ in range(3):
            hiddens.append(nn.Linear(20, 20))
            hiddens.append(nn.Sigmoid())
        return nn.Sequential(
            nn.Linear(n_inputs, 20),
            nn.Sigmoid(),
            *hiddens,
            nn.Linear(20, 1)
        )

    setup_seed(3)
    args = Args()
    cgp_param = CGPParameter(
        n_inputs=n_inputs,
        n_outputs=2,
        n_eph=1,
        args=args,
        function_set=["add", "sub", "mul", "sin", "diff"],
    )

    cgp = DiffCGPModel(cgp_param)
    diff = build_diff()
    print(f'diff:{diff}')
    diff.load_state_dict(cgp.diff.state_dict())
    for n, p in cgp.named_parameters():
        print(n, p.shape)
    print(f'CGP nodes:{cgp.nodes}')
    for i in range(cgp_param.n_outputs):
        print(f'active paths {i}\'s function operator:{[cgp.nodes[gidx].func for gidx in cgp.active_paths[i]]}')
    print(f'CGP expression:{cgp.expr()}')

    X = torch.randn(size=(2, n_inputs), requires_grad=True)
    print("X:", X)
    y = cgp.diff_forward(X)
    print("y:", y)
    # 这里的y其实就等于u，只不过y是经过sequeeze(dim=1)之后的u
    u = diff(X)
    print("u:", u)
    ux = torch.autograd.grad(u, X, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    print("ux:", ux)


# 测试EQL模型，测试无误，但不知道怎么分析
def test_eql():
    setup_seed(4)
    args = Args()
    args.n_layers = 2
    eql_param = EQLParameter(
        n_inputs=n_inputs,
        n_outputs=2,
        n_eph=0,
        args=args,
        function_set=["add", "mul", "log", "sin", "cos", "sqrt"]
    )
    eql = EQL(eql_param)
    print(f'EQL layers:{eql.layers}')
    print(f'EQL expression:{eql.expr().tolist()[0]}')  # 结果表达式太复杂
    input = torch.randn(size=(3, n_inputs), requires_grad=True)
    print(f'Input:{input},Input Shape:{input.shape}')
    u = eql(input)
    print(f'U:{u},U Shape:{u.shape}')
    ux = torch.autograd.grad(u, input, torch.ones_like(u), retain_graph=True, create_graph=True)[
        0]  # 因为torch.autograd.grad返回的是一个元组，取[0]只保留梯度值，去掉grad_fn梯度信息。
    print(f'Ux:{ux},Ux Shape:{ux.shape}')


# 测试处理图像的EQL模型，测试无误，但不知道怎么分析
def test_img_eql():
    setup_seed(5)
    args = Args()
    args.n_layers = 2

    eql_param = EQLParameter(
        n_inputs=n_inputs,
        n_outputs=2,
        n_eph=0,
        args=args,
        function_set=["add", "mul", "log", "sin", "cos", "sqrt"]
    )
    eql_param.input_channels = n_inputs
    kernel_size = (3, 3)
    eql = ImageEQL(eql_param, kernel_size=kernel_size)
    print(f'ImageEQL layers:{eql.layers}')
    print(f'ImageEQL expression:{eql.expr().tolist()[0]}')
    input = torch.randn(size=(3, 9, 5, 5), requires_grad=True)
    print(f'Input:{input},Input Shape:{input.shape}')  # (3,9,5,5)->(3,9,3,3,2)->(3,3,3,2)->(3,2,3,3)
    u = eql(input)
    print(f'U:{u},U Shape:{u.shape}')  # (3,2,3,3)
    ux = torch.autograd.grad(u, input, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    print(f'Ux:{ux},Ux Shape:{ux.shape}')  # (3,9,5,5)


# 测试PINN模型，在输出对输入求梯度的时候报错
def test_diff_mlp():
    setup_seed(6)
    diff_mlp = DiffMLP(in_features=n_inputs)
    input_data = torch.randn(10, n_inputs, requires_grad=True)
    print(f'Input:{input_data},Input Shape"{input_data.shape}')
    print(f'Model:{diff_mlp}')
    output = diff_mlp(input_data)
    print(f'Output:{output},Output Shape:{output.shape}')
    # 报错
    u_x = torch.autograd.grad(output, input_data[:, 0], torch.ones_like(output), retain_graph=True, create_graph=True)[
        0]
    print(f'u_x:{u_x},u_x Shape:{u_x.shape}')
    u_y = torch.autograd.grad(output, input_data[:, 1], torch.ones_like(output), retain_graph=True, create_graph=True)[
        0]
    print(f'u_y:{u_y},u_y Shape:{u_y.shape}')
    u_t = torch.autograd.grad(output, input_data[:, -1], torch.ones_like(output), retain_graph=True, create_graph=True)[
        0]
    print(f'u_t:{u_t},u_t Shape:{u_t.shape}')


def test_kan():
    setup_seed(7)
    args = Args()
    kan_param = KANParameter(
        n_inputs=args.layers_hidden[0],
        n_outputs=args.layers_hidden[-1],
        n_eph=0,
        args=args,
        function_set=None,
        one_in_one_out=False
    )
    kan = KAN(kan_param)
    print(kan)
    y = kan(X)
    print(y, y.shape)
