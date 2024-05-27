import sys

sys.path.append("/Users/lihaoyang/Projects/srn-m")
import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from neural_network import neural_networks
from SRNet.parameters import CGPParameter, EQLParameter
from SRNet.nets import ModuleCGPNet, ModuleEQLNet


class Args:
    n_rows = 5
    n_cols = 5
    levels_back = 4
    n_diff_cols = 2
    n_layers = 2


conv_net = neural_networks["testConv"]()
in_shape = (1, 32, 32)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_icgp():
    setup_seed(1)
    args = Args()
    sr_param = CGPParameter(
        n_inputs=1,
        n_outputs=16,
        n_eph=1,
        args=args
    )
    ic_srnet = ModuleCGPNet(
        sr_param, conv_net, in_shape
    )
    print(f'ImageCGP SRNet:{ic_srnet}')

    for i, layer in enumerate(ic_srnet.explained_layers):
        print(f'Layer-{i} Expression:{layer.expr()}, Expression Length:{len(layer.expr())}')
    print(f'ImageCGP SRNet Expression:{ic_srnet.expr()}')

    img = torch.randn(16, 1, 32, 32)
    predicts = ic_srnet(img)
    for o in predicts:
        print(f'Predict Shape:{o.shape}')

    outputs = conv_net(img, module_out=True)
    for o in outputs:
        print(f'Output Shape:{o.shape}')

    def select_features(X, idx, expr=False, symbol_constant=False):
        """ X: shape=(B, C, H, W) """
        C, H, W = X.shape[1:]
        kh, kw = 3, 3

        Hout, Wout = H - kh + 1, W - kw + 1
        i_row = idx // kw
        i_col = idx - i_row * kw
        # trick
        weight = torch.zeros(
            size=(kh, kw),
            dtype=X.dtype,
            device=X.device
        )
        weight[i_row, i_col] = 1.
        print(weight)
        weight = weight.reshape(1, 1, kh, kw).expand(C, C, -1, -1)
        selected = F.conv2d(
            X, weight, stride=1, padding=0, dilation=1
        )
        return selected

    X = torch.randn(1, 2, 4, 4)
    proj = nn.Linear(2, 6)
    out = proj(X.reshape(1, 4, 4, 2))
    print(out.shape)
    idx = 5
    print(X)
    print(select_features(X, idx))


def test_ieql():
    setup_seed(2)

    def print_expr(srnet):
        for i, layer in enumerate(srnet.explained_layers):
            print(f"Layer {i}:", layer.expr().tolist()[0])

    args = Args()
    sr_param = EQLParameter(
        n_inputs=1, n_outputs=10,
        n_eph=0, args=args,
        function_set=["add", "mul", "log", "sin", "cos", "sqrt", "square"]
    )
    ic_srnet = ModuleEQLNet(
        sr_param, conv_net, in_shape
    )
    # print_expr(ic_srnet)
    print(ic_srnet)

    img = torch.randn(16, 1, 32, 32)
    predicts = ic_srnet(img)
    for o in predicts:
        print(f'Predict Shape:{o.shape}')

    outputs = conv_net(img, module_out=True)
    for o in outputs:
        print(f'Output Shape:{o.shape}')
