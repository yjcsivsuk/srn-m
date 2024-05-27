import sys
sys.path.append("/Users/lihaoyang/Projects/srn-m")

import torch
import random
import numpy as np
from torch import nn
from SRNet.nets import ModuleCGPNet, ModuleEQLNet
from SRNet.parameters import CGPParameter, EQLParameter

n_inputs = 3
n_outputs = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Args:
    n_rows = 2
    n_cols = 2
    levels_back = 4

    n_layers = 2


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=8) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Tanh(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh()
        )
        self.out_proj = nn.Linear(hidden_size, out_features)
        self.out_act = nn.Sigmoid()

    def forward(self, X):
        out1 = self.in_proj(X)
        out2 = self.hidden_proj(out1)
        out3 = self.out_proj(out2)
        return [out1, out2, out3, self.out_act(out3)]


def loss_fn(outputs, targets):
    mse_fn = nn.MSELoss()
    loss = 0.
    n_layer = len(outputs)
    hidden_weight = 1 / (n_layer - 1)
    for o, t in zip(outputs[:-1], targets[:-1]):
        loss += mse_fn(o, t) * hidden_weight
    loss += mse_fn(outputs[-1], targets[-1])
    return loss


def train_srnet(srnet, X, y):
    optimizer = torch.optim.LBFGS(
        srnet.parameters(), lr=1e-2, max_iter=5
    )

    def closure():
        optimizer.zero_grad()
        predicts = srnet(X)
        loss = loss_fn(predicts, y) + 1e-3 * srnet.regularization()
        loss.backward()
        return loss

    for i in range(5):
        optimizer.step(closure)
        current_loss=closure().item()
        print(f"epoch {i}: loss={current_loss}")


# 测试不处理图像的模块CGP，测试结果无误
def test_ModuleCGPNet():
    setup_seed(1)
    X = torch.randn(size=(10, n_inputs))
    args = Args()
    cgp_param = CGPParameter(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_eph=1,
        args=args,
        function_set=None
    )

    neural_network = MLP(n_inputs, n_outputs)
    with torch.no_grad():
        y = neural_network(X)

    cgp_net = ModuleCGPNet(
        cgp_param, neural_network
    )

    cloned_cgpnet = cgp_net.copy_self()
    cloned_cgpnet.assign_genes(cgp_net.get_genes())

    def print_loss(srnet):
        loss, outputs = srnet(X, y)
        print("outputs length:", len(outputs), "loss:", loss.item())
        for i, o in enumerate(outputs):
            ex_layer = srnet.explained_layers[i]
            layer_expressions = ex_layer.expr().tolist()[0]
            # print(f'Layer Expressions {i}:{layer_expressions}')  # 太过复杂，不予显示

    print("***Before SRNet trained***")
    print(cgp_net)
    print_loss(cgp_net)
    # print(f'CGPNet Expression:{cgp_net.expr()}')  # 太过复杂，不予显示
    # print(f'Cloned CGPNet:{cloned_cgpnet.expr()}')  # 太过复杂，不予显示

    print("***After SRNet trained***")
    train_srnet(cloned_cgpnet, X, y)
    print_loss(cloned_cgpnet)
    # print(f'CGPNet Expression:{cgp_net.expr()}')  # 太过复杂，不予显示
    # print(f'Cloned CGPNet:{cloned_cgpnet.expr()}')  # 太过复杂，不予显示

    # 判断原始的cgp_net和克隆得到的cloned_cgpnet其中的参数是否一致，结果是不一致
    # 0,1,2层的ephs，weight，bias都不一致。但是各个参数都很接近
    for np1, np2 in zip(cgp_net.named_parameters(), cloned_cgpnet.named_parameters()):
        n1, p1 = np1
        n2, p2 = np2
        if "neural_network" in n1:
            continue
        print(f"##### p1 and p2: {n1, n2} #####")
        print(p1)
        print(p2)
        print("equals?", p1 == p2)


# 测试EQL网络结构
def test_ModuleEQLNet():
    setup_seed(2)
    X = torch.randn(size=(10, n_inputs))
    args = Args()
    eql_param = EQLParameter(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_eph=0,
        args=args,
        function_set=["add", "mul", "log", "sin", "cos", "sqrt"]
    )

    neural_network = MLP(n_inputs, n_outputs)
    with torch.no_grad():
        y = neural_network(X)

    eql_net = ModuleEQLNet(
        eql_param, neural_network
    )
    print(f'EQLNet:{eql_net}')
    # print(eql_net.expr())  # 不显示EQL的表达式，有问题
    print(f'EQLNet Regularization:{eql_net.regularization()}')
    train_srnet(eql_net,X,y)

