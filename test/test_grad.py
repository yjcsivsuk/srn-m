import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.model = nn.Linear(in_features, out_features)

    def forward(self, X):
        return self.model(X)


class SparseNet(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.model = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X):
        return self.model(X)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=10) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, X):
        outputs = []
        for layer in self.model:
            X = layer(X)
            outputs.append(X)
        return outputs


def test_grad():
    in_features = 3
    out_features = 1

    X = torch.randn(size=(10, in_features), requires_grad=True)
    net = Net(in_features=in_features, out_features=out_features)
    sparse_net = SparseNet(in_features=in_features, out_features=out_features)
    # check no_grad: this cannot be done!
    # with torch.no_grad():
    #     y = net(X)

    # check weight grad
    y = net(X)
    X_grad = torch.autograd.grad(y, X, torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    print("X grad:", X_grad)
    print("net weight grad:", net.model.weight.grad)
    print("sparse net weight grad:", sparse_net.model.weight.grad)

    loss = sparse_net(X_grad).mean()
    loss.backward()

    print("after backward, weight grad:", net.model.weight.grad)
    print("after backward, sparse net weight grad:", sparse_net.model.weight.grad)


def test_MLP_grad():
    in_features = 3
    out_features = 1

    X = torch.randn(size=(10, in_features), requires_grad=True)
    net = MLP(in_features=in_features, out_features=out_features)

    outputs = net(X)
    out_x_grad = torch.autograd.grad(outputs[-1], X, torch.ones_like(outputs[-1]), retain_graph=True, create_graph=True)[0]
    grad_list = [output.retain_grad for output in outputs]
    print("grad list:", grad_list)
    print("outputs:", outputs)
    print("out_x_grad:", out_x_grad)
    print("X.grad", X.grad)
    print("outputs[0].grad", outputs[0].retain_grad)
    print("outputs[1].grad:", outputs[1].retain_grad)
    print("outputs[2].grad:", outputs[2].retain_grad)

