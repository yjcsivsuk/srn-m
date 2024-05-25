import torch
from torch import nn


class MLP(nn.Module):
    # cgp: d+5*5+(8+5*5)*4+(16+5*5)*2=157+82=240
    def __init__(self, in_features, out_features, hidden_size=8) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.out_proj = nn.Linear(hidden_size, out_features)

    def forward(self, X, module_out=True):
        if module_out:
            out1 = self.in_proj(X)
            out2 = self.hidden_proj(out1)
            return [out1, out2, self.out_proj(out2)]
        # 保存每一层的输出值，存在outputs列表中
        outputs = []
        out = X
        for layer in self.in_proj:
            out = layer(out)
            if layer.__class__ == nn.Dropout:
                outputs.append(out)
        outputs.append(out)
        for layer in self.hidden_proj:
            out = layer(out)
            if layer.__class__ == nn.ReLU:
                outputs.append(out)
        outputs.append(out)
        out = self.out_proj(out)
        outputs.append(out)
        return outputs

    def train_step(self, X, target, optimizer, clip=False):
        optimizer.zero_grad()
        loss = self.loss(X, target)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)  # 梯度裁剪
        optimizer.step()
        return loss

    def loss(self, X, target):
        out = self(X)[-1]
        loss = nn.MSELoss()(out.squeeze(), target.squeeze())
        return loss


# PINN模型定义
class DiffMLP(nn.Module):
    def __init__(self, in_features, n_layer=5, hidden_size=20) -> None:
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
        return self.model(X).squeeze()  # 用于移除tensor中最后一个大小为1的维度


if __name__ == '__main__':

    # 测试mlp模型
    # mlp_model = MLP(in_features=2,out_features=1)
    # print(mlp_model)
    # inputs = torch.randn(3,2)
    # targets = torch.randn(3,2)
    # outputs = mlp_model(inputs)
    # print([outs.shape for outs in outputs])

    # 测试pinn模型
    pinn_model = DiffMLP(in_features=2)
    print(pinn_model)
    inputs= torch.randn(3,2)
    outputs = pinn_model(inputs)
    print(outputs.shape)
