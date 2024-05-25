import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    # 使其能够像列表或字典一样使用索引来获取元素
    def __getitem__(self, index):
        if index == 0:
            return self.conv1
        elif index == 1:
            return self.conv2
        elif index == 2:
            return self.fc
        else:
            raise ValueError("Index out of range")

    def forward(self, x, module_out=True):
        # 是否保留每层的输出
        if module_out:
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            out3 = self.fc(out2.reshape(out2.size(0), -1))
            return [out1, out2, out3]

        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)  # 卷积到全连接之间需要展平
        out = self.fc(out)
        return out

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
        loss = nn.CrossEntropyLoss()(out, target)
        return loss

    # 存在问题，暂时不知道如何解决
    def count_correct(self, X, target):
        logits = self(X)[-1]
        predicts = torch.argmax(logits, -1)  # 每个样本在最后一个维度上的最大值的索引。在分类问题中，这通常代表模型对于每个类别的预测得分最高的类别
        n_correct = torch.sum(predicts == target)
        return n_correct


# LeNet-5中的卷积模块，不包括全连接模块
class TestConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x, module_out=True):
        if module_out:
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            return [out1, out2]

        out = self.conv1(x)
        out = self.conv2(out)
        return out


# 卷积输出形状计算公式：(W−F+2P)/S+1
if __name__ == '__main__':
    lenet = LeNet5(num_classes=10)
    inputs = torch.randn(1, 1, 32, 32)
    outputs = lenet(inputs)
    for i in range(3):
        print(lenet[i])
        print(f'output shape:{outputs[i].shape}')
