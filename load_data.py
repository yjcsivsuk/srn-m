import torch
import os
import numpy as np
import kornia as K
import torchvision
import torchvision.transforms as transforms
import scipy.io as scio
from typing import Tuple
from torch.utils.data.dataset import Dataset, TensorDataset


class SRDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return self.data.size(0)


def load_pmlb_data(path, train=0.7, val=0.2, test=0.1) -> Tuple[TensorDataset, ...]:
    data = torch.from_numpy(np.loadtxt(path)).float()
    B = data.size(0)
    train_len = int(B * train)
    val_len = int(B * val)
    test_len = B - train_len - val_len

    idxs = list(range(data.size(0)))
    np.random.shuffle(idxs)
    idxs = torch.LongTensor(idxs)
    train_idxs = idxs[:train_len]
    val_idxs = idxs[train_len:train_len + val_len]
    test_idxs = idxs[-test_len:]

    train_data = data[train_idxs]
    val_data = data[val_idxs]
    test_data = data[test_idxs]

    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, -1])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, -1])
    test_dataset = TensorDataset(test_data[:, :-1], test_data[:, -1])

    return train_dataset, val_dataset, test_dataset


def load_mnist_data(path, train=0.7, val=0.2, test=0.1) -> Tuple[Dataset, Dataset]:
    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.MNIST(
        root=path,
        train=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=path,
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,))
        ]),
        download=True
    )

    return train_dataset, test_dataset


def load_sr_data(path, train=0.7, val=0.3) -> Tuple[SRDataset, SRDataset]:
    data_dir = "/".join(path.split("/")[:-1])
    data_name = path.split("/")[-1].split(".")[0]

    train_path = os.path.join(data_dir, f"{data_name}_train.txt")
    val_path = os.path.join(data_dir, f"{data_name}_val.txt")

    # 检查文件是否存在
    if os.path.exists(train_path) and os.path.exists(val_path):
        # 加载已存在的文件
        train_data = torch.from_numpy(np.loadtxt(train_path)).float()
        val_data = torch.from_numpy(np.loadtxt(val_path)).float()
    else:
        # 加载数据集文件并分割
        data = torch.from_numpy(np.loadtxt(path)).float()
        np.random.seed(0)  # 确保每次分割结果相同
        split_idx = int(len(data) * train)  # 根据比例分割
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # 将分割后的数据保存到文件
        np.savetxt(train_path, train_data.numpy())
        np.savetxt(val_path, val_data.numpy())

    train_dataset = SRDataset(train_data)
    val_dataset = SRDataset(val_data)

    return train_dataset, val_dataset


def load_pde_data(problem="burgers"):
    data = scio.loadmat('./data/burgers.mat')

    u = data.get("usol")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t").reshape(1, 100))

    # print(u.shape, x.shape, t.shape)
    # shape of (n_x, n_t)
    n, m = u.shape
    x = np.tile(x, (m, 1)).transpose((1, 0))
    t = np.tile(t, (n, 1))
    # print(x.shape, t.shape)

    x = x.flatten()
    t = t.flatten()
    u = u.flatten()
    data = torch.from_numpy(np.stack([x, t, u], axis=1)).float()

    train_dataset = SRDataset(data)
    return train_dataset


# 将图片数据转换为x,y,t，得到的三维数据将被输入进PINN里去
def build_image_pde_data(images, x_range=(-1, 1), y_range=(-1, 1), t_range=(0, 1), add_dx=False, add_dy=False):
    """
        For input image datas: (B, C_in, H, W),
        build them as grid PDE data i.e. construct the grid (x, y, t)
        if add_dx=True, add construct grid (x, y, t, dx)
        if add_dy=True, add construct grid (x, y, t, dy)
        if both are True, construct grid (x, y, t, dx, dy)
    """
    B, C, H, W = images.shape

    X = torch.linspace(x_range[0], x_range[1], H)
    Y = torch.linspace(y_range[0], y_range[1], W)
    T = torch.linspace(t_range[0], t_range[1], C)

    # shape=(H, W)
    X, Y = torch.meshgrid(X, Y)  # 这里把第二个参数indexing='xy'（以列优先）删掉之后不知道会不会有影响，变为默认的以行优先了。所以在下面加了转置。
    X = X.T
    Y = Y.T
    # shape=(CxHxW,)
    X = X.repeat(C, 1, 1).flatten()
    Y = Y.repeat(C, 1, 1).flatten()
    T = T.reshape(C, 1).repeat(1, H * W).flatten()

    input_data = [X, Y, T]
    input_data = torch.stack(input_data, dim=1)
    # shape=(B, CxHxW, 3)
    input_data = input_data.repeat(B, 1, 1)
    # print(input_data.shape)
    # 向pde数据中添加sober算子，为了后续PINN能区分每个样本
    if add_dx:
        dX = K.filters.spatial_gradient(images, normalized=False)[:, :, 0]
        # print(dX.shape)
        input_data = torch.cat([input_data, dX.reshape(B, -1, 1)], dim=-1)
    if add_dy:
        dY = K.filters.spatial_gradient(images, normalized=False)[:, :, 1]
        input_data = torch.cat([input_data, dY.reshape(B, -1, 1)], dim=-1)

    U = images.reshape(B, -1)
    return input_data, U


def build_image_from_pde_data(U, sample_size, time_steps, x_steps, y_steps):
    """
        For input grid PDE data, build them as image datas: (B, C_in, H, W)
    """
    B, C, H, W = sample_size, time_steps, x_steps, y_steps
    U = U.reshape(B, C, H, W)
    return U


if __name__ == "__main__":
    # 测试将图片和pde数据的互相转换，测试结果没有问题
    # from utils import show_img

    # images = torch.randn(2, 10, 4, 4)
    # show_img(images[0], 2, 5, save_path="./test/img/u_before.pdf")
    # input_data, U = build_image_pde_data(images, add_dx=True, add_dy=True)
    # print(f'Input Data:{input_data}')
    # print(f'U:{U}')
    # print(f'Input Data Shape:{input_data.shape},U Shape:{U.shape}')

    # input_data = input_data.reshape(-1, 4)
    # U = U.reshape(-1)
    # U = build_image_from_pde_data(U, 2, 10, 4, 4)

    # show_img(U[0], 2, 5, save_path="./test/img/u_after.pdf")

    # 测试burgers数据集，结果没有问题
    Train = load_pde_data()
    print(Train.X, Train.y)
