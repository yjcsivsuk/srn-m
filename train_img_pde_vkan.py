import os
import logging
import time
import torch
import random
import argparse
import numpy as np
import sympy as sp

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from SRNet.parameters import vKANParameter
from SRNet.vkan_models import vKANPDE
from load_data import load_mnist_data, build_image_pde_data, build_image_from_pde_data
from neural_network import neural_networks
from utils import pde_loss_fn, show_img


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_hideen_images(
        args,
        net,
        layer_idx,
        row, col,
        train_set,
        sample_ids=[1999, 213, 3456, 92]
):
    sample_ids = torch.tensor(sample_ids, dtype=torch.long)
    sample_images = torch.cat([
        train_set[i][0] for i in sample_ids
    ], dim=0)
    sample_images = sample_images.unsqueeze(dim=1)
    with torch.no_grad():
        hidden_images = net(sample_images)[layer_idx]
    return hidden_images


def train_pde_find_with_vkan(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb_dir = os.path.join(args.out_dir, "tb", cur_time)
    setup_seed(args.seed)

    # device
    device = torch.device(args.gpu)

    # loading trained neural network
    nn_class = neural_networks[args.nn_name]
    net = nn_class(10)
    net.load_state_dict(
        torch.load(args.nn_path, map_location=device)
    )
    net.eval()

    # Load the dataset
    train_set, val_set = load_mnist_data(args.data_dir)
    # layer_idx = 0
    row, col = 2, 3
    if args.layer_idx == 1:
        row, col = 4, 4
    sample_ids = [1999, 213, 3456, 92]
    hidden_images = extract_hideen_images(
        args, net, args.layer_idx, row, col, train_set,
        sample_ids=sample_ids
    )
    sample_size, time_steps, x_steps, y_steps = hidden_images.size()
    print("Hidden Images Shape:", hidden_images.shape)

    # Build the dataset
    # input_data.shape=(n_samples, flat_size, (3+add_dx+add_dy))
    input_data, U = build_image_pde_data(
        hidden_images,
        x_range=(-1, 1),
        y_range=(-1, 1),
        t_range=(0, 1),
        add_dx=True,
        add_dy=True
    )
    input_data = input_data.reshape(-1, input_data.size(-1))
    X, Y, T, dX, dY = map(lambda i: input_data[:, i].to(device), range(5))
    X = X.requires_grad_(True)
    Y = Y.requires_grad_(True)
    if args.with_fu:
        T = T.requires_grad_(True)

    U = U.reshape(sample_size, time_steps, x_steps, y_steps).to(device)
    input_data = [X, Y, T, dX, dY, U]

    # Load the SR
    param = vKANParameter(
        n_inputs=len(input_data) - 1,  # x, y, t, dx?, dy?
        n_outputs=1,  # u
        n_eph=0,
        args=args,
        function_set=None,
        one_in_one_out=False
    )
    PDE = vKANPDE(param, with_fu=args.with_fu)

    print(PDE)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train the PDE
    PDE = PDE.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(PDE.parameters(), lr=args.lr, history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-32, tolerance_change=1e-32)
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(PDE.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(PDE.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train vKANPDE", total=args.epoch)
    best_loss = float("inf")
    global_steps = 0
    for epoch in tqbar:
        def closure():
            global losses, loss
            optimizer.zero_grad()
            losses = pde_loss_fn(args, PDE, input_data, U)
            loss = losses["loss"]
            loss.backward()
            return loss

        optimizer.step(closure)
        global_steps += 1
        if loss.item() < best_loss:
            best_loss = loss
            torch.save(PDE.state_dict(), os.path.join(args.out_dir, "vKANPDE.pt"))
            if global_steps % args.save_steps == 0:
                U_pred = PDE(*input_data)["u_hat"].squeeze().cpu()
                img_pred = build_image_from_pde_data(U_pred, sample_size, time_steps, x_steps, y_steps)
                for i, sample_id in enumerate(sample_ids):
                    show_img(
                        img_pred[i], row, col,
                        f"Predicted U Hidden {args.layer_idx}",
                        save_path=os.path.join(
                            args.out_dir,
                            f"pred_sample{sample_id}_hidden_{args.layer_idx}.pdf"
                        )
                    )

        u_loss = losses["u_loss"].item()
        pd_loss = losses["pd_loss"].item()
        pde_loss = losses["pde_loss"].item()
        regularization = losses["regularization"].item()

        tqbar.set_postfix(
            step_loss=loss.item(),
            step_u_loss=u_loss,
            step_pd_loss=pd_loss,
            step_pde_loss=pde_loss,
            step_regularization=regularization
        )
        writer.add_scalar("loss/train_step_loss", loss.item(), epoch)
        writer.add_scalar("loss/train_step_u_loss", u_loss, epoch)
        writer.add_scalar("loss/train_step_pd_loss", pd_loss, epoch)
        writer.add_scalar("loss/train_step_pde_loss", pde_loss, epoch)
        writer.add_scalar("loss/train_step_regular", regularization, epoch)

    print("Train Over. The Expression is Below")
    expression = PDE.expr()
    expression = sp.sympify(expression)
    print(expression)


if __name__ == "__main__":
    def boolean_str(s):
        return s == "True"
    parser = argparse.ArgumentParser()

    # 要解释的卷积神经网络LeNet
    parser.add_argument("--data_dir", type=str, default="./data/img")
    parser.add_argument("--nn_name", type=str, default="LeNet")
    parser.add_argument("--nn_path", type=str, default="./output/LeNet/LeNet")
    parser.add_argument("--layer_idx", type=int, default=0)
    # PINN+KAN端到端训练相关参数
    parser.add_argument("--with_fu", type=boolean_str, default="False")
    parser.add_argument("--pd_weight", type=float, default=1.0)
    parser.add_argument("--pde_weight", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "LBFGS", "AdamW"])
    # KAN模型相关参数
    parser.add_argument("--pde_find_with_vkan", type=boolean_str, default="True")
    parser.add_argument("--n_layer", type=int, default=5)  # pinn隐藏层的数量
    parser.add_argument("--function_set", type=str, default=['sin', 'cos', 'x', 'x^2', 'log'])
    parser.add_argument("--width", type=str, default=[3, 3, 1])
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--noise_scale_base", type=float, default=0.1)
    parser.add_argument("--base_fun", type=str, default=torch.nn.SiLU())
    parser.add_argument("--symbolic_enabled", type=boolean_str, default="True")
    parser.add_argument("--bias_trainable", type=boolean_str, default="True")
    parser.add_argument("--grid_eps", type=float, default=0.02)
    parser.add_argument("--grid_range", type=str, default=[-1, 1])
    parser.add_argument("--sp_trainable", type=boolean_str, default="True")
    parser.add_argument("--sb_trainable", type=boolean_str, default="True")
    parser.add_argument("--device", type=str, default="cpu")  # kan中的设备
    # 训练结果相关参数
    parser.add_argument("--out_dir", type=str, default="./output/find-pde_with_vkan/test")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="cpu")  # 全局的设备
    args = parser.parse_args()
    if args.pde_find_with_vkan:
        train_pde_find_with_vkan(args)
