import os
import time
import torch
import numpy as np
import random
import argparse

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_network import neural_networks
from SRNet.usr_models import EQLParameter, EQL, EQLPDE
from load_data import build_image_pde_data, load_mnist_data, build_image_from_pde_data
from utils import get_warmup_linear_scheduler, pde_loss_fn, show_img, get_warmup_scheduler


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
    show_img(sample_images, 2, 2, save_path=os.path.join(args.data_dir, "sample_images.pdf"))

    sample_images = sample_images.unsqueeze(dim=1)
    with torch.no_grad():
        hidden_images = net(sample_images)[layer_idx]

    for i, sample_id in enumerate(sample_ids):
        show_img(hidden_images[i], row, col, f"Real LeNet-5 Hidden {layer_idx}",
                 save_path=os.path.join(args.data_dir, f"real_sample{sample_id}_hidden_{layer_idx}.pdf"))
    return hidden_images


# 个人感觉这个方法有问题，应该直接用PINN+EQL的方法来训练，而不是只用一个EQL
def train_img_pde(args):
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

    # Load the SR
    param = EQLParameter(
        n_inputs=3,  # x, y, t
        n_outputs=1,  # u
        n_eph=0,
        args=args,
        function_set=["add", "mul", "sin", "cos", "square", "log", "cube"]
    )
    SR = EQL(param)

    # Load the dataset
    train_set, val_set = load_mnist_data(args.data_dir)
    # layer_idx = 0
    row, col = 2, 3
    if args.layer_idx == 1:
        row, col = 4, 4
    sample_ids = [1999]
    hidden_images = extract_hideen_images(
        args, net, args.layer_idx, row, col, train_set,
        sample_ids=sample_ids
    )
    sample_size, time_steps, x_steps, y_steps = hidden_images.size()
    print("Hidden Images Shape:", hidden_images.shape)

    # Build the dataset
    input_data, U = build_image_pde_data(
        hidden_images,
        x_range=(-1, 1),
        y_range=(-1, 1),
        t_range=(0, 1)
    )
    input_data = input_data.reshape(-1, input_data.size(-1))
    X, Y, T = input_data[:, 0], input_data[:, 1], input_data[:, 2]
    U = U.reshape(-1)

    print(SR)
    print("Input data shape:", input_data.shape, "U shape:", U.shape)

    # Train the SR
    SR = SR.to(device)
    optimizer = torch.optim.Adam(SR.parameters(), lr=args.lr)
    total_steps = args.epoch
    scheduler = None
    if args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter(tb_dir)

    loss_fn = nn.MSELoss()
    tqbar = tqdm(range(args.epoch), desc="Train SR", total=args.epoch)

    best_loss = float("inf")
    input_data = input_data.to(device)
    U = U.to(device)

    global_steps = 0
    for epoch in tqbar:

        global_steps += 1

        optimizer.zero_grad()

        U_hat = SR(input_data).squeeze()
        loss = loss_fn(U_hat, U)
        if loss < best_loss:
            best_loss = loss
            torch.save(SR.state_dict(), os.path.join(args.out_dir, "SR.pt"))

            if global_steps % args.save_steps == 0:
                with torch.no_grad():
                    U_pred = SR(input_data).squeeze().cpu()
                img_pred = build_image_from_pde_data(U_pred, sample_size, time_steps, x_steps, y_steps)
                for i, sample_id in enumerate(sample_ids):
                    show_img(
                        img_pred[i], row, col,
                        f"Predicted SR Hidden {args.layer_idx}",
                        save_path=os.path.join(
                            args.out_dir,
                            f"pred_sample{sample_id}_hidden_{args.layer_idx}.pdf"
                        )
                    )

        regularization = torch.tensor(0)
        if args.weight_decay > 0:
            regularization = SR.regularization()
            loss += regularization * args.weight_decay

        loss.backward()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(SR.parameters(), args.clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        tqbar.set_postfix(step_loss=loss.item(), step_regularization=regularization.item())
        writer.add_scalar("loss/train_step_loss", loss.item(), epoch)
        writer.add_scalar("loss/train_step_regular", regularization.item(), epoch)


def train_pde_find(args):
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
    param = EQLParameter(
        n_inputs=len(input_data) - 1,  # x, y, t, dx?, dy?
        n_outputs=1,  # u
        n_eph=0,
        args=args,
        function_set=["add", "mul", "sin", "cos", "square", "log"]
    )
    PDE = EQLPDE(param, with_fu=args.with_fu)

    print(PDE)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train the PDE
    PDE = PDE.to(device)
    optimizer = torch.optim.Adam(PDE.parameters(), lr=args.lr)
    total_steps = args.epoch
    scheduler = None
    if args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train PDE Find", total=args.epoch)

    best_loss = float("inf")
    global_steps = 0
    for epoch in tqbar:

        global_steps += 1

        optimizer.zero_grad()

        losses = pde_loss_fn(args, PDE, input_data, U)
        loss = losses["loss"]

        if loss < best_loss:
            best_loss = loss
            torch.save(PDE.state_dict(), os.path.join(args.out_dir, "PDE.pt"))

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

        loss.backward()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(PDE.parameters(), args.clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        regularization = losses["regularization"].item()
        u_loss = losses["u_loss"].item()
        pde_loss = losses["pde_loss"].item()
        pd_loss = losses["pd_loss"].item()

        tqbar.set_postfix(
            step_loss=loss.item(),
            step_regularization=regularization,
            step_u_loss=u_loss,
            step_pde_loss=pde_loss,
            step_pd_loss=pd_loss
        )
        writer.add_scalar("loss/train_step_loss", loss.item(), epoch)
        writer.add_scalar("loss/train_step_regular", regularization, epoch)
        writer.add_scalar("loss/train_step_u_loss", u_loss, epoch)
        writer.add_scalar("loss/train_step_pde_loss", pde_loss, epoch)
        writer.add_scalar("loss/train_step_pd_loss", pd_loss, epoch)


if __name__ == "__main__":

    def boolean_str(s):
        return s == "True"


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/img")
    parser.add_argument("--nn_name", type=str, default="LeNet")
    parser.add_argument("--nn_path", type=str, default="./output/LeNet/LeNet")
    parser.add_argument("--layer_idx", type=int, default=0)

    # EQL PDE Find
    parser.add_argument("--pde_find", type=boolean_str, default="True")
    parser.add_argument("--with_fu", type=boolean_str, default="True")
    parser.add_argument("--pd_weight", type=float, default=1.0)
    parser.add_argument("--pde_weight", type=float, default=1.0)

    # EQL
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=1.0)

    parser.add_argument("--out_dir", type=str, default="./output/find-pde/layer3-test")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="cpu")
    args = parser.parse_args()

    if args.pde_find:
        train_pde_find(args)
    else:
        train_img_pde(args)