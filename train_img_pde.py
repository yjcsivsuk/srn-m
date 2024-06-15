
import os
import time
import torch
import numpy as np
import random
import argparse

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SRNet.parameters import eKANParameter
from neural_network import neural_networks
from SRNet.usr_models import EQLParameter, EQL, EQLPDE, eKANPDE, eKAN, eKANeKAN
from load_data import build_image_pde_data, load_mnist_data, build_image_from_pde_data
from utils import pde_loss_fn, show_img, get_warmup_scheduler


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
    # show_img(sample_images, 2, 2, save_path=os.path.join(args.data_dir, "sample_images.pdf"))

    sample_images = sample_images.unsqueeze(dim=1)
    with torch.no_grad():
        hidden_images = net(sample_images)[layer_idx]

    # for i, sample_id in enumerate(sample_ids):
    #     show_img(hidden_images[i], row, col, f"Real LeNet-5 Hidden {layer_idx}",
    #              save_path=os.path.join(args.data_dir, f"real_sample{sample_id}_hidden_{layer_idx}.pdf"))
    return hidden_images


# 这里只用了一个EQL。论文中提到，为了与卷积模块符号层保持一致，在解释卷积神经网络时，其全连接模块符号层中的通用公式也使用EQL来进行搜索优化，从而方便后续整体微调能够保持端到端训练。
# 但是总觉得不太对，没明白这一步是在干什么。省略掉pinn，直接用eql？
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
        function_set=["add", "mul", "sin", "cos", "square", "log"]
    )
    SR = EQL(param)

    # Load the dataset
    train_set, val_set = load_mnist_data(args.data_dir)
    # layer_idx = 0
    row, col = 2, 3
    if args.layer_idx == 1:
        row, col = 4, 4
    sample_ids = [1999, 1999, 1999, 1999]
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


def train_img_pde_with_kan(args):
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
    param = eKANParameter(
        n_inputs=3,  # x, y, t
        n_outputs=1,  # u
        n_eph=0,
        args=args,
        function_set=None
    )
    SR = eKAN(param)

    # Load the dataset
    train_set, val_set = load_mnist_data(args.data_dir)
    # layer_idx = 0
    row, col = 2, 3
    if args.layer_idx == 1:
        row, col = 4, 4
    sample_ids = [1999, 1999, 1999, 1999]
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
    writer = SummaryWriter(tb_dir)
    loss_fn = nn.MSELoss()
    tqbar = tqdm(range(args.epoch), desc="Train SR", total=args.epoch)
    input_data = input_data.to(device)
    U = U.to(device)
    best_loss = float("inf")
    global_steps = 0
    for epoch in tqbar:
        def closure():
            global loss, regularization
            optimizer.zero_grad()
            U_hat = SR(input_data).squeeze()
            loss = loss_fn(U_hat, U)
            regularization = SR.regularization_loss()
            if args.weight_decay > 0:
                loss += args.weight_decay * regularization
            loss.backward()
            return loss

        optimizer.step(closure)
        global_steps += 1
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


def train_pde_find_with_kan(args):
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
    param = eKANParameter(
        n_inputs=len(input_data) - 1,  # x, y, t, dx?, dy?
        n_outputs=1,  # u
        n_eph=0,
        args=args
    )
    PDE = eKANPDE(param, with_fu=args.with_fu)

    print(PDE)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train the PDE
    PDE = PDE.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(PDE.parameters(), lr=args.lr, line_search_fn="strong_wolfe")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(PDE.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(PDE.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train PDE Find", total=args.epoch)
    best_loss = float("inf")
    global_steps = 0
    for epoch in tqbar:
        def closure():
            global losses, loss
            optimizer.zero_grad()
            losses = pde_loss_fn(args, PDE, input_data, U)  # KANPDE运行到这，然后会报错
            loss = losses["loss"]
            loss.backward()
            # 这里梯度裁剪应该不起作用，估计可以删掉
            # if args.clip_norm > 0:
            #     torch.nn.utils.clip_grad.clip_grad_norm_(PDE.parameters(), args.clip_norm)
            return loss

        optimizer.step(closure)
        global_steps += 1
        if loss.item() < best_loss:
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


def train_pde_find_with_kan_without_sobel(args):
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
        add_dx=False,
        add_dy=False
    )
    input_data = input_data.reshape(-1, input_data.size(-1))
    X, Y, T= map(lambda i: input_data[:, i].to(device), range(3))
    dX, dY = None, None
    X = X.requires_grad_(True)
    Y = Y.requires_grad_(True)
    if args.with_fu:
        T = T.requires_grad_(True)

    U = U.reshape(sample_size, time_steps, x_steps, y_steps).to(device)
    input_data = [X, Y, T, dX, dY, U]

    # Load the SR
    param = eKANParameter(
        n_inputs=3,  # x, y, t
        n_outputs=1,  # u
        n_eph=0,
        args=args
    )
    PDE = eKANPDE(param, with_fu=args.with_fu)

    print(PDE)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train the PDE
    PDE = PDE.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(PDE.parameters(), lr=args.lr, line_search_fn="strong_wolfe")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(PDE.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(PDE.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train PDE Find", total=args.epoch)
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

def train_pde_find_only_with_kan(args):
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
    param = eKANParameter(
        n_inputs=len(input_data) - 1,  # x, y, t, dx?, dy?
        n_outputs=1,  # u
        n_eph=0,
        args=args
    )
    PDE = eKANeKAN(param, with_fu=args.with_fu)

    print(PDE)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train the PDE
    PDE = PDE.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(PDE.parameters(), lr=args.lr, line_search_fn="strong_wolfe")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(PDE.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(PDE.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train PDE Find", total=args.epoch)
    best_loss = float("inf")
    global_steps = 0
    for epoch in tqbar:
        def closure():
            global losses, loss
            optimizer.zero_grad()
            losses = pde_loss_fn(args, PDE, input_data, U)  # KANPDE运行到这，然后会报错
            loss = losses["loss"]
            loss.backward()
            # 这里梯度裁剪应该不起作用，估计可以删掉
            # if args.clip_norm > 0:
            #     torch.nn.utils.clip_grad.clip_grad_norm_(PDE.parameters(), args.clip_norm)
            return loss

        optimizer.step(closure)
        global_steps += 1
        if loss.item() < best_loss:
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
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "LBFGS", "AdamW"])

    # EQL PDE Find
    parser.add_argument("--pde_find", type=boolean_str, default="False")
    parser.add_argument("--with_fu", type=boolean_str, default="False")
    parser.add_argument("--pd_weight", type=float, default=1.0)
    parser.add_argument("--pde_weight", type=float, default=1.0)

    # EQL
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--clip_norm", type=float, default=1.0)

    # KAN PDE Find
    parser.add_argument("--pde_find_with_kan_without_sobel", type=boolean_str, default="False")
    parser.add_argument("--pde_find_with_kan", type=boolean_str, default="True")
    parser.add_argument("--pde_find_only_with_kan", type=boolean_str, default="False")
    parser.add_argument("--img_pde_find_with_kan", type=boolean_str, default="False")
    parser.add_argument("--layers_hidden", type=list, default=[3, 3, 1])
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--spline_order", type=int, default=3)
    parser.add_argument("--scale_noise", type=float, default=0.1)
    parser.add_argument("--scale_base", type=float, default=1.0)
    parser.add_argument("--scale_spline", type=float, default=1.0)
    parser.add_argument("--base_activation", type=str, default=nn.SiLU)
    parser.add_argument("--grid_eps", type=float, default=0.02)
    parser.add_argument("--grid_range", type=list, default=[-1, 1])

    parser.add_argument("--out_dir", type=str, default="./output/find_pde_with_kan_without_sobel/test")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="cpu")
    args = parser.parse_args()
    if args.pde_find_with_kan_without_sobel:
        train_pde_find_with_kan_without_sobel(args)
    elif args.pde_find_only_with_kan:
        train_pde_find_only_with_kan(args)
    elif args.img_pde_find_with_kan:
        train_img_pde_with_kan(args)
    elif args.pde_find_with_kan:
        train_pde_find_with_kan(args)
    elif args.pde_find:
        train_pde_find(args)
    else:
        train_img_pde(args)
