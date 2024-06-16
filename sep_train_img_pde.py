import os
import time
import torch
import argparse
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SRNet.img_models import PINN
from SRNet.parameters import eKANParameter
from SRNet.usr_models import eKAN
from neural_network import neural_networks
from load_data import build_image_pde_data, load_mnist_data, build_image_from_pde_data
from utils import show_img, pinn_loss, kan_loss
from train_img_pde import setup_seed, extract_hideen_images


# 单独训练pinn，得到的PDTs存在相应输出目录下的PDTs.pt中，测试没问题
def train_pinn(args):
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

    U = U.reshape(sample_size, time_steps, x_steps, y_steps).to(device)
    input_data = [X, Y, T, dX, dY, U]

    # Load the PINN
    ImagePINN = PINN(n_layer=args.n_layer)

    print(ImagePINN)
    print("Input data shape:", len(input_data), X.shape, "U shape:", U.shape)

    # Train PINN and get PDTs
    ImagePINN = ImagePINN.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(ImagePINN.parameters(), lr=args.lr, line_search_fn="strong_wolfe")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(ImagePINN.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(ImagePINN.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train PINN", total=args.epoch)
    best_loss = float("inf")
    global_steps = 0
    p_losses = []
    u_losses = []
    pd_losses = []
    for epoch in tqbar:
        def closure():
            global losses, loss
            optimizer.zero_grad()
            losses = pinn_loss(args, ImagePINN, input_data, U)
            loss = losses["p_loss"]
            loss.backward()
            return loss

        optimizer.step(closure)
        global_steps += 1
        if loss.item() < best_loss:
            best_loss = loss
            torch.save(ImagePINN.state_dict(), os.path.join(args.out_dir, "PINN.pt"))

        u_loss = losses["u_loss"].item()
        pd_loss = losses["pd_loss"].item()
        pd_hats = losses["pd_hats"]

        torch.save(pd_hats, os.path.join(args.out_dir, "PDTs.pt"))

        p_losses.append(loss.item())
        u_losses.append(u_loss)
        pd_losses.append(pd_loss)

        line1, = plt.plot(p_losses, "r-", label="p_loss")
        line2, = plt.plot(u_losses, "y-", label="u_loss")
        line3, = plt.plot(pd_losses, "b-", label="pd_loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend([line1, line2, line3], ["p_loss", "u_loss", "pd_loss"], loc="upper right")
        plt.savefig(os.path.join(args.out_dir, "PINN_Loss.png"))

        tqbar.set_postfix(
            step_p_loss=loss.item(),
            step_u_loss=u_loss,
            step_pd_loss=pd_loss
        )
        writer.add_scalar("loss/train_step_p_loss", loss.item(), epoch)
        writer.add_scalar("loss/train_step_u_loss", u_loss, epoch)
        writer.add_scalar("loss/train_step_pd_loss", pd_loss, epoch)


def train_kan(args):
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
    input_data = torch.load(os.path.join(args.pinn_path, "PDTs.pt"), map_location=device)

    # Load the KAN
    kan_param = eKANParameter(
        n_inputs=3,  # dx, dy, dxdy
        n_outputs=1,  # f(·)=0
        n_eph=0,
        args=args,
        function_set=None
    )
    ImageKAN = eKAN(args)

    print(ImageKAN)
    print(f"Input PDTs length:{len(input_data)}, shape:{input_data[0].shape}")

    # Train KAN and get PDE data, then get image
    ImagePINN = ImageKAN.to(device)
    optimizer = None
    if args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(ImagePINN.parameters(), lr=args.lr, line_search_fn="strong_wolfe")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(ImagePINN.parameters(), lr=args.lr)
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(ImagePINN.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    tqbar = tqdm(range(args.epoch), desc="Train KAN", total=args.epoch)
    best_loss = float("inf")
    global_steps = 0
    k_losses = []
    pde_losses = []
    reg = []
    for epoch in tqbar:
        def closure():
            global losses, loss
            optimizer.zero_grad()
            losses = kan_loss(args, ImageKAN, input_data)
            loss = losses["k_loss"]
            loss.backward()
            return loss

        optimizer.step(closure)
        global_steps += 1
        if loss.item() < best_loss:
            best_loss = loss
            torch.save(ImageKAN.state_dict(), os.path.join(args.out_dir, "KAN.pt"))
            if global_steps % args.save_steps == 0:
                U_pred = ImageKAN(input_data).squeeze().cpu()
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

        pde_loss = losses["pde_loss"].item()
        regularizations = losses["regularization"].item()

        k_losses.append(loss.item())
        pde_losses.append(pde_loss)
        reg.append(regularizations)

        line1, = plt.plot(k_losses, "r-", label="k_loss")
        line2, = plt.plot(pde_losses, "y-", label="pde_loss")
        line3, = plt.plot(reg, "b-", label="reg")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend([line1, line2, line3], ["k_loss", "pde_loss", "reg"], loc="upper right")
        plt.savefig(os.path.join(args.out_dir, "KAN_Loss.png"))

        tqbar.set_postfix(
            step_k_loss=loss.item(),
            step_pde_loss=pde_loss,
            step_regularizations=regularizations
        )
        writer.add_scalar("loss/train_step_p_loss", loss.item(), epoch)
        writer.add_scalar("loss/train_step_pde_loss", pde_loss, epoch)
        writer.add_scalar("loss/train_step_regularizations", regularizations, epoch)


if __name__ == "__main__":
    def boolean_str(s):
        return s == "True"


    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--data_dir", type=str, default="./data/img")
    parser.add_argument("--nn_name", type=str, default="LeNet")
    parser.add_argument("--nn_path", type=str, default="./output/LeNet/LeNet")
    parser.add_argument("--out_dir", type=str, default="./output/sep-find-pde/test")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--optim", type=str, default="LBFGS", choices=["LBFGS", "Adam", "AdamW"])
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="cpu")

    # train pinn
    parser.add_argument("--pinn_train", type=boolean_str, default="False")
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--pd_weight", type=float, default=1.0)

    # train kan
    parser.add_argument("--kan_train", type=boolean_str, default="True")
    parser.add_argument("--pinn_path", type=str, default="./output/sep-find-pde/test")
    parser.add_argument("--n_layers", type=int, default=5)  # pinn隐藏层的数量，在KANParameter中定义的，跟上面的n_layer参数一样，这里不定义会报错
    parser.add_argument("--layers_hidden", type=list, default=[3, 3, 1])
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--spline_order", type=int, default=3)
    parser.add_argument("--scale_noise", type=float, default=0.1)
    parser.add_argument("--scale_base", type=float, default=1.0)
    parser.add_argument("--scale_spline", type=float, default=1.0)
    parser.add_argument("--base_activation", type=str, default=nn.SiLU)
    parser.add_argument("--grid_eps", type=float, default=0.02)
    parser.add_argument("--grid_range", type=list, default=[-1, 1])
    parser.add_argument("--pde_weight", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    if args.pinn_train:
        train_pinn(args)
    elif args.kan_train:
        train_kan(args)
    else:
        pass
