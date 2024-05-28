"""
预训练一个pinn，后续再接eql。不是端到端。但是貌似没有使用。
"""
import os
import argparse
import torch
import time
import numpy as np
import random
import logging
import sympy as sp
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from SRNet import register_sr, register_params, diff_cgp_evo
from SRNet.usr_models import DiffMLP
from load_data import load_pde_data
from utils import save_srnet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_diff(diff, val_loader, device):
    val_loss = 0
    val_size = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X, y = X.to(device), y.to(device)
        loss = nn.functional.mse_loss(diff(X), y)

        val_size += 1
        val_loss += loss.item()

    val_loss /= val_size
    return val_loss


def validate_pde(diff_model, sr_model, val_loader, device, loss_fn):
    val_loss = 0
    val_size = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X, y = X.to(device), y.to(device)
        Xs = [
            Variable(X[:, i], requires_grad=True) for i in range(X.shape[1])
        ]
        X_var = torch.stack(Xs, dim=1)

        u_hat = diff_model(X_var).squeeze()
        loss = loss_fn(sr_model, Xs, y, u_hat)["loss"]

        val_size += 1
        val_loss += loss.item()

    val_loss /= val_size
    return val_loss


def pretrain_pde(args, pde, train_loader, writer, device):
    optimizer = torch.optim.Adam(
        pde.parameters(), lr=args.pretrain_lr
    )
    loss_fn = nn.MSELoss()

    step = 0
    tqbar = tqdm(range(1, args.pretrain_epochs + 1), total=args.pretrain_epochs, desc=f"Pretraining PDE {args.problem}")
    best_val_loss = float("inf")
    for ep in tqbar:
        batch_loss = 0
        batch_size = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            u = pde(X).squeeze()
            loss = loss_fn(u, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            batch_size += 1
            batch_loss += loss.item()
            writer.add_scalar("loss/pretrain_step_loss", loss.item(), step)
            tqbar.set_postfix(pt_step_loss=loss.item(), avg_loss=batch_loss / batch_size)

            if step % args.pretrain_eval_steps == 0:
                val_loss = validate_diff(pde, train_loader, device)
                writer.add_scalar("loss/pretrain_val_loss", val_loss, step)
                tqbar.set_postfix(pt_val_loss=val_loss)
                logging.info(f"Epoch {ep}, step {step}: val_loss={val_loss}")

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    logging.info(f"Hit best val loss: {best_val_loss}. Saving...")
                    torch.save(pde.state_dict(), os.path.join(args.out_dir, "pretrained_pde.pt"))

            if step == args.pretrain_steps:
                return best_val_loss

        batch_loss /= batch_size
        writer.add_scalar("loss/pretrain_epoch_loss", batch_loss, ep)
        tqbar.set_postfix(pt_epoch_loss=batch_loss)

    return best_val_loss


def train_pde(args):
    if args.out_dir == "":
        args.out_dir = os.path.join("output", args.problem)

    os.makedirs(args.out_dir, exist_ok=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb_dir = os.path.join(args.out_dir, "tb", cur_time)
    logging.basicConfig(
        filename=os.path.join(args.out_dir, "log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    setup_seed(args.seed)

    # load data
    train_dataset = load_pde_data(args.problem)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_inputs, n_outputs = train_dataset.X.shape[1], 1

    # load sr model
    sr_class = register_sr[args.sr_class]
    param_class = register_params[args.sr_class]
    sr_param = param_class(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_eph=1,
        args=args,
        function_set=["add", "sub", "mul", "div", "diff", "diff2"]
    )
    sr_model = sr_class(sr_param)
    # load shared diff model
    diff = DiffMLP(n_inputs, n_layer=5, hidden_size=20)

    if args.load_pretrain_pde:
        diff.load_state_dict(
            torch.load(args.pretrain_pde_path)
        )

    # logging
    logging.info(f"Problem: {args.problem}")
    logging.info(f"Model: {args.sr_class}; # inputs: {n_inputs}; # outputs: {n_outputs}")
    logging.info(f"Parameters: {vars(sr_param)}")
    logging.info(f"Training data: {train_dataset.X.shape}")
    logging.info(f"Output: {args.out_dir}")
    torch.save(args, os.path.join(args.out_dir, "args"))
    torch.save(sr_param, os.path.join(args.out_dir, "sr_param"))

    # train
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(tb_dir)
    diff = diff.to(device)
    sr_model = sr_model.to(device)

    if args.pretrain_pde:
        pretrain_loss = pretrain_pde(args, diff, train_loader, writer, device)
        diff.load_state_dict(
            torch.load(os.path.join(args.out_dir, "pretrained_pde.pt"), map_location=device)
        )

    if args.evolution:
        def evaluate(sr_model, X, y, u_hat, diff_loss=None):
            expression = sr_model.expr()
            penalty = sp.count_ops(expression)
            if penalty == 0:
                penalty = 1e3
            else:
                penalty = 0

            side_loss = torch.mean(sr_model(X + [u_hat]).squeeze() ** 2)
            if diff_loss is not None:
                return {
                    "diff_loss": diff_loss,
                    "side_loss": side_loss,
                    "penalty": penalty,
                    "loss": side_loss + penalty
                }

            if args.joint_training:
                diff_loss =F.mse_loss(u_hat, y)
                loss = args.joint_alpha * diff_loss + side_loss + penalty
                return {
                    "diff_loss": diff_loss,
                    "side_loss": side_loss,
                    "penalty": penalty,
                    "loss": loss,
                }

            loss = side_loss + penalty
            # 每种loss都代表什么意思？
            return {
                "diff_loss": torch.tensor(0.),
                "side_loss": side_loss,
                "penalty": penalty,
                "loss": loss,
            }

        global_step = 0
        tqbar = tqdm(range(1, args.n_epochs + 1), total=args.n_epochs, desc=f"Training PDE {args.problem}")
        best_loss = float("inf")
        best_val_loss = float("inf")
        for eq in tqbar:
            batch_loss = 0.
            batch_size = 0
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)

                all_loss, loss, parent = diff_cgp_evo(
                    diff, sr_model, X, y, evaluate, args, return_all_loss=True
                )
                diff_loss = all_loss["diff_loss"].item()
                side_loss = all_loss["side_loss"].item()
                penalty = all_loss["penalty"]

                global_step += 1
                batch_loss += loss
                batch_size += 1

                writer.add_scalar("loss/train_step_loss", loss, global_step)
                writer.add_scalar("loss/train_step_diff_loss", diff_loss, global_step)
                writer.add_scalar("loss/train_step_side_loss", side_loss, global_step)
                writer.add_scalar("loss/train_step_penalty", penalty, global_step)
                tqbar.set_postfix(step_loss=loss, side_loss=side_loss, diff_loss=diff_loss, penalty=penalty)

                if loss < best_loss:
                    sr_model = parent
                    best_loss = loss

                    if global_step % args.eval_steps == 0:
                        # evaluate model
                        val_loss = validate_pde(diff, sr_model, train_loader, device, evaluate)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(diff.state_dict(), os.path.join(args.out_dir, "diff.pt"))
                            save_srnet(sr_model, args.out_dir)

            batch_loss /= batch_size
            writer.add_scalar("loss/train_epoch_loss", loss, eq)
            tqbar.set_postfix(epoch_loss=batch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--sr_class", type=str, default="CGP")
    parser.add_argument("--load_pretrain_pde", action="store_true", default=True)
    parser.add_argument("--pretrain_pde_path", type=str, default="output/burgers/pretrain-joint/pretrained_pde.pt")
    # data
    parser.add_argument("--problem", type=str, default="burgers")
    # training
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="output/pretrain_joint")
    parser.add_argument("--seed", type=int, default=12)
    ## pretrain mlp for pde parser
    parser.add_argument("--pretrain_pde", action="store_true", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--pretrain_steps", type=int, default=500)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_eval_steps", type=int, default=100)
    ## srnet training params
    ### evolution
    parser.add_argument("--evolution", action="store_true", default=True)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--prob", type=float, default=0.4)
    parser.add_argument("--n_rows", type=int, default=10)
    parser.add_argument("--n_cols", type=int, default=6)
    parser.add_argument("--levels_back", type=int, default=20)
    parser.add_argument("--n_diff_cols", type=int, default=2)
    ### optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd", "lbfgs"])
    parser.add_argument("--optim_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--joint_training", action="store_true", default=True)  # 这个参数是什么意思？
    parser.add_argument("--joint_alpha", type=float, default=0.3)
    args = parser.parse_args()

    train_pde(args)


