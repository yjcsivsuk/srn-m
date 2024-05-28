from functools import partial
import os
import time
import argparse
import torch
import numpy as np
import random

from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from load_data import load_mnist_data
from neural_network import neural_networks
from SRNet import srnets, evolutions, add_sg_functions, default_functions
from utils import get_warmup_linear_scheduler, load_pickle, loss_map, save_pickle
from utils import srnet_layer_loss, srnet_layer_loss_with_grad


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_srnet_layer(net, srnet_layer, idx_layer, val_loader, device, metric_fn, micro=False):
    """
        if micro == True, then metric will be divided by samples, otherwise batches
    """
    was_training = srnet_layer.training
    srnet_layer.eval()

    if isinstance(val_loader, DataLoader):
        val_metric = 0.
        n_batch, n_samples = 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                nn_outputs = net(xb)
                layer_input = xb if idx_layer == 0 else nn_outputs[idx_layer - 1]
                layer_output = nn_outputs[idx_layer]
                predicts = srnet_layer(layer_input)
                val_metric += metric_fn(predicts, layer_output)
            n_batch += 1
            n_samples += xb.size(0)
        if micro:
            val_metric /= n_samples
        else:
            val_metric /= n_batch
    else:
        # since we use val_loader.X, so this must be srnet validation
        X = val_loader.X.to(device)
        with torch.no_grad():
            nn_outputs = net(xb)
            layer_input = xb if idx_layer == 0 else nn_outputs[idx_layer - 1]
            layer_output = nn_outputs[idx_layer]
            predicts = srnet_layer(layer_input)
            val_metric = metric_fn(predicts, layer_output)
        if micro:
            val_metric /= X.size(0)

    if was_training:
        srnet_layer.train()
    return val_metric


def optim_srnet(srnet, net, optimizer, X, loss_fn, fn_type, args, return_hidden_loss=False):
    if not args.grad_loss:
        with torch.no_grad():
            nn_outputs = net(X)

    def closure():
        optimizer.zero_grad()

        if args.grad_loss:
            cur_loss = loss_fn(srnet=srnet, net=net, X=X, fn_type=fn_type, merge=False)
        else:
            predicts = srnet(X)
            cur_loss = loss_fn(srnet=srnet, outputs=predicts, targets=nn_outputs, fn_type=fn_type, merge=False)
        if isinstance(cur_loss, tuple):
            cur_loss = cur_loss[-1]

        cur_loss.backward()
        return cur_loss

    if args.optim == "lbfgs":
        for i in range(args.optim_epoch):
            optimizer.step(closure)
    else:
        closure()
        optimizer.step()

    if args.grad_loss:
        loss = loss_fn(srnet=srnet, net=net, X=X, fn_type=fn_type, merge=False)
    else:
        predicts = srnet(X)
        loss = loss_fn(srnet=srnet, outputs=predicts, targets=nn_outputs, merge=False)

    hidden_losses = loss[0]
    if isinstance(loss, tuple):
        loss = loss[-1].item()

    if return_hidden_loss:
        return hidden_losses, loss
    return loss


def train_img_srnet(args):
    def correct_count(outputs, targets):
        return (outputs.argmax(dim=-1) == targets.argmax(dim=-1)).sum()

    if args.srnet_dir == "":
        srnet_dir = os.path.join(args.log_dir, f"{args.nn_name}-explained")
    else:
        srnet_dir = os.path.join(args.log_dir, args.srnet_dir)
    os.makedirs(srnet_dir, exist_ok=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb_dir = os.path.join(srnet_dir, "tb", cur_time)
    setup_seed(args.seed)

    # loading dataset
    train_dataset, val_dataset = load_mnist_data(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    img_example, target_example = train_dataset[0]
    in_shape = img_example.shape
    num_classes = 10
    n_inputs, n_outputs = in_shape[0], num_classes
    total_steps = len(train_loader) * args.epoch

    # device
    device = torch.device(args.gpu)

    # loading trained neural network
    nn_class = neural_networks[args.nn_name]
    net = nn_class(num_classes)
    net.load_state_dict(
        torch.load(args.nn_path, map_location=device)
    )
    net = net.to(device)
    net.eval()

    # loading srnet
    param_class, srnet_class = srnets[args.srnet_name]
    function_set = ["add", "mul", "sin", "log", "cube", "square"]
    if args.add_sg_function:
        function_set.extend(["image_dx", "image_dy"])
    sr_param = param_class(
        n_inputs, n_outputs,
        n_eph=0, args=args,
        function_set=function_set
    )
    srnet = srnet_class(sr_param=sr_param, neural_network=net, in_shape=in_shape).to(device)
    print(srnet)
    print("total steps:", total_steps)
    print("train n batch:", len(train_loader))

    if not args.evolution and args.srnet_name != "MEQL_net":
        srnet.load_state_dict(
            torch.load(args.srnet_model, map_location=device),
            strict=False
        )
        srnet_genes = load_pickle(
            os.path.join("/".join(args.srnet_model.split("/")[:-1]), "srnet_genes")
        )
        srnet.assign_genes(srnet_genes)
    n_layer = len(srnet.explained_layers)
    print("srnet # layers:", n_layer)

    ## save some arguments
    torch.save(args, os.path.join(srnet_dir, "args"))
    torch.save(sr_param, os.path.join(srnet_dir, "sr_param"))

    # preparing
    writer = SummaryWriter(tb_dir)

    save_step = args.epoch // 10 + 1

    optimizer = None
    scheduler = None
    if not args.evolution:
        if args.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                srnet.parameters(), lr=args.lr, max_iter=10
            )
        elif args.optim == "adam":
            optimizer = torch.optim.Adam(
                srnet.parameters(), lr=args.lr
            )
        else:
            optimizer = torch.optim.SGD(
                srnet.parameters(), lr=args.lr
            )

        if args.warmup_ratio > 0:
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps)
    if args.grad_loss:
        loss_fn = partial(srnet_layer_loss_with_grad, args=args)
    else:
        loss_fn = partial(srnet_layer_loss, args=args)

    # preparing
    writer = SummaryWriter(tb_dir)
    save_step = args.epoch // 10
    # training
    for idx_layer in range(n_layer):
        tqbar = tqdm(range(1, args.epoch + 1), total=args.epoch, desc=f"SRNet-{idx_layer} layer training")
        gb = 0

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_hidden_losses = []

        srnet_layer = srnet.explained_layers[idx_layer]
        net_layer = net[idx_layer]

        print(srnet_layer)
        print(net_layer)
        best_srnet_layer = srnet_layer
        layer_loss_fn = loss_map[args.hidden_fn if idx_layer < n_layer - 1 else args.out_fn]

        for ep in tqbar:
            n_batch = 0
            loss_batch = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if idx_layer == 0:
                    layer_input = x_batch
                else:
                    with torch.no_grad():
                        nn_outputs = net(x_batch)
                    layer_input = nn_outputs[idx_layer - 1]
                    layer_outputs = nn_outputs[idx_layer]
                if idx_layer == n_layer - 1:
                    layer_input = layer_input.reshape(layer_input.size(0), -1)

                gb += 1

                if args.evolution:
                    cur_loss, cur_srnet_layer = evolutions["layer"](
                        srnet_layer,
                        layer_input,
                        layer_outputs,
                        loss_fn,
                        args.hidden_fn if idx_layer < n_layer - 1 else args.out_fn,
                        args
                    )

                    if not torch.isnan(cur_loss) and not torch.isinf(cur_loss):
                        srnet_layer = cur_srnet_layer
                        cur_loss = cur_loss.item()

                else:
                    if args.grad_loss:
                        layer_input.requires_grad = True
                    cur_hidden_loss, cur_loss = optim_srnet(
                        srnet_layer,
                        net_layer,
                        optimizer,
                        layer_input,
                        loss_fn,
                        args.hidden_fn if idx_layer < n_layer - 1 else args.out_fn,
                        args, return_hidden_loss=True
                    )

                if scheduler is not None:
                    scheduler.step()

                if cur_loss < best_train_loss:
                    best_train_loss = cur_loss
                    best_hidden_losses = cur_hidden_loss

                loss_batch += best_train_loss
                n_batch += 1

                tqbar.set_postfix(step_loss=cur_loss, best_loss=best_train_loss, avg_loss=loss_batch / n_batch)
                writer.add_scalar(f"layer-{idx_layer}-loss/train_step_loss", best_train_loss, gb)
                for n, l in best_hidden_losses.items():
                    writer.add_scalar(f"layer-{idx_layer}-loss/train_step_{n}", l.item(), gb)

            loss_batch /= n_batch
            val_loss = validate_srnet_layer(
                net, srnet_layer, idx_layer,
                val_loader, device,
                layer_loss_fn if idx_layer < n_layer - 1 else correct_count,
                micro=False if idx_layer < n_layer - 1 else True
            )
            val_loss = val_loss.item()

            if idx_layer < n_layer - 1:
                tqbar.set_postfix(train_loss=best_train_loss, val_loss=val_loss, avg_loss=loss_batch)
                writer.add_scalar(f"layer-{idx_layer}-loss/val_loss", val_loss, ep)
            else:
                tqbar.set_postfix(train_loss=best_train_loss, val_acc=val_loss, avg_loss=loss_batch)
                writer.add_scalar(f"layer-{idx_layer}-metric/val_acc", val_loss, ep)
            writer.add_scalar(f"layer-{idx_layer}-loss/train_epoch_loss", loss_batch, ep)

            # if ep % save_step == 0:
            #     pass
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    srnet_layer.state_dict(),
                    os.path.join(srnet_dir, f"srnet-{idx_layer}")
                )
                if args.srnet_name != "MEQL_net":
                    save_pickle(srnet.get_genes(), os.path.join(srnet_dir, "srnet_genes"))
                best_srnet_layer = srnet_layer

        srnet.explained_layers[idx_layer].load_state_dict(best_srnet_layer.state_dict())

    torch.save(
        srnet.state_dict(),
        os.path.join(srnet_dir, "srnet")
    )


if __name__ == "__main__":
    def boolean_str(s):
        return s == "True"


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_imgsrnet", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="output")
    parser.add_argument("--srnet_dir", type=str, default="explained-mlp")
    # training neural network
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nn_name", type=str, default="LeNet")
    # training srnet
    parser.add_argument("--nn_path", type=str, default="output/LeNet/LeNet")
    parser.add_argument("--srnet_model", type=str, default=None)
    parser.add_argument("--srnet_name", type=str, default="MCGP_net")
    parser.add_argument("--add_sg_function", type=boolean_str, default="False")
    parser.add_argument("--levels_back", type=int, default=4)
    parser.add_argument("--n_rows", type=int, default=3)
    parser.add_argument("--n_cols", type=int, default=3)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optim", type=str, default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    parser.add_argument("--optim_epoch", type=int, default=5, help="for srnet lbfgs")
    parser.add_argument("--temperature", type=int, default=2, help="for srnet classification task")
    parser.add_argument("--hidden_weight", type=int, default=0, help="0 for 1/#layer")
    parser.add_argument("--out_fn", type=str, default="kl", choices=["kl", "mse", "ce"])
    parser.add_argument("--hidden_fn", type=str, default="mse", choices=['mse', 'bce', 'cos'])
    parser.add_argument("--out_weight", type=float, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    # for MEQL
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--grad_loss", type=boolean_str, default="False")
    parser.add_argument("--evolution", type=boolean_str, default="False")
    # common
    parser.add_argument("--dataset", type=str, default="data/img")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gpu", type=str, default="cpu")
    args = parser.parse_args()

    train_img_srnet(args)