from functools import partial
import os
import argparse
import torch
import time
import numpy as np
import random

from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from neural_network import neural_networks
from SRNet import srnets, evolutions

from load_data import load_pmlb_data, load_sr_data, load_mnist_data
from utils import save_pickle, load_pickle, get_warmup_linear_scheduler, srnet_loss_with_grad, srnet_loss


# 设置随机种子，使代码结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_srnet(net, srnet, val_loader, device, metric_fn, loss_fn, micro=True, grad_loss=False):
    """
        if micro == True, then metric will be divided by samples, otherwise batches
    """
    was_training = srnet.training
    srnet.eval()  # 评估模式

    if isinstance(val_loader, DataLoader):
        val_metric = 0.
        val_loss = 0.
        n_batch, n_samples = 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                nn_outputs = net(xb)[-1]
                predicts = srnet(xb)[-1]

            val_metric += metric_fn(predicts, nn_outputs)
            if grad_loss:
                xb = torch.autograd.Variable(xb, requires_grad=True)
                val_loss += loss_fn(srnet=srnet, X=xb)
            else:
                with torch.no_grad():
                    val_loss += loss_fn(srnet=srnet, outputs=srnet(xb), targets=net(xb))
            n_batch += 1
            n_samples += xb.size(0)
        if micro:
            val_metric /= n_samples
        else:
            val_metric /= n_batch
        val_loss /= n_batch
    else:
        # since we use val_loader.X, so this must be srnet validation
        X = val_loader.X.to(device)
        with torch.no_grad():
            nn_outputs = net(X)
            predicts = srnet(X)
        val_metric = metric_fn(predicts[-1], nn_outputs[-1])
        if grad_loss:
            X = torch.autograd.Variable(X, requires_grad=True)
            val_loss += loss_fn(srnet=srnet, X=X)
        else:
            val_loss = loss_fn(srnet=srnet, outputs=predicts, targets=nn_outputs)
        if micro:
            val_metric /= X.size(0)
        val_loss /= n_batch

    if was_training:
        srnet.train()  # 训练模式
    return val_metric, val_loss


def validate(net, val_loader, device, srnet=None, loss_fn=None, acc=False):
    if srnet is not None:
        assert loss_fn is not None
        srnet.eval()

    if isinstance(val_loader, DataLoader):
        val_loss = torch.tensor(0., dtype=torch.float32, device=device)
        n_batch = 0
        accuracy, total = 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                if srnet is not None:
                    val_loss += loss_fn(net(xb), srnet(xb))
                else:
                    if acc:
                        accuracy += net.count_correct(xb, yb)
                        total += xb.size(0)
                    else:
                        val_loss += net.loss(xb, yb)
            n_batch += 1
        val_loss /= n_batch
        if acc:
            accuracy = accuracy / total
    else:
        # since we use val_loader.X, so this must be srnet validation
        assert srnet is not None
        X = val_loader.X.to(device)
        with torch.no_grad():
            nn_outputs = net(X)
            val_loss = loss_fn(srnet(X), nn_outputs)

    if srnet is not None:
        srnet.train()
    if torch.isnan(val_loss) or torch.isinf(val_loss):
        return torch.tensor(float("inf"))
    if acc:
        return accuracy
    return val_loss


# 训练神经网络。使用了Dataloader,TensorBoard,tqdm。
def train_nn(args):
    def loss_fn(x, y):
        return nn.MSELoss()(x, y)

    model_dir = os.path.join(args.log_dir, args.nn_name)
    os.makedirs(model_dir, exist_ok=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb_dir = os.path.join(model_dir, "tb", cur_time)  # tensorboard日志的存储路径
    setup_seed(args.seed)

    nn_class = neural_networks[args.nn_name]
    if "mlp" in args.nn_name:
        train_dataset, val_dataset, test_dataset = load_pmlb_data(args.dataset)  # 全连接神经网络就用pmlb的数据集
    else:
        train_dataset, val_dataset = load_mnist_data(args.dataset)  # 卷积神经网络就用mnist手写识别数据集

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "mlp" in args.nn_name:
        n_inputs, n_outputs = train_dataset.tensors[0].size(-1), 1
        net = nn_class(n_inputs, n_outputs)
        acc = False
    else:
        net = nn_class(10)
        acc = True

    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter(tb_dir)
    global_step = 0
    save_step = args.epoch // 10
    best_loss = float("inf")

    for ep in tqdm(range(args.epoch), desc=f"training neural network {args.nn_name}"):
        epoch_loss = 0.
        n_batch = 0
        for xb, yb in train_loader:
            # print(xb.size(), yb.size())
            xb, yb = xb.to(device), yb.to(device)
            loss = net.train_step(xb, yb, optimizer, clip=True)

            global_step += 1
            epoch_loss += loss.item()
            n_batch += 1

            writer.add_scalar("loss/train_step_loss", loss.item(), global_step)

        epoch_loss /= n_batch
        val_loss = validate(net, val_loader, device, acc=acc)
        writer.add_scalar("loss/train_ep_loss", epoch_loss, ep)
        writer.add_scalar("loss/val_loss", val_loss.item(), ep)
        if ep % save_step == 0:
            torch.save(net.state_dict(), os.path.join(model_dir, f"checkpoint-{ep}"))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), os.path.join(model_dir, args.nn_name))


def optim_srnet(srnet, net, optimizer, X, loss_fn, args, return_hidden_loss=False):
    if not args.grad_loss:
        with torch.no_grad():
            nn_outputs = net(X)

    def closure():
        optimizer.zero_grad()

        if args.grad_loss:
            cur_loss = loss_fn(srnet=srnet, X=X, merge=False)
        else:
            predicts = srnet(X)
            cur_loss = loss_fn(srnet=srnet, outputs=predicts, targets=nn_outputs, merge=False)
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
        loss = loss_fn(srnet=srnet, X=X, merge=False)
    else:
        predicts = srnet(X)
        loss = loss_fn(srnet=srnet, outputs=predicts, targets=nn_outputs, merge=False)

    hidden_losses = loss[0]
    if isinstance(loss, tuple):
        loss = loss[-1].item()

    if return_hidden_loss:
        return hidden_losses, loss
    return loss


def train_srnet(args):
    def loss_fn(outputs, targets):
        mse_fn = nn.MSELoss()
        loss = 0.
        n_layer = len(outputs)
        hidden_weight = 1 / (n_layer - 1)
        for o, t in zip(outputs[:-1], targets[:-1]):
            loss += mse_fn(o, t) * hidden_weight
        loss += mse_fn(outputs[-1], targets[-1])
        return loss

    if args.srnet_dir == "":
        srnet_dir = os.path.join(args.log_dir, f"{args.nn_name}-explained")
    else:
        srnet_dir = os.path.join(args.log_dir, args.srnet_dir)
    os.makedirs(srnet_dir, exist_ok=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb_dir = os.path.join(srnet_dir, "tb", cur_time)
    setup_seed(args.seed)

    # loading dataset
    train_dataset, val_dataset = load_sr_data(args.dataset)
    X_train, y_train = train_dataset.X, train_dataset.y
    n_inputs, n_outputs = train_dataset.X.size(-1), 1

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading trained neural network
    nn_class = neural_networks[args.nn_name]
    net = nn_class(n_inputs, n_outputs)
    net.load_state_dict(
        torch.load(args.nn_path)
    )
    net = net.to(device)
    net.eval()

    # loading srnet
    param_class, srnet_class = srnets[args.srnet_name]
    sr_param = param_class(
        n_inputs, n_outputs, n_eph=1, args=args,
    )
    srnet = srnet_class(sr_param=sr_param, neural_network=net).to(device)
    print("srnet # layers:", len(srnet.explained_layers))

    ## save some arguments
    torch.save(args, os.path.join(srnet_dir, "args"))
    torch.save(sr_param, os.path.join(srnet_dir, "sr_param"))

    # preparing
    writer = SummaryWriter(tb_dir)

    epoch_loss = 0.
    save_step = args.epoch // 10
    best_loss = float("inf")
    best_val_loss = float("inf")

    # training
    X_train = X_train.to(device)
    with torch.no_grad():
        nn_outputs = net(X_train)
    tqbar = tqdm(range(1, args.epoch + 1), total=args.epoch, desc="SRNet training")
    gb = 0
    for ep in tqbar:
        gb += 1
        cur_loss, cur_srnet = evolutions[args.srnet_name](
            srnet,
            X_train, nn_outputs, loss_fn,
            args
        )

        if cur_loss < best_loss:
            srnet = cur_srnet
            best_loss = cur_loss

        val_loss = validate(net, val_dataset, device, srnet, loss_fn=loss_fn)

        tqbar.set_postfix(train_loss=best_loss, val_loss=val_loss.item(), avg_loss=best_loss / ep)
        writer.add_scalar("loss/train_step_loss", best_loss, ep)
        writer.add_scalar("loss/train_epoch_loss", best_loss / ep, ep)
        writer.add_scalar("loss/val_loss", val_loss.item(), ep)

        if ep % save_step == 0:
            torch.save(srnet.state_dict(), os.path.join(srnet_dir, f"checkpoint-{ep}"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(srnet.state_dict(), os.path.join(srnet_dir, "srnet"))
                if args.srnet_name != "MEQL_net":
                    save_pickle(srnet.get_genes(), os.path.join(srnet_dir, "srnet_genes"))


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
    function_set = ["add", "mul", "sin", "log", "cos", "square"]
    if args.add_sg_function:
        function_set.extend(["image_dx", "image_dy"])
    sr_param = param_class(
        n_inputs, n_outputs,
        n_eph=0, args=args,
        function_set=function_set
    )
    srnet = srnet_class(sr_param=sr_param, neural_network=net, in_shape=in_shape).to(device)
    print(srnet)
    if not args.evolution and args.srnet_name != "MEQL_net":
        srnet.load_state_dict(
            torch.load(args.srnet_model, map_location=device),
            strict=False
        )
        srnet_genes = load_pickle(
            os.path.join("/".join(args.srnet_model.split("/")[:-1]), "srnet_genes")
        )
        srnet.assign_genes(srnet_genes)
    print("srnet # layers:", len(srnet.explained_layers))

    ## save some arguments
    torch.save(args, os.path.join(srnet_dir, "args"))
    torch.save(sr_param, os.path.join(srnet_dir, "sr_param"))

    # preparing
    writer = SummaryWriter(tb_dir)

    save_step = args.epoch // 10
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_hidden_losses = []

    optimizer = None
    scheduler = None
    if not args.evolution:
        if args.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                srnet.parameters(), lr=args.lr, max_iter=10
            )
        elif args.optim == "adam":
            if args.regular_type == "l2" and args.weight_decay > 0:
                optimizer = torch.optim.Adam(
                    srnet.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    srnet.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
        else:
            if args.regular_type == "l2" and args.weight_decay > 0:
                optimizer = torch.optim.SGD(
                    srnet.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
            else:
                optimizer = torch.optim.SGD(
                    srnet.parameters(), lr=args.lr
                )

        if args.warmup_ratio > 0:
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps)
    if args.grad_loss:
        loss_fn = partial(srnet_loss_with_grad, args=args, net=net)
    else:
        loss_fn = partial(srnet_loss, args=args)

    # training
    tqbar = tqdm(range(1, args.epoch + 1), total=args.epoch, desc="SRNet training")
    gb = 0
    parents = None
    for ep in tqbar:
        n_batch = 0
        loss_batch = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            gb += 1

            if args.evolution:
                with torch.no_grad():
                    nn_outputs = net(x_batch)
                evo_results = evolutions[args.srnet_name](
                    srnet,
                    x_batch, nn_outputs, loss_fn,
                    args, return_hidden_loss=True, parents=parents
                )
                cur_hidden_loss, cur_loss, cur_srnet = evo_results[:3]
                if len(evo_results) == 4:
                    parents = evo_results[-1]
            else:
                if args.grad_loss:
                    x_batch = torch.autograd.Variable(x_batch, requires_grad=True)

                cur_hidden_loss, cur_loss = optim_srnet(
                    srnet,
                    net,
                    optimizer,
                    x_batch, loss_fn,
                    args, return_hidden_loss=True
                )

            if scheduler is not None:
                scheduler.step()

            if cur_loss < best_train_loss:
                if args.evolution:
                    srnet = cur_srnet
                best_train_loss = cur_loss
                best_hidden_losses = cur_hidden_loss

            loss_batch += best_train_loss
            n_batch += 1

            tqbar.set_postfix(step_loss=cur_loss, avg_loss=loss_batch / n_batch, best_loss=best_train_loss)
            writer.add_scalar("loss/train_step_loss", best_train_loss, gb)
            for n, l in best_hidden_losses.items():
                writer.add_scalar(f"loss/train_step_{n}", l.item(), gb)

        loss_batch /= n_batch
        val_acc, val_loss = validate_srnet(net, srnet, val_loader, device, correct_count, loss_fn,
                                           grad_loss=args.grad_loss)
        val_acc = val_acc.item()
        val_loss = val_loss.item()

        tqbar.set_postfix(train_loss=best_train_loss, val_loss=val_loss, val_acc=val_acc, avg_loss=loss_batch)
        writer.add_scalar("loss/train_epoch_loss", loss_batch, ep)
        writer.add_scalar("loss/val_loss", val_loss, ep)
        writer.add_scalar("metric/val_acc", val_acc, ep)

        if ep % save_step == 0:
            torch.save(srnet.state_dict(), os.path.join(srnet_dir, f"checkpoint-{ep}"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(srnet.state_dict(), os.path.join(srnet_dir, "srnet"))
                if args.srnet_name != "MEQL_net":
                    save_pickle(srnet.get_genes(), os.path.join(srnet_dir, "srnet_genes"))


if __name__ == "__main__":

    def boolean_str(s):
        return s == "True"


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_srnet", action="store_true", default=False)
    parser.add_argument("--train_imgsrnet", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="output")
    parser.add_argument("--srnet_dir", type=str, default="explained-mlp")
    # training neural network
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nn_name", type=str, default="MLP")
    # training srnet
    parser.add_argument("--nn_path", type=str, default="output/MLP/MLP")
    parser.add_argument("--srnet_model", type=str, default=None)
    parser.add_argument("--srnet_name", type=str, default="MCGP_net", choices=["MCGP_net", "MEQL_net"])
    parser.add_argument("--evolution", type=boolean_str, default="True")

    # sr hyperparameters
    ## CGP
    parser.add_argument("--add_sg_function", type=boolean_str, default="False")
    parser.add_argument("--levels_back", type=int, default=26)
    parser.add_argument("--n_rows", type=int, default=5)
    parser.add_argument("--n_cols", type=int, default=5)
    parser.add_argument("--prob", type=float, default=0.5)

    ## EQL
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--grad_loss", type=boolean_str, default="False")

    # optimization hyperparamters
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--regular_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--optim", type=str, default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    parser.add_argument("--optim_epoch", type=int, default=5, help="for srnet lbfgs")
    parser.add_argument("--temperature", type=int, default=2, help="for srnet classification task")
    parser.add_argument("--hidden_weight", type=float, default=0, help="0 for 1/#layer")
    parser.add_argument("--out_fn", type=str, default="kl", choices=["kl", "mse", "ce"])
    parser.add_argument("--hidden_fn", type=str, default="mse", choices=['mse', 'bce', 'ce', 'cos'])
    parser.add_argument("--out_weight", type=float, default=1)
    # common
    parser.add_argument("--dataset", type=str, default="data/529_pollen.txt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--gpu", type=str, default="cuda")
    args = parser.parse_args()

    if args.train_imgsrnet:
        train_img_srnet(args)
    elif args.train_srnet:
        train_srnet(args)
    else:
        train_nn(args)
