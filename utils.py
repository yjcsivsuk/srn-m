import torch
import pickle
import os

from matplotlib import pyplot as plt
from SRNet import srnets
from torch import nn
from torch.nn import functional as F


# 均方误差
def mse_loss(output, target):
    return F.mse_loss(output, target)


# KL散度
def kl_divergence(output, target, args):
    soft_output = output / args.temperature
    soft_target = target / args.temperature
    out_loss = F.kl_div(
        torch.log_softmax(soft_output, dim=-1),
        torch.softmax(soft_target, dim=-1),
        reduction="batchmean"
    )
    return out_loss


# 二元交叉熵
def bce_loss(output, target):
    stand_target = F.sigmoid(target)
    return F.binary_cross_entropy_with_logits(output, stand_target)


# 交叉熵
def cross_entropy_loss(output, target, args):
    softmaxed_target = F.softmax(target / args.temperature, dim=-1)
    soft_output = output / args.temperature
    return F.cross_entropy(soft_output, softmaxed_target)


def srnet_loss(args, srnet, outputs, targets, merge=True):
    hidden_fn = loss_map[args.hidden_fn]
    last_fn = loss_map[args.out_fn]

    loss = 0.
    all_losses = {}
    n_layer = len(outputs)
    if args.hidden_weight == 0:
        hidden_weight = 1 / (n_layer - 1)
    else:
        hidden_weight = args.hidden_weight

    for i, (o, t) in enumerate(zip(outputs[:-1], targets[:-1])):
        hidden_loss = hidden_fn(o, t, args)
        loss += hidden_loss * hidden_weight
        all_losses[f"hidden-{i}-loss"] = hidden_loss

    out_loss = last_fn(outputs[-1], targets[-1], args)
    loss += out_loss * args.out_weight

    regular = 0
    if args.weight_decay > 0:
        regular = srnet.regularization()
        all_losses["regular"] = regular
    loss += args.weight_decay * regular

    all_losses[f"out-loss"] = out_loss
    if merge:
        return loss
    return all_losses, loss


def srnet_loss_with_grad(args, srnet, net, X, merge=True):
    outputs = srnet(X)
    targets = net(X)

    outputs_grad = [
        torch.autograd.grad(o, X, torch.ones_like(o), retain_graph=True, create_graph=True)[0]
        for o in outputs
    ]
    # outputs_grad2 = [
    #     torch.autograd.grad(o, X, torch.ones_like(o), retain_graph=True, create_graph=True)[0]
    #     for o in outputs_grad
    # ]

    targets_grad = [
        torch.autograd.grad(t, X, torch.ones_like(t), retain_graph=True, create_graph=True)[0]
        for t in targets
    ]
    # targets_grad2 = [
    #     torch.autograd.grad(t, X, torch.ones_like(t), retain_graph=True, create_graph=True)[0]
    #     for t in targets_grad
    # ]

    hidden_fn = loss_map[args.hidden_fn]
    last_fn = loss_map[args.out_fn]

    loss = 0.
    all_losses = {}
    n_layer = len(outputs)
    if args.hidden_weight == 0:
        hidden_weight = 1 / (n_layer - 1)
    else:
        hidden_weight = args.hidden_weight

    for i, (o, t) in enumerate(zip(outputs[:-1], targets[:-1])):
        hidden_loss = hidden_fn(o, t, args)
        loss += hidden_loss * hidden_weight
        all_losses[f"hidden-{i}-loss"] = hidden_loss
    for i, (o, t) in enumerate(zip(outputs_grad[:-1], targets_grad[:-1])):
        hidden_loss = hidden_fn(o, t, args)
        loss += hidden_loss * hidden_weight
        all_losses[f"hidden-{i}-grad-loss"] = hidden_loss
    # for i, (o, t) in enumerate(zip(outputs_grad2[:-1], targets_grad2[:-1])):
    #     hidden_loss = hidden_fn(o, t, args)
    #     loss += hidden_loss * hidden_weight
    #     all_losses[f"hidden-{i}-grad2-loss"] = hidden_loss

    out_loss = last_fn(outputs[-1], targets[-1], args)
    all_losses[f"out-loss"] = out_loss
    grad_out_loss = last_fn(outputs_grad[-1], targets_grad[-1], args)
    all_losses["grad-out-loss"] = grad_out_loss
    # grad2_out_loss = last_fn(outputs_grad2[-1], targets_grad2[-1], args)
    loss += (out_loss + grad_out_loss) * args.out_weight

    regular = 0
    if args.weight_decay > 0 and args.regular_type == "l1":
        regular = srnet.regularization()
        all_losses["regular"] = regular
    loss += args.weight_decay * regular

    if merge:
        return loss
    return all_losses, loss


def srnet_layer_loss(args, srnet, outputs, targets, fn_type, merge=True):
    loss_fn = loss_map[fn_type]

    loss = 0.
    all_losses = {}

    out_loss = loss_fn(outputs, targets, args)
    all_losses["output-loss"] = out_loss
    loss += out_loss

    regular = 0
    if args.weight_decay > 0:
        regular = srnet.regularization()
        all_losses["regular"] = regular
    loss += args.weight_decay * regular

    if merge:
        return loss
    return all_losses, loss


def srnet_layer_loss_with_grad(args, srnet, net, X, fn_type, merge=True):
    # srnet and net: single layer
    outputs = srnet(X)
    targets = net(X)

    outputs_grad = torch.autograd.grad(
        outputs, X, torch.ones_like(outputs), retain_graph=True, create_graph=True
    )[0]

    targets_grad = torch.autograd.grad(
        targets, X, torch.ones_like(targets), retain_graph=True, create_graph=True
    )[0]

    loss_fn = loss_map[fn_type]

    loss = 0.
    all_losses = {}

    output_loss = loss_fn(outputs, targets, args)
    all_losses["output-loss"] = output_loss
    grad_loss = loss_fn(outputs_grad, targets_grad, args)
    all_losses["grad-loss"] = grad_loss
    loss += output_loss + grad_loss

    regular = 0
    if args.weight_decay > 0 and args.regular_type == "l1":
        regular = srnet.regularization()
        all_losses["regular"] = regular

    loss += args.weight_decay * regular

    if merge:
        return loss
    return all_losses, loss


# PINN+EQL的loss
def pde_loss_fn(args, PDE, input_data, U):
    mse_fn = nn.MSELoss()

    predict = PDE(*input_data)
    U_hat = predict["u_hat"]
    pd_reals, pd_hats = predict["pd_reals"], predict["pd_hats"]
    pde_out = predict["pde_out"]

    u_loss = mse_fn(U_hat.squeeze(), U.flatten())
    if pd_reals and pd_hats is not None:
        pd_loss = torch.tensor(0., device=args.gpu)
        for pd_real, pd_hat in zip(pd_reals, pd_hats):
            pd_loss += mse_fn(pd_real, pd_hat)
    pde_loss = torch.mean(torch.abs(pde_out.flatten()))
    regularizations = torch.tensor(0., device=args.gpu)
    if args.weight_decay > 0:
        regularizations = PDE.regularization()

    loss = u_loss + args.pd_weight * pd_loss \
           + args.pde_weight * pde_loss \
           + args.weight_decay * regularizations

    return {
        "u_loss": u_loss,  # 通过PINN神经网络预测的u^hat和真实的u之间的loss
        "pd_loss": pd_loss,  # 通过PINN神经网络预测的ux^hat和输入进PINN的ux之间的loss
        "pde_loss": pde_loss,  # 偏微分项经过EQL出来之后的结果，限制其为0的惩罚项
        "regularization": regularizations,  # EQL的L1正则化惩罚项
        "loss": loss  # 上述loss加权和，作为卷积模块符号层的总体loss
    }


loss_map = {
    "mse": mse_loss,
    "kl": kl_divergence,
    "bce": bce_loss,
    "ce": cross_entropy_loss,
}


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_srnet(srnet, path_dir):
    torch.save(srnet.state_dict(), os.path.join(path_dir, "srnet"))
    save_pickle(srnet.get_genes(), os.path.join(path_dir, "srnet_genes"))


def load_empty_srnet(srnet_name, neural_net, srnet_dir):
    param_cls, srnet_cls = srnets[srnet_name]
    sr_param = torch.load(os.path.join(srnet_dir, "sr_param"))
    srnet = srnet_cls(sr_param, neural_net, in_shape=(1, 32, 32))

    return srnet


def load_img_srnet(srnet_name, neural_net, srnet_dir, device):
    param_cls, srnet_cls = srnets[srnet_name]
    sr_param = torch.load(os.path.join(srnet_dir, "sr_param"))
    srnet = srnet_cls(sr_param, neural_net, in_shape=(1, 32, 32))

    state_dict = torch.load(os.path.join(srnet_dir, "srnet"), map_location=device)
    # print(list(state_dict.keys()))
    srnet.load_state_dict(state_dict)

    if srnet_name != "MEQL_net":
        srnet_genes = load_pickle(os.path.join(srnet_dir, "srnet_genes"))
        srnet.assign_genes(srnet_genes)

    return srnet


# EQL训练时的线性预热策略
def warmup_linear_schedule(step, warmup_steps, total_steps, lr0):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps)) * lr0
    else:
        progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
        return (1 - progress) * lr0


def get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step: int):
        return float(min(current_step, warmup_steps)) / float(max(1, warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


# 显示卷积图像
def show_img(img: torch.Tensor, row, col, title="", vmin=None, vmax=None, save_path=None):
    C, H, W = img.shape
    assert row * col == C
    kwargs = {"vmin": vmin, "vmax": vmax}
    figure, axis = plt.subplots(row, col)
    if C == 1:
        im = axis.imshow(img[0].detach().cpu(), **kwargs)
        axis.set_axis_off()
        figure.colorbar(im, ax=axis)
    else:
        for i in range(C):
            r = i // col
            c = i - r * col

            if row == 1:
                ax = axis[c]
            else:
                ax = axis[r, c]
            ax.imshow(img[i].detach().cpu(), **kwargs)
            ax.set_axis_off()

    plt.suptitle(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()
