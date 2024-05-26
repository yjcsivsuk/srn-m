import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def _legal_loss(loss):
    if np.isnan(loss) or np.isinf(loss):
        return False
    return True


class Population(nn.Module):
    def __init__(self, population) -> None:
        super().__init__()
        self.population = nn.ModuleList(population)

    def forward(self, x: torch.Tensor):
        # results: n_pop * List[Tensor]
        futures = [torch.jit.fork(model, x) for model in self.population]
        results = [torch.jit.wait(fut) for fut in futures]
        return results

    def __getitem__(self, key):
        return self.population[key]

    def __len__(self):
        return len(self.population)


# 使用CGP演化整个网络
def cgp_evo(srnet, X, nn_outputs, loss_fn, args, return_hidden_loss=False, parents=None):
    # Step 1: 1 -> \lambda mutataion
    parent = srnet.copy_self()

    pop_genes = [parent.get_genes()] + [
        parent.mutate(args.prob)
        for _ in range(args.pop_size - 1)
    ]
    pop_models = [parent] + [
        parent.create()
        for _ in range(args.pop_size - 1)
    ]
    for i in range(1, args.pop_size):
        pop_models[i].assign_genes(pop_genes[i])
    population = Population(pop_models)

    # Step 2: optimize constants and weights
    if args.optim == "lbfgs":
        optimizer = torch.optim.LBFGS(
            population.parameters(), lr=args.lr, max_iter=10
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            population.parameters(), lr=args.lr
        )
    else:
        optimizer = torch.optim.SGD(
            population.parameters(), lr=args.lr
        )

    def closure():
        optimizer.zero_grad()
        pop_predicts = population(X)
        loss = 0.
        for i, predicts in enumerate(pop_predicts):
            cur_loss = loss_fn(predicts, nn_outputs)
            if isinstance(cur_loss, tuple):
                cur_loss = cur_loss[-1]
            loss += cur_loss
        loss.backward()
        return loss

    if args.optim == "lbfgs":
        for i in range(args.optim_epoch):
            optimizer.step(closure)
    else:
        closure()
        optimizer.step()

    # Step 3: drag best individual
    with torch.no_grad():
        pop_predicts = population(X)

    best_idx = 0
    best_loss = float("inf")
    best_hidden_losses = None
    for i, predicts in enumerate(pop_predicts):
        cur_loss = loss_fn(predicts, nn_outputs)
        if isinstance(cur_loss, tuple):
            compare_loss = cur_loss[-1].cpu().detach().item()
        else:
            compare_loss = cur_loss.cpu().detach().item()
        if not _legal_loss(compare_loss):
            compare_loss = float("inf")

        if compare_loss < best_loss:
            best_idx = i
            best_loss = compare_loss
            if return_hidden_loss:
                best_hidden_losses = cur_loss[0]
    if return_hidden_loss:
        return best_hidden_losses, best_loss, population.population[best_idx]
    return best_loss, population.population[best_idx]


# 使用CGP演化网络中的每个层
def cgp_layer_evo(srnet, X, nn_outputs, loss_fn, fn_type, args):
    def _legal_loss(loss):
        if np.isnan(loss) or np.isinf(loss):
            return False
        return True

    # Step 1: 1 -> \lambda mutataion
    parent = srnet.copy_self()
    parent.assign_genes(srnet.get_genes())

    pop_genes = [parent.get_genes()] + [
        parent.mutate(args.prob)
        for _ in range(args.pop_size - 1)
    ]
    pop_models = [parent] + [
        parent.copy_self()
        for _ in range(args.pop_size - 1)
    ]
    for i in range(1, args.pop_size):
        pop_models[i].assign_genes(pop_genes[i])
    population = Population(pop_models)

    # Step 2: optimize constants and weights
    if args.optim == "lbfgs":
        optimizer = torch.optim.LBFGS(
            population.parameters(), lr=args.lr, max_iter=10
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            population.parameters(), lr=args.lr
        )
    else:
        optimizer = torch.optim.SGD(
            population.parameters(), lr=args.lr
        )

    def closure():
        optimizer.zero_grad()
        pop_predicts = population(X)
        loss = 0.
        for i, predicts in enumerate(pop_predicts):
            cur_loss = loss_fn(srnet=population[i], outputs=predicts, targets=nn_outputs, fn_type=fn_type)
            if isinstance(cur_loss, tuple):
                cur_loss = cur_loss[-1]
            loss += cur_loss
        loss.backward()
        return loss

    if args.optim == "lbfgs":
        for i in range(args.optim_epoch):
            optimizer.step(closure)
    else:
        closure()
        optimizer.step()

    # Step 3: drag best individual
    with torch.no_grad():
        pop_predicts = population(X)

    best_idx = 0
    best_loss = float("inf")
    for i, predicts in enumerate(pop_predicts):
        cur_loss = loss_fn(srnet=population[i], outptus=predicts, targets=nn_outputs, fn_type=fn_type)
        if isinstance(cur_loss, tuple):
            compare_loss = cur_loss[-1].cpu().detach()
        else:
            compare_loss = cur_loss.cpu().detach()
        if not _legal_loss(compare_loss):
            compare_loss = float("inf")

        if compare_loss < best_loss:
            best_idx = i
            best_loss = compare_loss

    return best_loss, population.population[best_idx]


def diff_cgp_evo(
        diff_model, sr_model, X, y, evaluate_fn, args, return_all_loss=False
):
    # (1, \lambda)
    parent = sr_model.copy_self()
    childs = [
        parent.mutate(args.prob)
        for _ in range(args.pop_size - 1)
    ]
    population = [parent] + [
        sr_model.__class__(sr_model.sr_param, genes=childs[i])
        for i in range(args.pop_size - 1)
    ]

    population = Population(population)

    # Step 2: optimize constants and weights
    population_parameters = list(population.parameters())
    if args.joint_training:
        diff_parameters = list(diff_model.parameters())
        # combine parameters
        population_parameters = population_parameters + diff_parameters

    if args.optim == "lbfgs":
        optimizer = torch.optim.LBFGS(
            population_parameters, lr=args.lr, max_iter=10
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            population_parameters, lr=args.lr
        )
    else:
        optimizer = torch.optim.SGD(
            population_parameters, lr=args.lr
        )

    # stupid trick to make sure that the inputs are differentiable
    Xs = [
        Variable(X[:, i], requires_grad=True) for i in range(X.shape[1])
    ]
    X_var = torch.stack(Xs, dim=1)
    u_hat = diff_model(X_var)
    diff_loss = None

    if args.joint_training:
        diff_loss = F.mse_loss(u_hat, y)

    def closure():
        optimizer.zero_grad()

        loss = 0.
        for i, indiv in enumerate(population):
            # clear gradient on X
            for j in range(len(Xs)):
                if Xs[j].grad is not None:
                    Xs[j].grad.data.zero_()

            cur_loss = evaluate_fn(indiv, Xs, y, u_hat, diff_loss)
            print(cur_loss)
            cur_loss = cur_loss["loss"]
            loss += cur_loss
        if args.joint_training:
            loss = loss + args.joint_alpha * diff_loss

        loss.backward()
        return loss

    if args.optim == "lbfgs":
        for i in range(args.optim_epoch):
            optimizer.step(closure)
    else:
        closure()
        optimizer.step()

    # Step 3: drag best individual
    Xs = [
        Variable(X[:, i], requires_grad=True) for i in range(X.shape[1])
    ]
    X_var = torch.stack(Xs, dim=1)
    u_hat = diff_model(X_var)
    if args.joint_training:
        diff_loss = F.mse_loss(u_hat, y)

    best_idx = 0
    best_loss = float("inf")
    best_all_loss = None

    for i, indiv in enumerate(population):
        # clear gradient on X
        for j in range(len(Xs)):
            if Xs[j].grad is not None:
                Xs[j].grad.data.zero_()

        cur_loss = evaluate_fn(indiv, Xs, y, u_hat, diff_loss)
        compare_loss = cur_loss["loss"].item()
        if not _legal_loss(compare_loss):
            compare_loss = float("inf")

        if compare_loss < best_loss:
            best_idx = i
            best_loss = compare_loss
            best_all_loss = cur_loss
    if return_all_loss:
        return best_all_loss, best_loss, population[best_idx]
    return best_loss, population[best_idx]
