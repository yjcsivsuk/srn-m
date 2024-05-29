import torch
import copy
from torch import nn
from torch.nn.modules import Module
from sr_models import CGPModel
from usr_models import EQL
from sr_layers import SRLayer
from parameters import BaseSRParameter, CGPParameter, EQLParameter
from nn_parser import NetworkInformation, Parser


class BaseSRNet(nn.Module):
    def __init__(
            self,
            sr_class: str,
            sr_param: BaseSRParameter,
            neural_network: nn.Module,
            network_info: NetworkInformation,
            in_shape: tuple
    ) -> None:
        super().__init__()
        assert network_info is not None or neural_network is not None

        self._fitness = None

        self.sr_class = sr_class
        self.sr_param = sr_param
        self.network_info = None
        self.explained_layers = None
        self.network_info = network_info
        self.in_shape = in_shape
        if self.in_shape is None:
            self.in_shape = (self.sr_param.n_inputs,)

        self.init(neural_network)

    def init_network_info(self, neural_network):
        # construct network_info
        if self.network_info is None:
            self.network_info = Parser.parser_network(self.in_shape, neural_network)

    # 对每一个通过autopum划分出来的模块（经nn_parser解析出的）构建各自相应的CGP符号层
    def init_srnet(self):
        em_modules = self.network_info.modules  # modules是不同类型层的组会形成的模块
        em_types = self.network_info.module_types  # module_types是命名
        # clone CGP layers
        unique_em_modules = []
        unique_CGP_layers = []
        CGP_layers = []
        for i, em_module in enumerate(em_modules):
            if em_module in unique_em_modules:
                # construct unique CGP layers for each em type:
                i_module = unique_CGP_layers.index(em_module)
                CGP_layers.append(unique_CGP_layers[i_module])
            else:
                layer = SRLayer(copy.deepcopy(self.sr_param), self.sr_class, em_module)  # autopum算法划分出的模块，在SRLayer中构建不同的层，在这里组成网络
                unique_em_modules.append(em_module)
                unique_CGP_layers.append(layer)
                CGP_layers.append(layer)
        self.explained_layers = nn.ModuleList(CGP_layers)

    def init(self, neural_network):
        self.init_network_info(neural_network)
        self.init_srnet()

    def forward(self, X, y=None):
        """
            X: L * shape=(N, H)
            y: L * shape=(N, H), where L: # modules, N: batch size, H: hidden size
        """
        outputs = []
        input = X
        for cgp in self.explained_layers:
            input = cgp(input)
            outputs.append(input)
        if y is None:
            return outputs

        assert len(outputs) == len(y)
        loss_fn = nn.MSELoss()
        loss = 0
        for cgp_out, target in zip(outputs, y):
            loss += loss_fn(cgp_out, target)
        loss /= len(outputs)
        return loss, outputs

    def expr(self, vars=None):
        out_expr = vars
        for ex_layer in self.explained_layers:
            out_expr = ex_layer.expr(out_expr)
        return out_expr

    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg

    def mutate(self, probability):
        return [
            ex_layer.mutate(probability)
            for ex_layer in self.explained_layers
        ]

    def get_genes(self):
        return [
            ex_layer.genes()
            for ex_layer in self.explained_layers
        ]

    def assign_genes(self, genes):
        for i, layer_genes in enumerate(genes):
            self.explained_layers[i].assign_genes(layer_genes[:])

    def copy_self(self):
        new_module = copy.deepcopy(self)
        return new_module.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device


class ModuleCGPNet(BaseSRNet):
    def __init__(
            self,
            sr_param: CGPParameter,
            neural_network: nn.Module = None,
            in_shape: tuple = None,
            network_info: NetworkInformation = None
    ) -> None:
        super().__init__(CGPModel, sr_param, neural_network, network_info, in_shape)

    def create(self):
        return ModuleCGPNet(
            self.sr_param,
            in_shape=self.in_shape,
            network_info=self.network_info
        ).to(self.device)


class ModuleEQLNet(BaseSRNet):
    def __init__(
            self, sr_param: EQLParameter,
            neural_network: Module = None,
            in_shape: tuple = None,
            network_info: NetworkInformation = None,
    ) -> None:
        super().__init__(EQL, sr_param, neural_network, network_info, in_shape)

    def create(self):
        return ModuleEQLNet(
            self.sr_param,
            in_shape=self.in_shape,
            network_info=self.network_info
        ).to(self.device)

    def regularization(self, type="l1"):
        reg = 0.
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg += torch.sum(torch.abs(param))
        return reg
