import torch
from torch import nn
import math
from neural_network import MLP,LeNet5


# 公共模块
official_modules = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.MaxPool1d, nn.MaxPool2d,
    nn.Softmax, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
    nn.Sigmoid, nn.Tanh, nn.ReLU, nn.GELU, nn.Dropout
)
# 激活函数模块
activation_modules = (
    nn.Sigmoid, nn.Tanh, nn.ReLU, nn.GELU, nn.Softmax, nn.Dropout
)
# 归一化模块
norm_modules = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm
)
# 卷积模块
conv_modules = (
    nn.Conv1d, nn.Conv2d, nn.MaxPool1d, nn.MaxPool2d
)
# 神经网络模块的类型，在论文表4.2中体现
# Base：单个网络层
# Seq：多个网络层
# Ident：自定义子网络
# Official：公共模块
nm_types = {
    "Base", "Seq", "Ident", "Official",
}

# 解释模块的类型（神经网络模块中的每一层作用的索引）
em_types = {
    "Activation": 0,
    "Linear": 1,
    "Conv": 2,
    "Norm": 3,
    "Ident": 4,
    "Seq-Linear": 5,
    "Seq-Conv": 6,
    "Seq-Ident": 7,
    "Ident-Linear": 8,
    "Ident-Conv": 9,
    "Other": 10
}


# 定义Base类型的神经网络模块
class NeuralModule:
    nm_type: str = "Base"
    in_shape: tuple = None
    out_shape: tuple = None

    def __init__(self, nm_type, in_shape=None, out_shape=None) -> None:
        self.nm_type = nm_type
        self.modules = []
        self.in_shape = in_shape
        self.out_shape = out_shape

    def add_module(self, module):
        self.modules.append(module)

    # 判断两个网络模块的类型是否相等，在使用==进行比较时会调用这个方法
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, NeuralModule):
            return False

        if self.nm_type != __value.nm_type:
            return False

        if self.nm_type in ["Seq", "Ident"]:
            if len(self.modules) != len(__value.modules):
                return False
            for nm1, nm2 in zip(self.modules, __value.modules):
                if nm1 != nm2:
                    # 递归
                    return False
        return True

    # 在print的时候会调用__repr__方法
    def __repr__(self) -> str:
        return f"{self.nm_type}: {self.modules}"


# 定义要解释的神经网络模块
class ExplainedModule:
    def __init__(
            self,
            in_shape,
            module_name,
            out_shape=None,
            modules=None,
            em_type=None
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.module_name = module_name
        self.modules = [] if modules is None else modules
        self.em_type = em_type
        self.em_type_id = None

    # 添加模块函数，添加的是上面的神经网络模块，因为要解释它
    def add_module(self, module: NeuralModule):
        self.modules.append(module)
        em_type = None
        # 因为Seq代表的是多个网络层模块，所以要看它的子模块网络类型，并更改相应的网络层类型
        if module.nm_type == "Seq":
            sub_modules = [nm.nm_type for nm in module.modules]
            if "Ident" in sub_modules:
                em_type = "Seq-Ident"
            elif "Conv" in sub_modules:
                em_type = "Seq-Conv"
            else:
                em_type = "Seq-Linear"
        elif module.nm_type == "Ident":
            sub_modules = [nm.nm_type for nm in module.modules]
            if "Conv" in sub_modules:
                em_type = "Ident-Conv"
            elif "Linear" in sub_modules:
                em_type = "Ident-Linear"
            else:
                em_type = "Ident"
        else:
            em_type = module.nm_type
        self.em_type = em_type
        self.em_type_id = em_types[self.em_type]

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExplainedModule):
            return False

        if self.em_type != __value.em_type:
            return False
        if self.in_shape != __value.in_shape or self.out_shape != __value.out_shape:
            return False

        for self_module, target_module in zip(self.modules, __value.modules):
            if self_module != target_module:
                return False
        return True

    def equals(self, mi):
        return mi.modules == self.modules

    def __repr__(self) -> str:
        return f"{self.module_name}: (\n" + \
            f"\t   In_shape={self.in_shape}\n" + \
            f"\t   Out_shape={self.out_shape}\n" + \
            f"\t   Modules={self.modules}\n\t)"


# 整合神经网络各个模块的信息
class NetworkInformation:
    em_type_2_id = {
        k: 0 for k in em_types.keys()
    }

    def __init__(
            self,
            in_shape,
            out_shape=None,
            modules=None
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.modules = [] if modules is None else modules
        self.module_types = []

    # 添加模块函数，添加的是上面的解释模块
    def add_module(self, ex_module: ExplainedModule):
        is_in = False
        for i, em in enumerate(self.modules):
            if em == ex_module:
                self.module_types.append(self.module_types[i])
                is_in = True
                break
        if not is_in:
            self.module_types.append(ex_module.em_type + f"-{self.em_type_2_id[ex_module.em_type]}")
            self.em_type_2_id[ex_module.em_type] += 1
        self.modules.append(ex_module)

    def __repr__(self) -> str:
        s = f"input_shape:{self.in_shape};\n" + \
            f"output_shape:{self.out_shape};\n"
        module_s = "Modules: ["
        for i, module in enumerate(self.modules):
            module_s += f"\n\t「{self.module_types[i]}」" + str(module)
        s = s + module_s + "\n]\n"
        s = s + f"Module_types: {self.module_types}"
        return s


# 模块划分算法的主要部分
class Parser:

    @staticmethod
    def parse_mlp_module(module: nn.Module):
        neurons = []
        if isinstance(module, nn.Sequential):
            for linear in module:
                if isinstance(linear, nn.Linear):
                    neurons.append(linear.out_features)
        elif isinstance(module, nn.Linear):
            neurons.append(module.out_features)
        return neurons

    @staticmethod
    def parse_mlp_network(network: nn.Module):
        neurons = []
        for module in network.children():
            neurons.extend(Parser.parse_mlp_module(module))  # 将上面解析mlp模块函数返回的神经元列表分别加进这里
        return neurons

    @staticmethod
    # 解析神经网络中的不同层，并返回其输出形状
    def parse_layer(in_shape: tuple, layer: nn.Module):

        # 计算卷积层的输出形状，dila一般为1
        def conv_out(in_d, k, s, p, dila):
            return math.ceil((in_d + 2 * p - dila * (k - 1) - 1) / s + 1)

        # 对于线性层，计算输出特征数，返回形状
        if isinstance(layer, nn.Linear):
            if len(in_shape) > 1:
                out_features = math.prod(in_shape)
            else:
                out_features = in_shape[-1]
            assert out_features == layer.in_features
            shape = (layer.out_features,)
        # 对于卷积层，计算卷积后的形状
        elif isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d):
            # in_shape should be (in_C, L) or (in_C, H, W)
            in_C = layer.in_channels
            out_C = layer.out_channels
            assert len(in_shape) in [2, 3] and in_C == in_shape[0]

            k_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation

            shape = (out_C, conv_out(in_shape[1], k_size[0], stride[0], padding[0], dilation[0]))
            if len(in_shape) == 3:
                shape = shape + (conv_out(in_shape[-1], k_size[1], stride[1], padding[1], dilation[1]),)
        # 对于池化层，跟卷积层一样
        elif isinstance(layer, nn.MaxPool1d) or isinstance(layer, nn.MaxPool2d):
            assert len(in_shape) in [2, 3]
            out_C = in_shape[0]

            shape = (
                out_C,
                conv_out(in_shape[1], layer.kernel_size, layer.stride, layer.padding, layer.dilation)
            )
            if len(in_shape) == 3:
                shape = shape + (
                    conv_out(in_shape[-1], layer.kernel_size, layer.stride, layer.padding, layer.dilation),)
        # 对于其他层，假设不改变形状
        else:
            # maybe other module such as activation or Softmax...
            # assume that these modules can not change the shape...
            shape = in_shape
        return shape

    @staticmethod
    def parse_neural_module(in_shape, module: nn.Module):
        # 如果是公共模块，就分类并返回
        if isinstance(module, official_modules):
            if isinstance(module, activation_modules):
                name = "Activation"
            elif isinstance(module, conv_modules):
                name = "Conv"
            elif isinstance(module, norm_modules):
                name = "Norm"
            else:
                name = module.__class__.__name__
            nm = NeuralModule(name)
            nm.add_module(module.__class__.__name__)
            nm.out_shape = Parser.parse_layer(in_shape, module)
            return nm
        # 对于嵌套的多个网络Seq，递归解析
        if isinstance(module, nn.Sequential):
            name = "Seq"
            iterator = module
        else:
            name = "Ident"
            iterator = module.children()

        nm = NeuralModule(name)
        out_shape = in_shape
        for next_module in iterator:
            tmp_nm = Parser.parse_neural_module(out_shape, next_module)
            nm.add_module(tmp_nm)
            out_shape = tmp_nm.out_shape
        nm.out_shape = out_shape
        return nm

    @staticmethod
    # 解析整个神经网络
    def parser_network(in_shape, network: nn.Module):
        network_info = NetworkInformation(in_shape)
        out_shape = in_shape
        # 迭代网络中的所有子模块，获取它们的名字
        for name, module in network.named_children():
            # get module's information
            nm = Parser.parse_neural_module(out_shape, module)
            ex_module = ExplainedModule(out_shape, name)
            ex_module.add_module(nm)
            ex_module.out_shape = nm.out_shape
            network_info.add_module(ex_module)
            out_shape = nm.out_shape
        network_info.out_shape = out_shape
        return network_info


if __name__ == '__main__':
    # 测试mlp的模块划分
    # inshape=(2,)
    # mlp=MLP(in_features=2,out_features=1)
    # mlp_info=Parser.parser_network(inshape,mlp)
    # print(mlp_info)

    # 测试lenet的模块划分
    inshape=(1,32,32)
    lenet=LeNet5(num_classes=10)
    lenet_info=Parser.parser_network(inshape,lenet)
    print(lenet_info)
