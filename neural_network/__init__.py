from .mlp import MLP
from .conv_net import LeNet5, TestConvNet
from .rnn import VanillaRNN


neural_networks = {
    "testConv": TestConvNet,
    "mlp": MLP,
    "MLP": MLP,
    "LeNet": LeNet5,
    "lenet": LeNet5,
    "rnn": VanillaRNN
}