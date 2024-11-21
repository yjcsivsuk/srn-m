import torch
from torch import nn


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        r_out, h_n = self.rnn(x, h0)  # rnn隐藏的输出形状是(N,L,h)，其中h是隐藏层神经元的数量
        r_t_out = self.fc(r_out)  # 加了一个输出，用来表示每个时间步下的输出
        fc_out = r_t_out[:, -1, :]  # 最后用于预测的输出
        # print(r_out[:, -1, :] == h_n[0, :, :])  # True
        # r_out最后一层就是h_n，所以h_n其实可以不用输出，之后重点分析r_out中每一中间维度，即每个时间步下的状态
        return {
            "r_out": r_out,  # (N,L,h)
            "h_n": h_n,  # (num_layers,N,h)
            "r_t_out": r_t_out,  # (N,L,output_size)
            "fc_out": fc_out  # (N,ouput_size)
        }
