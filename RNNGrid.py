import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.functional import conv2d


class RNNGridLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, num_hidden, distance):
        """
        direction coding:
            0   1   2
            3   4   5
            6   7   8
        """
        super().__init__()
        self.dist = distance
        self.epsilon = 1e-9
        moves = distance * 2 + 1
        movesq = moves**2
        self.conv_ih = Conv2d(in_channels, 3*num_hidden, kernel_size, padding=kernel_size//2)
        self.conv_hh = Conv2d(num_hidden, 3*num_hidden, 1)
        self.conv_ho = Conv2d(num_hidden, movesq, 1)
        self.conv_oo = Conv2d(moves, movesq, 1, bias=False)
        self.merge_kernel = torch.zeros(1, movesq, moves, moves, requires_grad=False)
        for i in range(movesq):
            height = i // kernel_size
            width = i % kernel_size
            self.merge_kernel[0, movesq-i-1, height, width] = 1

    def forward(self, x, hidden):
        # taking from https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/cells.py
        i_r, i_i, i_n = self.conv_ih(x).chunk(3, 1)
        h_r, h_i, h_n = self.conv_hh(hidden).chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        o = torch.softmax(self.conv_ho(hy), 1)
        merged_o = conv2d(o, self.merge_kernel, padding=self.dist) + self.epsilon
        hxo = torch.einsum("bcwh, bdwh -> bcdwh", x, o).view(x.shape[0], o.shape[1]*x.shape[1], x.shape[2], x.shape[3])
        merged_h = conv2d(hxo, )



