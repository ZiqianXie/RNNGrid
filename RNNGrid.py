import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.functional import conv2d


class RNNGridLayer(nn.Module):
    def __init__(self, in_channels, num_hidden, kernel_size, distance, time_steps, loss_weight):
        """
        direction coding:
            0   1   2
            3   4   5
            6   7   8
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.dist = distance
        self.num_hidden = num_hidden
        self.time_steps = time_steps
        # self.epsilon = 1e-9
        moves = distance * 2 + 1
        self.movesq = moves**2
        self.conv_ih = Conv2d(in_channels, 3*num_hidden, kernel_size, padding=kernel_size//2)
        self.conv_hh = Conv2d(num_hidden, 3*num_hidden, 1)
        self.conv_ho = Conv2d(num_hidden, self.movesq, 1)
        self.merge_kernel = torch.zeros(1, self.movesq, moves, moves, requires_grad=False)
        self.gather_kernel = torch.zeros(self.movesq, 1, moves, moves, requires_grad=False)
        for i in range(self.movesq):
            height = i // moves
            width = i % moves
            self.merge_kernel[0, self.movesq-i-1, height, width] = 1
            self.gather_kernel[i, 0, height, width] = 1

    def forward(self, x, mask, init_hidden=None):
        # taking from https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/cells.py
        Loss = torch.nn.BCELoss()
        num_batch = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        loss = torch.tensor([0.])
        i_r, i_i, i_n = self.conv_ih(x).chunk(3, 1)
        if init_hidden:
            hiddens = [init_hidden]
        else:
            hiddens = [torch.zeros(num_batch, self.num_hidden, height, width)]
        outs = []
        for _ in range(self.time_steps):
            h_r, h_i, h_n = self.conv_hh(hiddens[-1]).chunk(3, 1)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hiddens[-1] - newgate)
            o = torch.softmax(self.conv_ho(hy), 1)
            diff = (conv2d(o[:, 4:5], self.gather_kernel, padding=1) - o)**2 * o
            loss = loss + Loss(1 - o[:, 4], mask) + diff * self.loss_weight
            outs.append(o)
            # merged_o = conv2d(o, self.merge_kernel, padding=self.dist) + self.epsilon
            hxo = torch.einsum("bcwh, bdwh -> bcdwh", hy, o).view(num_batch, self.movesq*self.num_hidden, width, height)
            conv2d(mask, self.merge_kernel)
            hiddens.append(conv2d(hxo, self.merge_kernel.repeat(self.num_hidden, 1, 1, 1),
                                  padding=self.dist, groups=self.num_hidden))
        return hiddens[1:], outs, loss
