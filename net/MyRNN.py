# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 9:43
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch
from torch import nn


class MyRNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers):
        super(MyRNN, self).__init__()
        self.rnn_layer = nn.LSTM(num_inputs, num_hiddens, num_layers=num_layers)
        self.LinearSeq = nn.Sequential(
            nn.Linear(num_hiddens, 256), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(128, 1))

    def forward(self, X):
        X, _ = self.rnn_layer(X)
        # print(f"X shape:{X.shape}")
        LX = self.LinearSeq(X[:, -1, :])
        # print(f"LX shape:{LX.shape}")
        return LX.reshape(-1)


if __name__ == '__main__':
    X = torch.rand((32, 360, 6))
    Net = MyRNN(6, 128, 2)
    print(Net(X).shape)
