import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.env import Env


class LSTNet(nn.Module):
    def __init__(self, args_in):
        super(LSTNet, self).__init__()
        self.window = args_in.window
        self.data_m = args_in.data_m
        self.hidR = args_in.hidRNN
        self.hidC = args_in.hidCNN
        self.hidS = args_in.hidSkip
        self.Ck = args_in.CNN_kernel
        self.skip = args_in.skip
        # window在kernel作用下，以skip为周期的数据数量。周期数目。
        self.pt = int((self.window - self.Ck) / self.skip)
        self.hw = args_in.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.data_m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args_in.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.data_m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.data_m)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = nn.ReLU()
        self.final = nn.Linear(self.data_m, 1)

    def forward(self, x):
        x = x.reshape(-1, self.window, self.data_m)
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.window, self.data_m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.data_m)
            res = res + z

        res = self.final(res)
        if self.output:
            res = self.output(res)
        return res.reshape(-1)


if __name__ == '__main__':
    Net = LSTNet(Env())

    X = torch.rand((32, 120, 6))
    print(Net(X).shape)

    test = np.random.uniform(25, 28, size=(360, 6)).astype(float)
    test = torch.from_numpy(test)
    print(Net(test))
