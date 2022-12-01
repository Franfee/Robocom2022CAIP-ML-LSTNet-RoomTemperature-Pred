# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 19:18
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import torch
from torch.utils import data


def load_iter(X, y, batch_size=32):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    dataset = data.TensorDataset(X, y)
    return data.DataLoader(dataset, batch_size, shuffle=False)
