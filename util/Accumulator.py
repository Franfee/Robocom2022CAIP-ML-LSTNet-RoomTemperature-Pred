# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 19:14
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
