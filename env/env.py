# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 19:12
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
class Env:
    def __init__(self, ):
        super(Env, self).__init__()
        self.window = 120           # 时间序列窗口，输入网络的时间戳长度
        self.data_m = 6             # 时间序列输入维度
        self.hidRNN = 10            # RNN层输出维度
        self.hidCNN = 10            # CNN层输出维度
        self.hidSkip = 5            # Skip-RNN层输出维度
        self.CNN_kernel = 6         # CNN层kernel大小
        self.skip = 38              # 周期长度
        self.highway_window = 24    # highway通道的输出节点数目
        self.dropout = 0.5
