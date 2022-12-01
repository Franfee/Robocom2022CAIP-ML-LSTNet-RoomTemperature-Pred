# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 16:52
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import numpy as np
import pandas as pd
import copy

import torch
from sklearn import preprocessing

from net.LSTNet import LSTNet, Env

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择


def load_model(model_path):
    # net = MyRNN(num_inputs=6, num_hiddens=128, num_layers=2)
    net = LSTNet(Env())
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return net


# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 model.mdl 模型，则 model_path = 'results/demo.h5'
model_path = 'results/model.mdl'

# 加载模型，如果采用 keras 框架训练模型，则 model=load_model(model_path)
print("model loading...")
model = load_model(model_path)
model.eval()
model.to(DEVICE)
print("model loaded.")


# 如果使用了归一化，需要确保预测阶段的归一化和反归一化使用的分布相同
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# ---------------------------------------------------------------------------


def predict(sequence):
    """
    对输入序列进行预测，sequence为一360分钟的时序数据，
    输出结果为该序列第 60 分钟后的预测温度值
    param: sequence: np.array矩阵 shape:[360,6] [时间步, 特征]，
                     其中特征的索引顺序与数据集相同，分别为:
                     [outdoor_temperature,outdoor_humidity,indoor_temperature,
                      indoor_humidity,fan_status,power_consumption]
    return: 温度（标量），浮点数表示，限定使用np.float64或者python的float类型
    """
    # ----------------------- 实现预测部分的代码，以下样例可代码自行删除 -----------------------
    # 数据处理
    sensor_data = pd.DataFrame(sequence,
                               columns=['outdoor_temperature', 'outdoor_humidity', 'indoor_temperature',
                                        'indoor_humidity', 'fan_status', 'power_consumption'])
    # 数据处理1 将耗电量进行 差分 处理
    data = copy.deepcopy(sensor_data)
    # DataFrame.shift()函数可以把数据移动指定的位数
    data['power_consumption'] = data['power_consumption'] - data.shift(4)['power_consumption']
    data = data.round(2)
    # 确保所有数据是 float32 类型
    data = data.astype('float32')
    print("data shape:", data.shape)
    # 归一化
    scaled_data = scaler.fit_transform(data)
    print("scaled_data shape:", scaled_data.shape)

    # 转为tensor
    seq_in_X = torch.tensor(scaled_data, dtype=torch.float32).reshape(-1, 360, 6)
    print(f"seq_X_in shape:{seq_in_X.shape}")
    seq_in_X = seq_in_X.to(DEVICE)

    # 模型预测
    with torch.no_grad():
        predicted_data = model(seq_in_X)

    # 转为np
    predicted_data = predicted_data.cpu().detach().numpy()
    print(f"predicted_data shape:{predicted_data.shape}")
    print(f"predicted_data :{predicted_data}")
    predicted_data = predicted_data[2].reshape(1, 1)
    print(f"predicted_data shape:{predicted_data.shape}")
    print(f"predicted_data :{predicted_data}")
    # scaler只能对整体反归一化，构造两个空矩阵
    temp_array_1 = np.ones((len(predicted_data), 2))
    temp_array_2 = np.ones((len(predicted_data), 3))

    # 反归一化
    predicted_seq = np.concatenate((temp_array_1, predicted_data, temp_array_2), axis=1)
    predicted_seq = scaler.inverse_transform(predicted_seq)
    pre_temperature = predicted_seq[:, 2]
    # -------------------------------------------------------------------------------------
    return float(pre_temperature)


if __name__ == '__main__':
    test = np.random.uniform(25, 28, size=(360, 6))
    # test = torch.tensor(test, dtype=torch.float32).to(DEVICE)
    print(predict(test))
