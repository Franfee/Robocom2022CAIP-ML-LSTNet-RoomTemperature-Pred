# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 16:53
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


# 导入相关包
import math
import copy
import pandas as pd
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------
from util.Accumulator import Accumulator

# -------data set-------------------
from util.dataLoader import load_iter

# --------net-----------------------
from env.env import Env
from net.LSTNet import LSTNet

# ----------------------------------

# 用于归一化特征,定义在外部以保存它的状态
# 测试集和预测阶段使用的归一化分布与训练时应该统一
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    数据处理
    :param data:数据
    :param n_in:输入特征个数
    :param n_out:目标值
    :param dropnan:是否删除 Nan 值
    :return:
    """
    df = pd.DataFrame(data)
    n_vars = df.shape[1]  # n_vars 列数
    cols, names = list(), list()

    # 时间间隔跨度, 时间点个数，共 n_in 个
    # 首先添加当前时刻之前的时间点
    for i in range(n_in - 1, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 然后添加当前时刻
    cols.append(df)
    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]

    # 添加 target 为未来 n_out 分钟后时刻的温度
    cols.append(df.shift(-n_out))
    names += [('var%d(t+%d)' % (j + 1, n_out)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def processing_data(data_path, validation_split=0.3):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_X,train_y,test_X,test_y
    """
    # --------------- 在这里实现将时序数据转化为监督学习问题的特征数据 ------------------
    print("In processing_data...")
    # 读取数据
    sensor_data = pd.read_csv(data_path, index_col=0)
    # ------check data---------
    # 展示特征值图表
    # sensor_data.plot(subplots=True, figsize=(20, 10), legend=True);

    # 数据处理1 将耗电量进行 差分 处理
    data = copy.deepcopy(sensor_data)
    # DataFrame.shift()函数可以把数据移动指定的位数
    data['power_consumption'] = data['power_consumption'] - data.shift(4)['power_consumption']
    # data.dropna(inplace=True)
    data = data.round(2)
    # 查看数据分布
    # data.plot(subplots=True, figsize=(20, 10), legend=True)
    # plt.show()

    # 确保所有数据是 float32 类型
    data = data.astype('float32')
    print("data shape:", data.shape)

    # 归一化
    scaled_data = scaler.fit_transform(data)
    print("scaled_data shape:", scaled_data.shape)
    # 构建成监督学习数据集
    n_in, n_out = 120, 6
    reframed = series_to_supervised(scaled_data, n_in, n_out, dropnan=True)
    print("reframed shape:", reframed.shape)
    # 丢弃我们不想预测的列,这里预测的是室内温度，丢弃其他列
    if n_in == 120:
        drop_col = [720, 721, 723, 724, 725]
    else:
        drop_col = [-1, -2, -3, -5, -6]
    reframed.drop(reframed.columns[drop_col], axis=1, inplace=True)
    # 把数据分为训练集和测试集
    values = reframed.values
    print("values shape:", values.shape)
    X, y = values[:, :-1], values[:, -1]
    # 把数据分为输入和输出
    train_num = int((1 - validation_split) * reframed.shape[0])
    train_X, train_y = X[:train_num], y[:train_num]
    test_X, test_y = X[train_num:], y[train_num:]

    # 把输入重塑成符合LSTM输入的3D格式 [样例， 时间步, 特征]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_out))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_out))
    print("训练集数据 shape:", train_X.shape)
    print("训练集标签 shape:", train_y.shape)
    print("测试集数据 shape:", test_X.shape)
    print("测试集标签 shape:", test_y.shape)
    # --------------------------------------------------------------------------------------------

    return train_X, train_y, test_X, test_y


def model(train_X, train_y, test_X, test_y, save_model_path):
    """
    创建、训练和保存深度学习模型
    :param train_X: 训练集特征
    :param train_y: 训练集target
    :param test_X: 测试集特征
    :param test_y: 测试集target
    :param save_model_path: 保存模型的路径和名称
    """
    print("In model...")
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    # -----生成数据集-----
    batch_size = 32
    train_iter = load_iter(train_X, train_y, batch_size)
    # ------------------

    # -----参数设置区-----
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    # net = MyRNN(num_inputs=6, num_hiddens=128, num_layers=2)
    net = LSTNet(Env())
    optimizer = torch.optim.Adam(net.parameters())
    lossFun = nn.MSELoss()
    PLOT = True
    # ------------------

    # ----训练过程参数----
    train_loss_lst, test_loss_lst = [], []
    train_rmse_lst, test_rmse_lst = [], []
    train_mae_lst, test_mae_lst = [], []
    # ------------------

    # ------训练过程------
    for epoch in range(epochs):
        # 设置训练模式
        net.train()
        # 转移设备计算
        net.to(DEVICE)
        # 指标累加器
        metric = Accumulator(4)

        # 一轮中的批次
        for X_batch_test, y_batch in train_iter:
            optimizer.zero_grad()
            X_batch_test, y_batch = X_batch_test.to(DEVICE), y_batch.to(DEVICE)

            y_logist = net(X_batch_test)
            loss = lossFun(y_logist, y_batch)
            loss.backward()
            optimizer.step()

            RMSE, MAE = evaluate_model(y_logist.cpu().detach().numpy(), y_batch.cpu().detach().numpy(), scaler)
            metric.add(RMSE * y_batch.numel(), MAE * y_batch.numel(), float(loss.sum()), y_batch.numel())
        # end one train epoch
        train_rmse, train_mae, train_loss = metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]

        # =====eval start one test epoch========
        test_rmse, test_mae, test_loss = evaluate_mode(test_X, test_y, (net, lossFun, DEVICE, epoch))
        # ======================================

        # 训练参数
        train_loss_lst.append(train_loss)
        train_rmse_lst.append(train_rmse)
        train_mae_lst.append(train_mae)
        test_loss_lst.append(test_loss)
        test_rmse_lst.append(test_rmse)
        test_mae_lst.append(test_mae)
        print(f'epoch: {epoch}, train loss: {train_loss:.5f}, train rmse: {train_rmse:.5f}, train mae: {train_mae:.5f}')
        print(f'epoch: {epoch}, test loss : {test_loss:.5f}, test rmse : {test_rmse:.5f}, test mae : {test_mae:.5f}')
    # ------------------
    # 保存模型（请写好保存模型的路径及名称）
    torch.save(net.state_dict(), save_model_path)
    print("model saved.")
    # vis
    if PLOT:
        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_loss_lst)), train_loss_lst, label=u"train_loss_lst")
        plt.plot(range(len(train_rmse_lst)), train_rmse_lst, label=u"train_rmse_lst")
        plt.plot(range(len(train_mae_lst)), train_mae_lst, label=u"train_mae_lst")
        plt.grid(True, linestyle='-.')  # 网格化
        plt.legend()  # 打开label标记
        plt.subplot(1, 2, 2)
        plt.plot(range(len(test_loss_lst)), test_loss_lst, label=u"test_loss_lst")
        plt.plot(range(len(test_rmse_lst)), test_rmse_lst, label=u"test_rmse_lst")
        plt.plot(range(len(test_mae_lst)), test_mae_lst, label=u"test_mae_lst")
        plt.grid(True, linestyle='-.')  # 网格化
        plt.legend()  # 打开label标记
        plt.show()
    # --------------------------------------------------------------------------------------------


def evaluate_mode(test_X, test_y, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_X: 测试集特征
    :param test_y: 测试集target
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    print("In evaluate_mode!!!")
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    # 加载模型和计算
    batch_size = 32
    test_iter = load_iter(test_X, test_y, batch_size)
    net, lossFun, DEVICE, epoch = save_model_path

    # 设置评估模式
    net.eval()
    # 转移设备计算
    net.to(DEVICE)
    # 指标累加器
    metric = Accumulator(4)
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_iter:
            X_batch_test, y_batch_test = X_batch_test.to(DEVICE), y_batch_test.to(DEVICE)
            # predicted_data = np.array(predict(test_X))
            predicted_data = net(X_batch_test)
            loss = lossFun(predicted_data, y_batch_test)
            # evaluate_model(predicted_data, test_y, scaler)
            RMSE, MAE = evaluate_model(predicted_data.cpu().detach().numpy(), y_batch_test.cpu().detach().numpy(), scaler)
            metric.add(RMSE * y_batch_test.numel(), MAE * y_batch_test.numel(), float(loss.sum()), y_batch_test.numel())

    # 评估参数
    test_rmse, test_mae, test_loss = metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]
    return test_rmse, test_mae, test_loss
    # ---------------------------------------------------------------------------


def evaluate_model(predicted_data, true_data, scaler, printlog=False, plot=False):
    """
    模型预测值与真实值处理，获取 RMSE、MAE 等评价指标信息
    :param predicted_data:预测值，一维向量
    :param true_data:真实值，一维向量
    :param scaler:归一化处理对象
    :return:
    """
    if printlog:
        print("In evaluate_model...")
        print(f"predicted_data.shape:{predicted_data.shape},true_data.shape:{true_data.shape}")
    assert predicted_data.shape == true_data.shape
    predicted_data = predicted_data.reshape(predicted_data.shape[0], 1)
    true_data = true_data.reshape(true_data.shape[0], 1)

    # scaler只能对整体反归一化，构造两个空矩阵
    temp_array_1 = np.ones((len(predicted_data), 2))
    temp_array_2 = np.ones((len(predicted_data), 3))

    # 反归一化
    predicted_seq = np.concatenate((temp_array_1, predicted_data, temp_array_2), axis=1)
    predicted_seq = scaler.inverse_transform(predicted_seq)
    predicted_data = predicted_seq[:, 2]

    true_seq = np.concatenate((temp_array_1, true_data, temp_array_2), axis=1)
    true_seq = scaler.inverse_transform(true_seq)
    true_data = true_seq[:, 2]

    # 画出预测图形
    if plot:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        ax.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show()

    # 计算RMSE,MAE
    rmse = math.sqrt(mean_squared_error(true_data, predicted_data))
    mae = mean_absolute_error(true_data, predicted_data)
    if printlog:
        print("RMSE:{}".format(round(rmse, 3)))
        print("MAE: {}".format(round(mae, 3)))
    else:
        return rmse, mae


def simple_predict(sequences):
    # 朴素法使用序列最后一个周期作为预测值
    results = []
    for s in range(sequences.shape[0]):
        results.append(sequences[s, -1, -4])
    return results


def mean_predict(sequences):
    # 使用均值法进行预测
    results = []
    for s in range(sequences.shape[0]):
        results.append(np.mean(sequences[s, :, -4]))
    return results


def drift_predict(sequences):
    # 使用飘移法预测
    results = []
    for s in range(sequences.shape[0]):
        l_boundary = sequences[s, 0, -4]
        r_boundary = sequences[s, -1, -4]
        l_time = 1
        r_time = 120
        pre_time = r_time + 60
        predict = (r_boundary - l_boundary) / (r_time - l_time) * (pre_time - l_time) + l_boundary
        results.append(predict)
    return results


def ModelCompare(test_X, test_y):
    # =============================对比区====================================
    # -------------------------
    # 可以看到朴素法预测值是真实值滞后 60 分钟的结果
    print("-" * 20 + "simple_predict:" + "-" * 20)
    predicted_data = np.array(simple_predict(test_X))
    evaluate_model(predicted_data, test_y, scaler, printlog=True)
    print("-" * 55 + "\n")

    # -----------------------
    # 可以发现均值法有更严重的滞后性
    print("-" * 20 + "mean_predict:" + "-" * 20)
    predicted_data = np.array(mean_predict(test_X))
    evaluate_model(predicted_data, test_y, scaler, printlog=True)
    print("-" * 55 + "\n")

    # -----------------------
    # 可以发现漂移法在平缓上升或下降序列上有优于朴素法的预测能力，但是对于温度变化较大的情况无法很好处理，且异常值较多
    print("-" * 20 + "drift_predict:" + "-" * 20)
    predicted_data = np.array(drift_predict(test_X))
    evaluate_model(predicted_data, test_y, scaler, printlog=True)
    print("-" * 55 + "\n")
    # ========================================================================


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以进行测试提交!
    :return:
    """
    data_path = "dataset/dataset.csv"  # 数据集路径
    save_model_path = "results/model.mdl"  # 保存模型路径和名称
    validation_split = 0.2  # 验证集比重

    # 获取数据、并进行预处理
    train_X, train_y, test_X, test_y = processing_data(data_path, validation_split=validation_split)

    # 创建、训练和保存模型
    TRAIN_SWITCH = True
    if TRAIN_SWITCH:
        model(train_X, train_y, test_X, test_y, save_model_path)

    # 评估模型
    EVAL_SWITCH = False     # (训练过程已经包含,save_model_path定义不同)
    if EVAL_SWITCH:
        evaluate_mode(test_X, test_y, save_model_path)

    # 模型对比
    CMP_SWITCH = False
    if CMP_SWITCH:
        ModelCompare(test_X, test_y)


if __name__ == '__main__':
    main()
