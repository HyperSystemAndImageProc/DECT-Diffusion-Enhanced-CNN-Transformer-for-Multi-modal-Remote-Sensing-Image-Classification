import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import logging
import pandas as pd


# def save_loss_to_excle(losses,column_name):
#     # 训练结束后，尝试读取已存在的Excel文件
#     try:
#         loss_df = pd.read_excel('loss_data.xlsx')
#     except FileNotFoundError:
#         loss_df = pd.DataFrame()
#
#     # 将新的损失数据作为新列添加到DataFrame中
#     # 如果DataFrame是空的，就创建一个新列；否则，添加为下一个可用的列
#     if len(losses) > len(loss_df):
#         # 扩展DataFrame以匹配新数据的长度
#         loss_df = loss_df.reindex(range(len(losses)))
#     if len(losses) < len(loss_df):
#         # 如果新损失数据比原有数据短，用NaN填充剩余的行
#         loss_df[column_name] = loss_df[column_name].reindex(loss_df.index).fillna('NaN')
#     # 将更新后的DataFrame保存回Excel文件
#     loss_df.to_excel('loss_data.xlsx', index=False)
#     return None
def save_loss_to_excle(losses, column_name):
    # 训练结束后，尝试读取已存在的Excel文件
    try:
        loss_df = pd.read_excel('loss_data.xlsx')
    except FileNotFoundError:
        loss_df = pd.DataFrame()

    # 如果DataFrame是空的，或者不存在该列，则创建一个新列
    if column_name not in loss_df.columns:
        loss_df[column_name] = pd.Series(losses)
    else:
        # 将新的损失数据作为新列添加到DataFrame中
        if len(losses) > len(loss_df):
            # 扩展DataFrame以匹配新数据的长度
            loss_df = loss_df.reindex(range(len(losses)))
        # 如果新损失数据比原有数据短，用NaN填充剩余的行
        loss_df[column_name] = pd.Series(losses).reindex(loss_df.index).fillna('NaN')

    # 将更新后的DataFrame保存回Excel文件
    loss_df.to_excel('loss_data.xlsx', index=False)
    return None

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        if not self.closed:
            self.logfile.close()
            self.closed = True


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y1 = y.expand_as(x)
        out = y1
        return out


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


# 3-D权重
class SCA(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SCA, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([58, 138, 71]) / 255.
        if item == 1:
            y[index] = np.array([204, 180, 206]) / 255.
        if item == 2:
            y[index] = np.array([150, 84, 54]) / 255.
        if item == 3:
            y[index] = np.array([251, 193, 150]) / 255.
        if item == 4:
            y[index] = np.array([137, 145, 200]) / 255.
        if item == 5:
            y[index] = np.array([238, 45, 42]) / 255.
        if item == 6:
            y[index] = np.array([86, 132, 193]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 12:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 255]) / 255.
    return y


def generate_png(total_iter, net, gt_hsi, device, total_indices, path):
    pred_test = []
    for X1, XD, X2, y in total_iter:
        # X = X.permute(0, 3, 1, 2)
        X1 = X1.to(device)
        XD = XD.to(device)

        X2 = X2.to(device)

        net.eval()
        pred_test.extend(net(X1, XD, X2).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + '.eps')
    classification_map(y_re, gt_hsi, 300,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '_gt.png')


def generate_all_png(all_iter, net, gt_hsi, device, all_indices, path):
    pred_test = []
    for X1, X2 in all_iter:
        # X = X.permute(0, 3, 1, 2)
        X1 = X1.to(device)
        X2 = X2.to(device)
        net.eval()
        pred_test.extend(net(X1, X2).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[all_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + '.eps')
    classification_map(y_re, gt_hsi, 300,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '_gt.png')
    print('------Get classification maps successful-------')
