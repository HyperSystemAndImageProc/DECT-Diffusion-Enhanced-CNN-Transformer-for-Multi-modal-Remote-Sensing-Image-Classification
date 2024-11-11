import argparse
import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from torch.backends import cudnn
import geniter_2H2L
import Utils
from Dataload import loadData
import warnings
import DECT

warnings.filterwarnings("ignore", message="Argument interpolation should be")
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Trento', 'Houston', 'Berlin', 'Muufl', 'Augsburg'], default='Muufl',
                    help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='test', help='testing mark')
parser.add_argument('--mode', choices=['DEMT'], default='DEMT',
                    help='mode choice')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='number of batch size')
parser.add_argument('--Patch_size', type=int, default=13, help='number of patches')
parser.add_argument('--lidar_d', type=int, default=1, help='spectral of lidar')
parser.add_argument('--Pca_Components', type=int, default=30, help='number of related band')  # 5 10  15 20 25 30
parser.add_argument('--Epoch', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')  # 1e-5,5e-5, 1e-4, 5e-4, 1e-3, 5e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--lidar', choices=['lidar', 'nolidar', 'bott', '0'], default='updatatest1', help='lidar')

args = parser.parse_args()
torch.cuda.synchronize()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False


def num_classe():
    global num_classes
    if args.dataset == 'Trento':
        num_classes = 6
    elif args.dataset == 'Houston':
        num_classes = 15
    elif args.dataset == 'Houston2018':
        num_classes = 20
    elif args.dataset == 'Muufl':
        num_classes = 11
    elif args.dataset == 'Berlin':
        num_classes = 8
    elif args.dataset == 'Augsburg':
        num_classes = 7
    return num_classes


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
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


def sampling1(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m + 1):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i
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
    for i in range(m + 1):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def select_traintest(groundTruth, dataset):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))
    if dataset == 'Trento':
        amount = [50, 100, 80, 50, 70, 90]  # 六类
    elif dataset == 'Muufl':
        amount = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    elif dataset == 'Houston':
        amount = [100, 100, 80, 50, 70, 90, 100, 150, 150, 170, 90, 100, 80, 20, 70]  # 15类
    elif dataset == 'Houston2018':
        amount = [100, 100, 80, 50, 70, 90, 100, 150, 150, 170, 90, 100, 80, 20, 70, 50, 50, 50, 50, 50]  # 15类
    elif dataset == 'Berlin':
        amount = [50, 100, 80, 50, 70, 90, 50, 70]  # 8类
    elif dataset == 'Augsburg':
        amount = [300, 800, 300, 1000, 50, 300, 180]  # 7类
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]
    #    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def create_data_loader(dataset):
    # 读入数据
    # data_HSI, data_Diff, data_lidar, labels
    global ALL_SIZE
    X1, XD, X2, y = loadData(dataset)
    # 每个像素周围提取 patch 的尺寸
    patch_size = args.Patch_size
    PATCH_LENGTH = int((patch_size - 1) / 2)
    if dataset == 'Trento':
        TOTAL_SIZE = 30214
        ALL_SIZE = 99600
    elif dataset == 'Muufl':
        # Muufl
        TOTAL_SIZE = 53687
        ALL_SIZE = 71500
    elif dataset == 'Houston':
        # Houston
        TOTAL_SIZE = 15029
        ALL_SIZE = 664845
    elif dataset == 'Houston2018':
        # Houston
        TOTAL_SIZE = 2181544
        ALL_SIZE = 1432784
    elif dataset == 'Berlin':
        TOTAL_SIZE = 464671
        ALL_SIZE = 820148
    elif dataset == 'Augsburg':
        TOTAL_SIZE = 78294
        ALL_SIZE = 161020
    # 使用 PCA 降维，得到主成分的数量
    pca_components = args.Pca_Components

    print('Hyperspectral data shape: ', X1.shape)
    print('Hyperspectral dataDiff shape: ', XD.shape)
    print('Lidar data shape: ', X2.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('Data shape after PCA: ', X1.shape)
    XD = applyPCA(XD, numComponents=pca_components)
    print('DataD shape after PCA: ', X1.shape)

    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(int)
    CLASSES_NUM = max(gt)
    print(CLASSES_NUM)

    print('\n... ... create train & test data ... ...')
    train_indices, test_indices = select_traintest(gt, dataset=dataset)
    # train_indices, test_indices = sampling(0.95, gt)

    _, all_indices = sampling1(1, gt)
    _, total_indices = sampling(1, gt)
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)

    # 将数据变换维度： [m,n,k]->[m*n,k]
    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    XD_all_data = XD.reshape(np.prod(XD.shape[:2]), np.prod(XD.shape[2:]))

    # 数据标准化
    X1_all_data = preprocessing.scale(X1_all_data)
    XD_all_data = preprocessing.scale(XD_all_data)
    # y=preprocessing.scale(y)

    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    data_XD = XD_all_data.reshape(XD.shape[0], XD.shape[1], XD.shape[2])

    whole_data_X1 = data_X1
    whole_data_XD = data_XD

    padded_data_X1 = np.lib.pad(whole_data_X1, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                                'constant', constant_values=0)
    padded_data_XD = np.lib.pad(whole_data_XD, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                                'constant', constant_values=0)

    if args.dataset == 'Augsburg':
        X2_all_data = X2.reshape(np.prod(X2.shape[:2]), np.prod(X2.shape[2:]))
        X2_all_data = preprocessing.scale(X2_all_data)
        data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1], X2.shape[2])
        whole_data_X2 = data_X2
        padded_data_X2 = np.lib.pad(whole_data_X2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                                    'constant', constant_values=0)
        train_iter, test_iter, total_iter, all_iter = geniter_2H2L.generate_iter_AuBer(
            TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
            ALL_SIZE, all_indices, whole_data_X1, whole_data_XD, whole_data_X2, PATCH_LENGTH, padded_data_X1,
            padded_data_XD, padded_data_X2,
            pca_components, args.BATCH_SIZE, gt)
    else:
        X2_all_data = X2.reshape(np.prod(X2.shape[:2]), )
        X2_all_data = preprocessing.scale(X2_all_data)
        data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1])
        whole_data_X2 = data_X2
        padded_data_X2 = np.lib.pad(whole_data_X2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH)),
                                    'constant', constant_values=0)
        train_iter, test_iter, total_iter, all_iter = geniter_2H2L.generate_iter(
            TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
            ALL_SIZE, all_indices, whole_data_X1, whole_data_XD, whole_data_X2, PATCH_LENGTH, padded_data_X1,
            padded_data_XD, padded_data_X2,
            pca_components, args.BATCH_SIZE, gt)

    print('\n-----Selecting Small Cube from the Original Cube Data-----')

    ratio = TRAIN_SIZE / TOTAL_SIZE
    ratio = "{:.3f}".format(ratio)
    return train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, ratio


def train(train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    total_loss = 0
    losses = []  # 创建一个空列表来存储损失值
    for epoch in range(epochs):
        net.train()
        for i, (data1, dataD, data2, target) in enumerate(train_loader):
            data1, dataD, data2, target = data1.to(device), dataD.to(device), data2.to(device), target.to(device)
            outputs = net(data1, dataD, data2)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # scheduler.step()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.5f]' % (
            epoch + 1, total_loss / (epoch + 1), loss.item()))
        losses.append(loss.item())  # 将损失值添加到列表中
        if epoch > 90 and loss.item() < 0.0001:
            print('Training completed due to loss < 0.0002')
            break
    print('Finished Training')

    return net, device, losses


def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for (data1, dataD, data2, labels) in test_loader:
        data1, dataD, data2, = data1.to(device), dataD.to(device), data2.to(device)
        outputs = net(data1, dataD, data2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    global target_names
    if args.dataset == 'Muufl':
        target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                        'Building shadow', 'Building',
                        'Sidewalk', 'Yellow curb', 'Cloth panels']
    elif args.dataset == 'Trento':
        target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads']
    elif args.dataset == 'Houston':
        target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads', 'Apple Tree', 'Building',
                        'Ground', 'Wood', 'Vineyard', 'Building', 'Ground', 'Wood', 'Vineyard']
    elif args.dataset == 'Berlin':
        target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads', 'Apple Tree', 'Building', ]
    elif args.dataset == 'Augsburg':
        target_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area',
                        'Water']

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, ratio = create_data_loader(args.dataset)
    # 网络放到GPU上
    num_class = num_classe()
    if args.mode == 'DEMT':
        net = DECT.DEMTnet(num_classes=num_class, patch=args.Patch_size, num_patches=args.lidar_d).to(device)
    print('train iter:', len(train_iter))
    if args.flag_test == 'train':
        for i in range(1):
            tic1 = time.perf_counter()
            net, device, losses = train(train_iter, epochs=args.Epoch)
            # 只保存模型参数
            torch.save(net.state_dict(),
                       'result_weight/' + args.mode + '/' + args.dataset + '/' + '{}{}_{}epoch_{}ratio_{}pca_{}patch_params{}.pth'.format(
                           args.lidar,
                           args.dataset,
                           args.Epoch,
                           ratio,
                           args.Pca_Components,
                           args.Patch_size,
                           i))
            toc1 = time.perf_counter()
            tic2 = time.perf_counter()
            y_pred_test, y_test = test(device, net, test_iter)
            toc2 = time.perf_counter()
            # 评价指标b
            classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
            classification = str(classification)
            Training_Time = toc1 - tic1
            Test_time = toc2 - tic2
            file_name = "result_report/{}/{}/{}{}_{}epochs_{}ratio_{}pca_{}patch_classification_report{}.txt".format(
                args.mode,
                args.dataset,
                args.lidar,
                args.dataset,
                args.Epoch,
                ratio,
                args.Pca_Components,
                args.Patch_size, i)

            with open(file_name, 'w') as x_file:
                x_file.write('{} Training_Time (s)'.format(Training_Time))
                x_file.write('\n')
                x_file.write('{} Test_time (s)'.format(Test_time))
                x_file.write('\n')
                x_file.write('{} Kappa accuracy (%)'.format(kappa))
                x_file.write('\n')
                x_file.write('{} Overall accuracy (%)'.format(oa))
                x_file.write('\n')
                x_file.write('{} Average accuracy (%)'.format(aa))
                x_file.write('\n')
                x_file.write('{} Each accuracy (%)'.format(each_acc))
                x_file.write('\n')
                x_file.write('{}'.format(classification))
                x_file.write('\n')
                x_file.write('{}'.format(confusion))
            print('------Get classification results successful-------')
            print(f'Kappa Coefficient: {kappa:.2f}')
            print(f'Overall Accuracy: {oa:.2f}')
            print(f'Average accuracy: {aa:.2f}')

            path = 'result_map/{}/{}/{}{}_{}epochs_{}ratio_{}pca_{}patch_map{}'.format(
                args.mode,
                args.dataset,
                args.lidar,
                args.dataset,
                args.Epoch,
                ratio,
                args.Pca_Components,
                args.Patch_size,
                i)
            if not os.path.exists(path):
                os.makedirs(path)
            path = path + '/' + args.dataset
            Utils.generate_png(
                total_iter, net, y, device, total_indices, path)

    if args.flag_test == 'test':
        if args.dataset == 'Muufl':
            net.load_state_dict(
                torch.load('result_weight/DEMT/Muufl/5e-4finalMuufl_100epoch_0.031ratio_30pca_13patch_params0.pth'))
        elif args.dataset == 'Muufl120':
            net.load_state_dict(
                torch.load('result_weight/DEMT/Muufl/updatatestMuufl_120epoch_0.031ratio_30pca_13patch_params0.pth'))

        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_iter)
        toc2 = time.perf_counter()
        # 评价指标b
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        Test_time = toc2 - tic2
        file_name = "result_report/{}/{}/test1_{}_{}epochs_{}ratio_{}pca_{}patch_classification_report.txt".format(
            args.mode,
            args.dataset,
            args.dataset,
            args.Epoch,
            ratio,
            args.Pca_Components,
            args.Patch_size,
        )

        with open(file_name, 'w') as x_file:
            x_file.write('\n')
            x_file.write('{} Test_time (s)'.format(Test_time))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(kappa))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(oa))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(aa))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(each_acc))
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))
        print('------Get classification results successful-------')
        print(f'Overall Accuracy: {oa:.2f}')
        print(f'Average accuracy: {aa:.2f}')
        print(f'Kappa Coefficient: {kappa:.2f}')
        path = 'result_map/{}/{}/test1_{}_{}epochs_{}ratio_{}pca_{}patch_map'.format(args.mode, args.dataset,
                                                                                     args.dataset,
                                                                                     args.Epoch,
                                                                                     ratio,
                                                                                     args.Pca_Components,
                                                                                     args.Patch_size)
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/' + args.dataset
        Utils.generate_png(
            total_iter, net, y, device, total_indices, path)
