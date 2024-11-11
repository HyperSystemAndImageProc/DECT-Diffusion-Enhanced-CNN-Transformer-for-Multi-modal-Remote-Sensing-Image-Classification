import scipy.io as sio


def loadData(dataset):
    global data_Diff, data_HSI, data_lidar, data_L, labels
    if dataset == 'Trento':
        data_HSI = sio.loadmat('data/Trento/HSI_Trento.mat')['hsi_trento']
        data_Diff = sio.loadmat('data/Trento/TrentoDiff/Trento7.mat')['Trento']
        data_lidar = sio.loadmat('data/Trento/Lidar1_Trento.mat')['lidar1_trento']
        labels = sio.loadmat('data/Trento/GT_Trento.mat')['gt_trento']
    elif dataset == 'Muufl':
        data_HSI = sio.loadmat('data/Muufl/Muufl_hsi.mat')['hsi']
        data_Diff = sio.loadmat('data/Muufl/MuuflDiff/Muufl9.mat')['Muufl']
        data_lidar = sio.loadmat('data/Muufl/Muufl_lidar.mat')['lidar']
        labels = sio.loadmat('data/Muufl/Muufl_gt0.mat')['gt']
    elif dataset == 'Augsburg':
        data_HSI = sio.loadmat('data/Augsburg/Au/Augsburg_hsi.mat')['hsi']
        data_Diff = sio.loadmat('data/Augsburg/AugsburgDiff/Aupad/Augsburgdiff9.mat')['hsi']
        data_lidar = sio.loadmat('data/Augsburg/Au/data_SAR_HR.mat')['data_SAR_HR']
        labels = sio.loadmat('data/Augsburg/Au/Augsburg_gt.mat')['gt']
    else:
        print("NO dataset")
    return data_HSI, data_Diff, data_lidar, labels
