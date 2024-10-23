# utils/oppo_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

WINDOW_SIZE = 90  # 窗口大小
OVERLAP_RATE = 0.5  # 重叠率
VALIDATION_FILES = {'S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat'}

def load_oppo_data(data_dir):
    '''
    加载 OPPORTUNITY 数据集的所有 .dat 文件，并返回列表
    每个元素为 (file_name, data, labels)
    '''
    filelist = os.listdir(data_dir)
    all_files_data = []

    print('Loading OPPORTUNITY subject data...')
    for file in filelist:
        if not file.endswith('.dat'):
            continue  # 只处理 .dat 文件

        print(f'Processing File: {file}', end='')
        is_validation = file in VALIDATION_FILES
        if is_validation:
            print('   ----   Validation Data')
        else:
            print()

        file_path = os.path.join(data_dir, file)
        # 根据提供的 OPPO 函数，选择特定列
        # 选择的列：[1 to 46], [50 to 58], [63 to 71], [76 to 84], [89 to 97], [102 to 133], [249]
        select_col = list(range(1, 46)) + list(range(50, 59)) + list(range(63, 72)) + list(range(76, 85)) + list(range(89, 98)) + list(range(102, 134)) + [249]

        # 读取数据
        try:
            content = pd.read_csv(file_path, sep=' ', usecols=select_col, engine='python')
        except Exception as e:
            print(f'Error reading {file}: {e}')
            continue

        # 线性插值填充 NaN
        x = content.iloc[:, :-1].interpolate(method='linear', limit_direction='both', axis=0).to_numpy()
        y = content.iloc[:, -1].to_numpy()

        # 去除标签为0的样本
        x = x[y != 0]
        y = y[y != 0]

        all_files_data.append((file, x, y))

    return all_files_data

def sliding_window(array, windowsize, overlaprate, padding_size=None):
    '''
    滑窗函数，返回窗口切分后的数组，并在需要时对时间维度进行填充
    '''
    step = int(windowsize * (1 - overlaprate))
    num_windows = (len(array) - windowsize) // step + 1
    windows = np.array([array[i*step : i*step + windowsize] for i in range(num_windows)])
    
    # 如果窗口的时间维度小于指定值，执行 padding
    if padding_size and windows.shape[1] < padding_size:
        padding_width = padding_size - windows.shape[1]
        windows = np.pad(windows, ((0, 0), (0, padding_width), (0, 0)), mode='constant')
    
    return windows


def normalize_signals(X, mean=None, std=None):
    '''
    标准化函数，应用 z-score 标准化
    '''
    if mean is None:
        mean = X.mean(axis=(0, 1), keepdims=True)
    if std is None:
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def create_dataloaders_oppo(data_dir, batch_size=32, validation_split=0.2, normalize=False):
    '''
    创建 OPPORTUNITY 数据集的 DataLoader
    '''
    # 定义验证集文件名
    VALIDATION_FILES = {'S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat'}

    # 加载数据
    all_files_data = load_oppo_data(data_dir)

    xtrain, xtest, ytrain, ytest = [], [], [], []

    # 标签转换（17 分类不含 null 类）
    label_seq = {
        406516: 0,  # Open Door 1
        406517: 1,  # Open Door 2
        404516: 2,  # Close Door 1
        404517: 3,  # Close Door 2
        406520: 4,  # Open Fridge
        404520: 5,  # Close Fridge
        406505: 6,  # Open Dishwasher
        404505: 7,  # Close Dishwasher
        406519: 8,  # Open Drawer 1
        404519: 9,  # Close Drawer 1
        406511: 10, # Open Drawer 2
        404511: 11, # Close Drawer 2
        406508: 12, # Open Drawer 3
        404508: 13, # Close Drawer 3
        408512: 14, # Clean Table
        407521: 15, # Drink from Cup
        405506: 16  # Toggle Switch
    }
    for file, data, label in all_files_data:
        # 滑窗
        # 调用时，传入 padding_size 参数，确保时间步长不小于 2
        windows = sliding_window(array=data, windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE, padding_size=3)
        labels = sliding_window(array=label.reshape(-1, 1), windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)
        labels = labels[:, 0]  # 假设每个窗口的标签取第一个

        # 映射标签
        mapped_labels = []
        for lbl in labels:
            lbl = lbl[0]  # 将 NumPy 数组转换为标量值
            if lbl in label_seq:
                mapped_labels.append(label_seq[lbl])
            else:
                # 如果标签不在 label_seq 中，则跳过
                continue

        # 过滤掉未映射的标签
        valid_indices = [i for i, lbl in enumerate(labels) if lbl[0] in label_seq]
        windows = windows[valid_indices]
        labels = np.array(mapped_labels)

        # 分配到训练集或测试集
        if file in VALIDATION_FILES:
            xtest.extend(windows)
            ytest.extend(labels)
            print(f' --> File {file} added to Test set')
        else:
            xtrain.extend(windows)
            ytrain.extend(labels)
            print(f' --> File {file} added to Train set')

    # 转换为 NumPy 数组
    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, dtype=np.int64)
    ytest = np.array(ytest, dtype=np.int64)

    # 标准化
    if normalize:
        xtrain, mean, std = normalize_signals(xtrain)
        xtest, _, _ = normalize_signals(xtest, mean=mean, std=std)

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print(f'xtrain shape: {xtrain.shape}')
    print(f'xtest shape: {xtest.shape}')
    print(f'ytrain shape: {ytrain.shape}')
    print(f'ytest shape: {ytest.shape}')

    # 转换为 PyTorch 张量
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32)
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32)
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)

    # 确保数据的形状为 [samples, channels, sequence_length]
    # OPPORTUNITY 数据集：windowsize=30, channels=113
    # 当前 windows 的形状为 [samples, 30, 113], 需要转置为 [samples, 113, 30]
    train_segments_tensor = train_segments_tensor.permute(0, 2, 1)
    test_segments_tensor = test_segments_tensor.permute(0, 2, 1)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)

    # 从训练集中划分出验证集
    num_train = int((1 - validation_split) * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

    # 创建 DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
