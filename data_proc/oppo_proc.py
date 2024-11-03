# utils/oppo_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

WINDOW_SIZE = 90  # 默认窗口大小
OVERLAP_RATE = 0.5  # 默认重叠率
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

        print(f'Processing File: {file}', end=' ')
        is_validation = file in VALIDATION_FILES
        if is_validation:
            print('   ----   Validation Data')
        else:
            print()

        file_path = os.path.join(data_dir, file)
        select_col = list(range(1, 46)) + list(range(50, 59)) + list(range(63, 72)) + list(range(76, 85)) + list(range(89, 98)) + list(range(102, 134)) + [249]

        try:
            content = pd.read_csv(file_path, sep=' ', usecols=select_col, engine='python')
        except Exception as e:
            print(f'Error reading {file}: {e}')
            continue

        x = content.iloc[:, :-1].interpolate(method='linear', limit_direction='both', axis=0).to_numpy()
        y = content.iloc[:, -1].to_numpy()

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

def generate_client_ratios(num_clients):
    if num_clients is None or num_clients <= 1:
        return [1.0]
    return np.random.dirichlet(np.ones(num_clients) * 0.3)

def create_dataloaders_oppo(data_dir, batch_size=32, validation_split=0.2, normalize=False, client_id=None, num_clients=None):
    '''
    创建 OPPORTUNITY 数据集的 DataLoader
    '''
    # 检查联邦学习环境并引入异质性
    if client_id is not None and num_clients is not None:
        window_sizes = [80, 90, 100]
        overlap_rates = [0.5, 0.25, 0.75]
        chosen_window_size = np.random.choice(window_sizes)
        chosen_overlap_rate = np.random.choice(overlap_rates)
        ratios = generate_client_ratios(num_clients)
    else:
        chosen_window_size = WINDOW_SIZE
        chosen_overlap_rate = OVERLAP_RATE
        ratios = [1.0]

    # 加载数据
    all_files_data = load_oppo_data(data_dir)

    xtrain, xtest, ytrain, ytest = [], [], [], []

    # 标签转换（17 分类不含 null 类）
    label_seq = {
        406516: 0, 406517: 1, 404516: 2, 404517: 3, 406520: 4,
        404520: 5, 406505: 6, 404505: 7, 406519: 8, 404519: 9,
        406511: 10, 404511: 11, 406508: 12, 404508: 13, 408512: 14,
        407521: 15, 405506: 16
    }
    
    for file, data, label in all_files_data:
        windows = sliding_window(array=data, windowsize=chosen_window_size, overlaprate=chosen_overlap_rate, padding_size=3)
        labels = sliding_window(array=label.reshape(-1, 1), windowsize=chosen_window_size, overlaprate=chosen_overlap_rate)
        labels = labels[:, 0]  # 取每个窗口的第一个标签

        mapped_labels = [label_seq[lbl[0]] for lbl in labels if lbl[0] in label_seq]

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
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32).permute(0, 2, 1)
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32).permute(0, 2, 1)
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)

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

    return train_loader, val_loader, test_loader, chosen_window_size, 113, len(label_seq)
