# utils/ucihad_proc.py

import os
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def sliding_window(array, windowsize, overlaprate):
    '''
    滑窗函数，返回窗口切分后的数组
    '''
    step = int(windowsize * (1 - overlaprate))
    if step <= 0:
        step = 1  # 避免步长为0
    num_windows = (len(array) - windowsize) // step + 1
    if num_windows <= 0:
        return np.empty((0, windowsize, array.shape[1]))
    windows = np.array([array[i*step : i*step + windowsize] for i in range(num_windows)])
    return windows

def normalize_signals(X, xtest, mean=None, std=None):
    '''
    标准化函数，应用 z-score 标准化
    '''
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
    if std is None:
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    xtest_norm = (xtest - mean) / std
    return X_norm, xtest_norm, mean, std

def generate_client_ratios(num_clients):
    if num_clients is None or num_clients <= 1:
        return [1.0]
    return np.random.dirichlet(np.ones(num_clients) * 0.3)

def create_dataloaders_uschad(data_dir, batch_size=32, validation_split=0.2, normalize=False, client_id=None, num_clients=None):
    '''
    创建 USC-HAD 数据集的 DataLoader，并根据第一个 .mat 文件动态确定通道数
    '''
    # 检查是否处于联邦学习环境中
    if client_id is not None and num_clients is not None:
        window_sizes = [80, 100, 120]
        overlap_rates = [0.1, 0.25, 0.5]
        chosen_window_size = np.random.choice(window_sizes)
        chosen_overlap_rate = np.random.choice(overlap_rates)
        ratios = generate_client_ratios(num_clients)
    else:
        chosen_window_size = 100
        chosen_overlap_rate = 0.1
        ratios = [1.0]

    # 定义验证集受试者
    VALIDATION_SUBJECTS = {1, 2, 3, 4}
    label_seq = {i: i - 1 for i in range(1, 13)}  # 12个活动
    xtrain, xtest, ytrain, ytest = [], [], [], []
    channels = None  # 用于存储动态确定的通道数

    subject_list = os.listdir(data_dir)
    print('Loading USC-HAD subject data...')
    
    for subject in subject_list:
        if not os.path.isdir(os.path.join(data_dir, subject)):
            continue

        try:
            subject_id = int(subject.lstrip('Subject'))  # 解析subject编号
        except ValueError:
            print(f"Skipping invalid folder: {subject}")
            continue

        print(f'Processing Subject {subject_id}', end=' ')
        if subject_id in VALIDATION_SUBJECTS:
            print('   ----   Validation Data')
        else:
            print()

        mat_list = os.listdir(os.path.join(data_dir, subject))
        
        for mat in mat_list:
            if not mat.endswith('.mat'):
                continue

            try:
                label_str = ''.join(filter(str.isdigit, mat.split('t')[0]))
                label_id = int(label_str)
                if label_id not in label_seq:
                    print(f" - Skipping unknown label in file: {mat}")
                    continue
                mapped_label = label_seq[label_id]
            except:
                print(f" - Skipping file with invalid label: {mat}")
                continue

            try:
                content = scio.loadmat(os.path.join(data_dir, subject, mat))['sensor_readings']
            except KeyError:
                print(f" - 'sensor_readings' not found in {mat}, skipping.")
                continue

            # 动态确定通道数
            if channels is None:
                channels = content.shape[1]

            # 滑窗切分
            windows = sliding_window(content, windowsize=chosen_window_size, overlaprate=chosen_overlap_rate)

            # 分配到训练集或验证集
            if subject_id in VALIDATION_SUBJECTS:
                xtest.extend(windows)
                ytest.extend([mapped_label] * len(windows))
            else:
                xtrain.extend(windows)
                ytrain.extend([mapped_label] * len(windows))

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, dtype=np.int64)
    ytest = np.array(ytest, dtype=np.int64)

    if normalize and len(xtrain) > 0 and len(xtest) > 0:
        xtrain, xtest, mean, std = normalize_signals(xtrain, xtest)
    else:
        mean, std = None, None

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print(f'xtrain shape: {xtrain.shape}')
    print(f'xtest shape: {xtest.shape}')
    print(f'ytrain shape: {ytrain.shape}')
    print(f'ytest shape: {ytest.shape}')

    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32) if len(xtrain) > 0 else torch.empty((0, channels, chosen_window_size), dtype=torch.float32)
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long) if len(ytrain) > 0 else torch.empty((0,), dtype=torch.long)

    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32) if len(xtest) > 0 else torch.empty((0, channels, chosen_window_size), dtype=torch.float32)
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long) if len(ytest) > 0 else torch.empty((0,), dtype=torch.long)

    if len(xtrain) > 0:
        train_segments_tensor = train_segments_tensor.permute(0, 2, 1)
    if len(xtest) > 0:
        test_segments_tensor = test_segments_tensor.permute(0, 2, 1)

    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)

    num_train = int((1 - validation_split) * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, channels
