# utils/pamap2_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_pamap2_data(data_dir):
    # 加载所有数据文件
    filelist = os.listdir(data_dir)
    print('Loading PAMAP2 subject data...')
    all_subjects_data = []

    for file in filelist:
        if not file.endswith('.dat'):
            continue  # 只处理 .dat 文件
        subject_id = int(file.split('.')[0][7:10])  # 假设文件名中包含subject ID，例如: subject101.dat
        print(f'Processing Subject {subject_id}...')
        
        file_path = os.path.join(data_dir, file)
        content = pd.read_csv(file_path, sep=' ', usecols=[1]+list(range(4,16))+list(range(21,33))+list(range(38,50)), engine='python')
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()  # 线性插值填充NaN
        
        # 降采样1/3，100Hz -> 33.3Hz
        data = content[::3, 1:]  # 数据 （n, 36)
        label = content[::3, 0]   # 标签

        # 去除标签为0的样本
        data = data[label != 0]
        label = label[label != 0]

        all_subjects_data.append((subject_id, data, label))
    
    return all_subjects_data

def sliding_window(array, windowsize, overlaprate):
    step = int(windowsize * (1 - overlaprate))
    num_windows = (len(array) - windowsize) // step + 1
    windows = np.array([array[i*step : i*step + windowsize] for i in range(num_windows)])
    return windows

def normalize_signals(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
    if std is None:
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def generate_client_ratios(num_clients):
    if num_clients is None or num_clients <= 1:
        return [1.0]  # 没有联邦环境时，返回整个数据集的比例为1
    ratios = np.random.dirichlet(np.ones(num_clients) * 0.3)  # 为联邦环境生成比例
    return ratios

def create_dataloaders_pamap2(data_dir, batch_size=32, validation_split=0.2, normalize=False, client_id=None, num_clients=None):
    '''
        data_dir: 源数据目录，包含所有 .dat 文件
        batch_size: 批大小
        validation_split: 验证集比例
        normalize: 是否进行标准化
        client_id: 客户端ID，用于数据划分
        num_clients: 客户端总数，用于分配数据比例（仅在联邦学习中使用）
    '''
    # 加载所有数据
    all_subjects_data = load_pamap2_data(data_dir)
    
    # 如果在联邦场景中，随机选择滑动窗口大小和重叠率
    if client_id is not None and num_clients is not None:
        window_sizes = [128, 171, 256]
        overlap_rates = [0.5, 0.25]
        chosen_window_size = np.random.choice(window_sizes)
        chosen_overlap_rate = np.random.choice(overlap_rates)
        ratios = generate_client_ratios(num_clients)
    else:
        # 非联邦学习场景下使用固定的滑动窗口和重叠率
        chosen_window_size = 171
        chosen_overlap_rate = 0.5
        ratios = [1.0]

    xtrain, ytrain = [], []
    xtest, ytest = [], []

    # 设定验证集subject ID
    VALIDATION_SUBJECTS = {105}  # 可根据需要调整

    for subject_id, data, label in all_subjects_data:
        print(f'Processing Subject {subject_id}...')
        
        # 滑动窗口处理
        windows = sliding_window(array=data, windowsize=chosen_window_size, overlaprate=chosen_overlap_rate)
        labels = sliding_window(array=label.reshape(-1, 1), windowsize=chosen_window_size, overlaprate=chosen_overlap_rate)
        
        labels = labels[:, 0].squeeze()  # 确保标签是1D

        if subject_id in VALIDATION_SUBJECTS:
            xtest.extend(windows)
            ytest.extend(labels)
            print(f' --> Subject {subject_id} added to Test set')
        else:
            xtrain.extend(windows)
            ytrain.extend(labels)
            print(f' --> Subject {subject_id} added to Train set')

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, dtype=np.int64).squeeze()
    ytest = np.array(ytest, dtype=np.int64).squeeze()

    # 数据标准化
    if normalize:
        xtrain, mean, std = normalize_signals(xtrain)
        xtest, _, _ = normalize_signals(xtest, mean, std)

    # 转换为 PyTorch 张量，并在这里调整维度
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32).permute(0, 2, 1)
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32).permute(0, 2, 1)
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)

    # 划分训练集和验证集
    num_train = int((1 - validation_split) * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

    # 创建 DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 返回DataLoader和当前客户端选定的窗口大小
    return train_loader, val_loader, test_loader, chosen_window_size, 36, len(np.unique(ytrain))  # input_length=chosen_window_size, num_input_channels=36, num_classes
