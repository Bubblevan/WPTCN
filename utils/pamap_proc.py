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

def create_dataloaders_pamap2(data_dir, batch_size=32, validation_split=0.2, normalize=False):
    '''
        data_dir: 源数据目录，包含所有 .dat 文件
        batch_size: 批大小
        validation_split: 验证集比例
        normalize: 是否进行标准化
    '''
    # 加载所有数据
    all_subjects_data = load_pamap2_data(data_dir)
    
    # 分割训练集和验证集
    xtrain, ytrain = [], []
    xtest, ytest = [], []
    VALIDATION_SUBJECTS = {105}  # 您可以根据需要调整

    for subject_id, data, label in all_subjects_data:
        print(f'Processing Subject {subject_id}...')
        # 滑窗
        windows = sliding_window(array=data, windowsize=171, overlaprate=0.5)
        labels = sliding_window(array=label.reshape(-1, 1), windowsize=171, overlaprate=0.5)
        
        # 使用 .squeeze() 来确保标签是1D
        labels = labels[:, 0].squeeze()  # 保证每个窗口的标签是一维的

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
    ytrain = np.array(ytrain, dtype=np.int64).squeeze()  # 确保标签为一维
    ytest = np.array(ytest, dtype=np.int64).squeeze()    # 确保标签为一维

    # 标准化
    if normalize:
        # 对训练数据进行标准化，并获取均值和标准差
        xtrain, mean, std = normalize_signals(xtrain)
        # 使用训练集的均值和标准差对测试数据进行标准化
        xtest, _, _ = normalize_signals(xtest, mean, std)

    # 转换为 PyTorch 张量，并在这里调整维度
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32).permute(0, 2, 1)  # [batch_size, channels, sequence_length]
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32).permute(0, 2, 1)  # [batch_size, channels, sequence_length]
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)

    # 确保数据的形状为 [samples, channels, sequence_length]

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
