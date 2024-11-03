# utils/ucihar_proc.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_inertial_signals(data_dir, dataset='train'):
    signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                    'total_acc_x', 'total_acc_y', 'total_acc_z']

    signals = []
    for signal_type in signal_types:
        filename = os.path.join(data_dir, dataset, 'Inertial Signals', f"{signal_type}_{dataset}.txt")
        signals.append(np.loadtxt(filename))

    signals = np.array(signals)
    if signals.shape[2] != 128:
        raise ValueError(f"每个样本应有128个时间步长，但在文件中找到 {signals.shape[2]} 个。")

    return np.transpose(signals, (1, 0, 2))  # 形状: (num_samples, num_channels, num_timesteps)

def load_labels(data_dir, dataset='train'):
    filename = os.path.join(data_dir, dataset, f'y_{dataset}.txt')
    y = np.loadtxt(filename).astype(int) - 1  # 标签从0开始
    return y

def load_subjects(data_dir, dataset='train'):
    filename = os.path.join(data_dir, dataset, f'subject_{dataset}.txt')
    subjects = np.loadtxt(filename).astype(int)
    return subjects

def load_test_data_ucihar(data_dir):
    X_test = load_inertial_signals(data_dir, 'test')
    y_test = load_labels(data_dir, 'test')
    return X_test, y_test

def normalize_signals(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    return (X - mean) / std

def create_dataloaders_ucihar(data_dir, batch_size, normalize=False, validation_split=0.2, client_id=None, num_clients=2):
    # 加载训练数据和测试数据
    X_train = load_inertial_signals(data_dir, 'train')  # 形状: (num_samples, num_channels, 128)
    y_train = load_labels(data_dir, 'train')            # 形状: (num_samples,)
    subjects_train = load_subjects(data_dir, 'train')    # 形状: (num_samples,)

    X_test, y_test = load_test_data_ucihar(data_dir)     # 测试集不分配给客户端

    if normalize:
        X_train = normalize_signals(X_train)
        X_test = normalize_signals(X_test)

    # 如果提供了client_id，则根据client_id划分数据
    if client_id is not None:
        # 假设client_id格式为 "client_1", "client_2", etc.
        client_index = int(client_id.split('_')[-1]) - 1  # 0-based index

        # 获取所有唯一的subjects并排序
        unique_subjects = np.unique(subjects_train)
        unique_subjects = np.sort(unique_subjects)

        # 将subjects均匀分配给不同的客户端
        subjects_per_client = np.array_split(unique_subjects, num_clients)

        # 获取当前客户端负责的subjects
        if client_index >= num_clients:
            raise ValueError(f"client_id {client_id} 超出了总客户端数量 {num_clients}。")

        client_subjects = subjects_per_client[client_index]

        # 获取属于当前客户端的样本索引
        client_indices = np.isin(subjects_train, client_subjects)

        # 根据client_indices筛选训练数据和标签
        X_train = X_train[client_indices]
        y_train = y_train[client_indices]
        subjects_train = subjects_train[client_indices]

    # 划分训练集和验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
    )

    # 创建TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train_split, dtype=torch.float32),
                                  torch.tensor(y_train_split, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_split, dtype=torch.float32),
                                torch.tensor(y_val_split, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, 128, 9, 6  # input_length=128, num_input_channels=9, num_classes=6
