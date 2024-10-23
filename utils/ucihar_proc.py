import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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

    return np.transpose(signals, (1, 0, 2))

def normalize_signals(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    return (X - mean) / std
# 这部分使用上准确率会更高

def create_dataloaders_ucihar(data_dir, batch_size, normalize=False, validation_split=0.2):
    X_train = load_inertial_signals(data_dir, 'train')
    X_test = load_inertial_signals(data_dir, 'test')

    if normalize:
        X_train = normalize_signals(X_train)
        X_test = normalize_signals(X_test)

    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')) - 1
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.txt')) - 1

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    num_train = int((1 - validation_split) * len(X_train_tensor))
    train_dataset = TensorDataset(X_train_tensor[:num_train], y_train_tensor[:num_train])
    val_dataset = TensorDataset(X_train_tensor[num_train:], y_train_tensor[num_train:])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
