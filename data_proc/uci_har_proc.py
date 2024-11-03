# data_proc/uci_har_proc.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .base_processor import BaseProcessor

class UCIHARProcessor(BaseProcessor):
    def __init__(self, data_dir, batch_size, normalize=True, validation_split=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.validation_split = validation_split

    def load_inertial_signals(self, dataset='train'):
        signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                        'total_acc_x', 'total_acc_y', 'total_acc_z']

        signals = []
        for signal_type in signal_types:
            filename = os.path.join(self.data_dir, dataset, 'Inertial Signals', f"{signal_type}_{dataset}.txt")
            signals.append(np.loadtxt(filename))

        signals = np.array(signals)
        if signals.shape[2] != 128:
            raise ValueError(f"每个样本应有128个时间步长，但在文件中找到 {signals.shape[2]} 个。")

        return np.transpose(signals, (1, 0, 2))

    def normalize_signals(self, X):
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        return (X - mean) / std

    def load_data(self):
        self.X_train = self.load_inertial_signals('train')
        self.X_test = self.load_inertial_signals('test')
        self.y_train = np.loadtxt(os.path.join(self.data_dir, 'train', 'y_train.txt')) - 1
        self.y_test = np.loadtxt(os.path.join(self.data_dir, 'test', 'y_test.txt')) - 1

    def preprocess(self):
        if self.normalize:
            self.X_train = self.normalize_signals(self.X_train)
            self.X_test = self.normalize_signals(self.X_test)

    def create_dataloaders(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)

        num_train = int((1 - self.validation_split) * len(X_train_tensor))
        train_dataset = TensorDataset(X_train_tensor[:num_train], y_train_tensor[:num_train])
        val_dataset = TensorDataset(X_train_tensor[num_train:], y_train_tensor[num_train:])
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
