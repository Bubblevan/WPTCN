# data_proc/wisdm_proc.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .base_processor import BaseProcessor

class WISDMProcessor(BaseProcessor):
    def __init__(self, data_dir, batch_size, window_size=128, step_size=64, train_ratio=0.8, validation_split=0.2, normalize=True, activity_mapping=None):
        """
        初始化WISDMProcessor。

        参数:
            data_dir (str): 数据集的根目录。
            batch_size (int): 批量大小。
            window_size (int): 分割窗口的大小。
            step_size (int): 窗口滑动的步长。
            train_ratio (float): 训练集占总数据的比例。
            validation_split (float): 训练集中验证集的比例。
            normalize (bool): 是否对数据进行标准化。
            activity_mapping (dict): 活动标签的映射字典。
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.validation_split = validation_split
        self.normalize = normalize
        self.activity_mapping = activity_mapping if activity_mapping else {
            'Walking': 0,
            'Jogging': 1,
            'Upstairs': 2,
            'Downstairs': 3,
            'Sitting': 4,
            'Standing': 5
        }

    def load_data(self):
        """
        加载原始WISDM数据。
        """
        data = self.load_wisdm_data()
        self.raw_data = data

    def preprocess(self):
        """
        预处理WISDM数据，包括清洗、分割、标准化等。
        """
        self.split_and_save_data(self.raw_data)
        
        # 加载训练和测试数据
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train', 'train_data.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, 'test', 'test_data.csv'))
        
        # 分割数据
        self.train_segments, self.train_labels = self.segment_data(train_data)
        self.test_segments, self.test_labels = self.segment_data(test_data)
        
        # 标准化
        if self.normalize:
            self.train_segments = self.normalize_signals(self.train_segments)
            self.test_segments = self.normalize_signals(self.test_segments)

    def create_dataloaders(self, batch_size=None):
        """
        创建训练、验证和测试的DataLoader。

        参数:
            batch_size (int, optional): 批量大小。如果未指定，使用初始化时的batch_size。

        返回:
            tuple: (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 转换为张量
        train_segments_tensor = torch.tensor(self.train_segments, dtype=torch.float32)
        train_labels_tensor = torch.tensor(self.train_labels, dtype=torch.long)
        test_segments_tensor = torch.tensor(self.test_segments, dtype=torch.float32)
        test_labels_tensor = torch.tensor(self.test_labels, dtype=torch.long)

        # 划分训练集和验证集
        num_train = int((1 - self.validation_split) * len(train_segments_tensor))
        train_dataset = TensorDataset(train_segments_tensor[:num_train], train_labels_tensor[:num_train])
        val_dataset = TensorDataset(train_segments_tensor[num_train:], train_labels_tensor[num_train:])
        test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)

        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_wisdm_data(self):
        """
        加载WISDM数据集的原始数据。

        返回:
            pd.DataFrame: 处理后的数据框。
        """
        filename = os.path.join(self.data_dir, 'WISDM_ar_v1.1_raw.txt')
        data = pd.read_csv(filename, header=None, names=['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel'], sep=',')
        
        # 清洗数据：移除z-accel中的分号，并转换为浮点数
        data['z-accel'] = data['z-accel'].astype(str).str.replace(';', '').astype(float)
        
        # 转换活动标签为数值
        data['activity'] = data['activity'].map(self.activity_mapping)
        
        # 移除包含NaN的行
        data = data.dropna().reset_index(drop=True)
        
        return data

    def split_and_save_data(self, data):
        """
        将数据集划分为训练集和测试集，并保存到相应的目录。

        参数:
            data (pd.DataFrame): 原始数据框。
        """
        # 打乱数据
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 分割训练和测试集
        train_size = int(self.train_ratio * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 保存分割后的数据
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(train_dir, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(test_dir, 'test_data.csv'), index=False)

    def segment_data(self, data):
        """
        将数据分割成固定大小的窗口。

        参数:
            data (pd.DataFrame): 数据框。

        返回:
            np.ndarray: 分割后的信号数据。
            np.ndarray: 对应的标签。
        """
        segments = []
        labels = []
        for activity in self.activity_mapping.values():
            activity_data = data[data['activity'] == activity]
            for i in range(0, len(activity_data) - self.window_size + 1, self.step_size):
                segment = activity_data.iloc[i:i + self.window_size][['x-accel', 'y-accel', 'z-accel']].values
                if len(segment) == self.window_size:
                    segments.append(segment)
                    labels.append(activity)
        return np.array(segments), np.array(labels)

    def normalize_signals(self, X):
        """
        对信号进行标准化。

        参数:
            X (np.ndarray): 原始信号数据。

        返回:
            np.ndarray: 标准化后的信号数据。
        """
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        return (X - mean) / std
