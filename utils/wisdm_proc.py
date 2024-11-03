# utils/wisdm_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_wisdm_data(data_dir):
    # Load the raw data file
    filename = os.path.join(data_dir, 'WISDM_ar_v1.1_raw.txt')
    data = pd.read_csv(filename, header=None, names=['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel'], sep=',')
    
    # Clean the data
    data['z-accel'] = data['z-accel'].astype(str).str.replace(';', '').astype(float)
    
    # Drop any rows with missing values
    data.dropna(inplace=True)
    
    # Convert activity labels to numerical values
    activity_mapping = {
        'Walking': 0,
        'Jogging': 1,
        'Upstairs': 2,
        'Downstairs': 3,
        'Sitting': 4,
        'Standing': 5
    }
    data['activity'] = data['activity'].map(activity_mapping)
    data = data[data['activity'].isin(activity_mapping.values())]  # 确保只包含有效标签
    
    return data

def split_and_save_data(data_dir, train_ratio=0.8):
    data = load_wisdm_data(data_dir)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data into train and test sets
    train_size = int(train_ratio * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Save the split data to train and test folders
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_data.to_csv(os.path.join(train_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(test_dir, 'test_data.csv'), index=False)

def segment_data(data, window_size=128, step_size=64):
    segments = []
    labels = []
    for activity in data['activity'].unique():
        activity_data = data[data['activity'] == activity]
        activity_data = activity_data.reset_index(drop=True)
        for i in range(0, len(activity_data) - window_size + 1, step_size):
            segment = activity_data.iloc[i:i + window_size][['x-accel', 'y-accel', 'z-accel']].values
            if len(segment) == window_size:
                segments.append(segment)
                labels.append(activity)
    return np.array(segments), np.array(labels)

def normalize_signals(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
    if std is None:
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def create_dataloaders_wisdm(data_dir, batch_size=32, validation_split=0.2, normalize=False):
    # Split and save data if not already split
    split_and_save_data(data_dir)
    
    # Load train and test data
    train_data = pd.read_csv(os.path.join(data_dir, 'train', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test', 'test_data.csv'))
    
    # Segment data
    train_segments, train_labels = segment_data(train_data)
    test_segments, test_labels = segment_data(test_data)
    
    # Normalize if required
    if normalize:
        train_segments, mean, std = normalize_signals(train_segments)
        test_segments, _, _ = normalize_signals(test_segments, mean, std)
    else:
        mean, std = None, None  # Not used, but for consistency
    
    # Convert to PyTorch tensors
    train_segments_tensor = torch.tensor(train_segments, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_segments_tensor = torch.tensor(test_segments, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # 确保数据的形状为 [samples, channels, sequence_length]
    # 目前 segment_data 返回的是 [samples, sequence_length, channels]
    # 需要转置为 [samples, channels, sequence_length]
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
