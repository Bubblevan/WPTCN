# utils/wisdm_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_wisdm_data(data_dir):
    filename = os.path.join(data_dir, 'WISDM_ar_v1.1_raw.txt')
    data = pd.read_csv(filename, header=None, names=['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel'], sep=',')

    # Clean the data
    data['z-accel'] = data['z-accel'].astype(str).str.replace(';', '').astype(float)
    data.dropna(inplace=True)

    # Convert activity labels to numerical values
    activity_mapping = {'Walking': 0, 'Jogging': 1, 'Upstairs': 2, 'Downstairs': 3, 'Sitting': 4, 'Standing': 5}
    data['activity'] = data['activity'].map(activity_mapping)
    data = data[data['activity'].isin(activity_mapping.values())]

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
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_data.to_csv(os.path.join(train_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(test_dir, 'test_data.csv'), index=False)

def segment_data(data, window_size=128, step_size=64):
    segments, labels = [], []
    for activity in data['activity'].unique():
        activity_data = data[data['activity'] == activity].reset_index(drop=True)
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

def generate_client_ratios(num_clients):
    # Generate random ratios greater than 0.1
    ratios = np.random.dirichlet(np.ones(num_clients) * 0.3)  # Increase 0.1 to ensure min ratio > 0.1
    return ratios

def create_dataloaders_wisdm(data_dir, batch_size=32, validation_split=0.2, normalize=False, client_id=None, num_clients=3):
    split_and_save_data(data_dir)

    train_data = pd.read_csv(os.path.join(data_dir, 'train', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test', 'test_data.csv'))

    if client_id is not None:
        client_index = int(client_id.split('_')[-1]) - 1
        unique_users = np.sort(train_data['user'].unique())

        # Adjust user data size ratios for heterogeneity
        ratios = generate_client_ratios(num_clients)
        user_counts = (ratios * len(unique_users)).astype(int)
        
        # Ensure at least one user per client
        user_counts = np.clip(user_counts, 1, None)
        
        users_per_client = np.split(unique_users[:sum(user_counts)], np.cumsum(user_counts[:-1]))

        if client_index >= num_clients:
            raise ValueError(f"client_id {client_id} 超出了总客户端数量 {num_clients}。")

        client_users = users_per_client[client_index]
        train_data = train_data[train_data['user'].isin(client_users)]
        
    # Adjust window size and step size for each client to introduce heterogeneity
    window_size = np.random.choice([64, 128, 256])
    step_size = window_size // 2

    # Segment the data
    train_segments, train_labels = segment_data(train_data, window_size=window_size, step_size=step_size)
    test_segments, test_labels = segment_data(test_data, window_size=window_size, step_size=step_size)

    if normalize:
        train_segments, mean, std = normalize_signals(train_segments)
        test_segments, _, _ = normalize_signals(test_segments, mean, std)

    train_segments_tensor = torch.tensor(train_segments, dtype=torch.float32).permute(0, 2, 1)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_segments_tensor = torch.tensor(test_segments, dtype=torch.float32).permute(0, 2, 1)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)
    num_train = int((1 - validation_split) * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, window_size, 3, 6  # input_length=window_size, num_input_channels=3, num_classes=6
