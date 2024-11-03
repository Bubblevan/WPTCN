# utils/dasa_proc.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def sliding_window(array, windowsize, overlaprate):
    """
    滑动窗口函数
    """
    step = int(windowsize * (1 - overlaprate))
    num_windows = (len(array) - windowsize) // step + 1
    windows = np.array([array[i*step : i*step + windowsize] for i in range(num_windows)])
    return windows

def z_score_standard(xtrain, xtest):
    """
    Z-score 标准化
    """
    mean = xtrain.mean(axis=(0, 1), keepdims=True)
    std = xtrain.std(axis=(0, 1), keepdims=True) + 1e-8
    xtrain = (xtrain - mean) / std
    xtest = (xtest - mean) / std
    return xtrain, xtest

def load_dasa_data(dataset_dir, window_size=125, overlap_rate=0.4, validation_subjects=None, z_score=True):
    """
    加载并预处理 DASA 数据集
    """
    if validation_subjects is None:
        validation_subjects = set()
        use_average_split = True
    else:
        use_average_split = False

    xtrain, xtest, ytrain, ytest = [], [], [], []
    activities = sorted(os.listdir(dataset_dir))

    for label_id, activity in enumerate(activities):
        activity_path = os.path.join(dataset_dir, activity)
        if not os.path.isdir(activity_path):
            continue
        print(f'处理活动: {activity} (标签: {label_id})')
        
        participants = sorted(os.listdir(activity_path))
        for participant in participants:
            print(f'  处理参与者: {participant}', end='')

            # 根据验证集划分
            if not use_average_split and participant in validation_subjects:
                print(' -> 验证集')
                target_data, target_labels = xtest, ytest
            else:
                print(' -> 训练集')
                target_data, target_labels = xtrain, ytrain
            
            participant_path = os.path.join(activity_path, participant)
            if not os.path.isdir(participant_path):
                continue
            
            files = sorted(os.listdir(participant_path))
            data_list = []
            for file in files:
                if not file.endswith('.txt'):
                    continue
                file_path = os.path.join(participant_path, file)
                data = pd.read_csv(file_path, sep=',', header=None).to_numpy()
                data_list.append(data)
            if not data_list:
                continue
            concat_data = np.vstack(data_list)
            
            # 滑动窗口
            windows = sliding_window(array=concat_data, windowsize=window_size, overlaprate=overlap_rate)
            labels = np.array([label_id] * len(windows))

            if use_average_split:
                trainlen = int(len(windows) * 0.8)
                xtrain.append(windows[:trainlen])
                xtest.append(windows[trainlen:])
                ytrain.append(labels[:trainlen])
                ytest.append(labels[trainlen:])
            else:
                target_data.append(windows)
                target_labels.append(labels)
    
    if xtrain:
        xtrain = np.concatenate(xtrain, axis=0).astype(np.float32)
        ytrain = np.concatenate(ytrain, axis=0).astype(np.int64)
    else:
        xtrain = np.empty((0, window_size, 45), dtype=np.float32)
        ytrain = np.empty((0,), dtype=np.int64)
        
    if xtest:
        xtest = np.concatenate(xtest, axis=0).astype(np.float32)
        ytest = np.concatenate(ytest, axis=0).astype(np.int64)
    else:
        xtest = np.empty((0, window_size, 45), dtype=np.float32)
        ytest = np.empty((0,), dtype=np.int64)
    
    print('\n数据加载完成。')
    print(f'xtrain shape: {xtrain.shape}')
    print(f'xtest shape: {xtest.shape}')
    print(f'ytrain shape: {ytrain.shape}')
    print(f'ytest shape: {ytest.shape}')
    
    if z_score:
        print('进行 Z-score 标准化...')
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    return xtrain, xtest, ytrain, ytest

def generate_client_ratios(num_clients):
    if num_clients is None or num_clients <= 1:
        return [1.0]
    ratios = np.random.dirichlet(np.ones(num_clients) * 0.3)
    return ratios

def create_dataloaders_dasa(data_dir, batch_size=32, validation_split=0.2, window_size=125, overlap_rate=0.4, validation_subjects=None, z_score=True, client_id=None, num_clients=None):
    '''
        data_dir: 源数据目录
        batch_size: 批大小
        validation_split: 验证集比例（仅在不使用留一法时有效）
        window_size: 滑窗大小
        overlap_rate: 滑窗重叠率
        validation_subjects: 留一法中用于验证的受试者编号
        z_score: 是否进行标准化
        client_id: 客户端ID，用于数据划分
        num_clients: 客户端总数，用于分配数据比例（仅在联邦学习中使用）
    '''
    if client_id is not None and num_clients is not None:
        window_sizes = [100, 125, 150]
        overlap_rates = [0.4, 0.3, 0.5]
        chosen_window_size = np.random.choice(window_sizes)
        chosen_overlap_rate = np.random.choice(overlap_rates)
        ratios = generate_client_ratios(num_clients)
    else:
        chosen_window_size = window_size
        chosen_overlap_rate = overlap_rate
        ratios = [1.0]

    xtrain, xtest, ytrain, ytest = load_dasa_data(
        dataset_dir=data_dir,
        window_size=chosen_window_size,
        overlap_rate=chosen_overlap_rate,
        validation_subjects=validation_subjects,
        z_score=z_score
    )
    
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32).permute(0, 2, 1)
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32).permute(0, 2, 1)
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)
    
    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)
    
    if not validation_subjects:
        num_train = int((1 - validation_split) * len(train_dataset))
        num_val = len(train_dataset) - num_train
        train_subset, val_subset = random_split(train_dataset, [num_train, num_val])
    else:
        train_subset = train_dataset
        val_subset = test_dataset
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, chosen_window_size, 45, len(np.unique(ytrain))
