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

    参数:
        dataset_dir (str): 数据集目录
        window_size (int): 滑窗大小
        overlap_rate (float): 滑窗重叠率
        validation_subjects (set or None): 验证集的受试者编号（如 {'p7', 'p8'}）
        z_score (bool): 是否进行标准化

    返回:
        xtrain, xtest, ytrain, ytest: 训练集和测试集的数据及标签
    """
    
    # 如果 validation_subjects 是 None，则将其初始化为空集合
    if validation_subjects is None:
        validation_subjects = set()  # 初始化为空集，以防后续判断出错
        use_average_split = True  # 使用平均法
    else:
        use_average_split = False  # 使用留一法

    xtrain, xtest, ytrain, ytest = [], [], [], []
    activities = sorted(os.listdir(dataset_dir))  # 活动目录，如 'a01', 'a02', ...
    
    for label_id, activity in enumerate(activities):
        activity_path = os.path.join(dataset_dir, activity)
        if not os.path.isdir(activity_path):
            continue  # 跳过非目录文件
        print(f'处理活动: {activity} (标签: {label_id})')
        
        participants = sorted(os.listdir(activity_path))  # 参与者文件夹，如 'p1', 'p2', ...
        for participant in participants:
            print(f'  处理参与者: {participant}', end='')

            # 如果使用留一法，根据 validation_subjects 判断是否为验证集
            if not use_average_split and participant in validation_subjects:
                print(' -> 验证集')
                target_data, target_labels = xtest, ytest
            else:
                print(' -> 训练集')
                target_data, target_labels = xtrain, ytrain
            
            participant_path = os.path.join(activity_path, participant)
            if not os.path.isdir(participant_path):
                print(f"    跳过非目录文件: {participant_path}")
                continue
            
            # 读取并拼接所有txt文件
            files = sorted(os.listdir(participant_path))
            data_list = []
            for file in files:
                if not file.endswith('.txt'):
                    continue
                file_path = os.path.join(participant_path, file)
                data = pd.read_csv(file_path, sep=',', header=None).to_numpy()  # [125, 45]
                data_list.append(data)
            if not data_list:
                print(f"    跳过没有有效数据的参与者: {participant}")
                continue
            concat_data = np.vstack(data_list)  # [125 * num_files, 45]
            
            # 滑动窗口
            windows = sliding_window(array=concat_data, windowsize=window_size, overlaprate=overlap_rate)  # [n, window_size, 45]
            labels = np.array([label_id] * len(windows))

            # 平均法时，将数据按比例分割到训练集和验证集
            if use_average_split:
                trainlen = int(len(windows) * 0.8)  # 80% 的数据作为训练集
                testlen = len(windows) - trainlen   # 20% 作为验证集
                xtrain.append(windows[:trainlen])
                xtest.append(windows[trainlen:])
                ytrain.append(labels[:trainlen])
                ytest.append(labels[trainlen:])
            else:
                # 留一法时，根据参与者划分训练集和验证集
                target_data.append(windows)
                target_labels.append(labels)
    
    # 转换为 NumPy 数组
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
    
    # 标准化
    if z_score:
        print('进行 Z-score 标准化...')
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    return xtrain, xtest, ytrain, ytest

def create_dataloaders_dasa(data_dir, batch_size=32, validation_split=0.2, window_size=125, overlap_rate=0.4, validation_subjects=None, z_score=True):
    '''
        data_dir: 源数据目录，包含所有 DASA 数据
        batch_size: 批大小
        validation_split: 验证集比例（仅在不使用留一法时有效）
        window_size: 滑窗大小
        overlap_rate: 滑窗重叠率
        validation_subjects: 留一法中用于验证的受试者编号
        z_score: 是否进行标准化
    '''
    # 加载数据
    xtrain, xtest, ytrain, ytest = load_dasa_data(
        dataset_dir=data_dir,
        window_size=window_size,
        overlap_rate=overlap_rate,
        validation_subjects=validation_subjects,
        z_score=z_score
    )
    
    # 创建 Tensor
    train_segments_tensor = torch.tensor(xtrain, dtype=torch.float32).permute(0, 2, 1)  # [batch_size, channels, sequence_length]
    train_labels_tensor = torch.tensor(ytrain, dtype=torch.long)
    test_segments_tensor = torch.tensor(xtest, dtype=torch.float32).permute(0, 2, 1)    # [batch_size, channels, sequence_length]
    test_labels_tensor = torch.tensor(ytest, dtype=torch.long)
    
    # 创建 TensorDataset
    train_dataset = TensorDataset(train_segments_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_segments_tensor, test_labels_tensor)
    
    # 从训练集中划分出验证集
    if not validation_subjects:
        num_train = int((1 - validation_split) * len(train_dataset))
        num_val = len(train_dataset) - num_train
        print(f'训练集样本数: {num_train}, 验证集样本数: {num_val}')
        train_subset, val_subset = random_split(train_dataset, [num_train, num_val])
        print("平均法划分训练集和验证集")
    else:
        # 如果使用留一法，测试集已经包含验证集
        train_subset = train_dataset
        val_subset = test_dataset
    
    # 创建 DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
