# utils/util.py

import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def load_dataset(dataset, batch_size, validation_split, client_id=None, num_clients=None):
    """
    加载数据集。
    
    参数:
    - dataset (str): 数据集名称。
    - batch_size (int): 批大小。
    - validation_split (float): 验证集比例。
    - client_id (str, optional): 客户端ID，例如 "client_1"。默认为None。
    - num_clients (int, optional): 客户端总数。默认为None。
    
    返回:
    - train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes
    """
    if dataset == 'UCI-HAR':
        from data_proc.ucihar_proc import create_dataloaders_ucihar
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = create_dataloaders_ucihar(
            data_dir='../data/UCI-HAR-Dataset',
            batch_size=batch_size,
            normalize=True,
            validation_split=validation_split,
            client_id=client_id,
            num_clients=num_clients
        )
    elif dataset == 'WISDM':
        from data_proc.wisdm_proc import create_dataloaders_wisdm
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = create_dataloaders_wisdm(
            data_dir='../data/WISDM_ar_v1.1',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True,
            client_id=client_id,
            num_clients=num_clients
        )
        input_length = 128
        num_input_channels = 3
        num_classes = 6
    elif dataset == 'PAMAP2':
        from data_proc.pamap_proc import create_dataloaders_pamap2
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = create_dataloaders_pamap2(
            data_dir='../data/PAMAP2_Dataset/Protocol',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True,
            client_id=client_id,
            num_clients=num_clients
        )
        input_length = 171
        num_input_channels = 36
        num_classes = 12
    elif dataset == 'OPPORTUNITY':
        from data_proc.oppo_proc import create_dataloaders_oppo
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = create_dataloaders_oppo(
            data_dir='../data/OpportunityUCIDataset/dataset',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True,
            client_id=client_id,
            num_clients=num_clients
        )
        input_length = 90
        num_input_channels = 113
        num_classes = 17
    elif dataset == 'USC-HAD':
        from data_proc.uschad_proc import create_dataloaders_uschad
        train_loader, val_loader, test_loader, num_input_channels = create_dataloaders_uschad(
            data_dir='../data/USC-HAD',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True,
            client_id=client_id,
            num_clients=num_clients
        )
        input_length = 100
        num_classes = 12
    elif dataset == 'DASA':
        from data_proc.dasa_proc import create_dataloaders_dasa
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = create_dataloaders_dasa(
            data_dir='../data/Daily_and_Sports_Activities/data',
            batch_size=batch_size,
            validation_split=validation_split,
            window_size=125,
            overlap_rate=0.4,
            validation_subjects=None,
            z_score=True,
            client_id=client_id,
            num_clients=num_clients
        )
        input_length = 125
        num_input_channels = 45
        num_classes = 19
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))

    return epoch_loss, epoch_acc

def evaluate(model, device, dataloader, criterion, mode='Validation'):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating ({mode})", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有预测值和真实标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算损失和准确率
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算F1分数（宏平均）
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # 计算F1分数（微平均）
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    # 打印混淆矩阵和F1分数
    print(f"{mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    print(f"{mode} Confusion Matrix:\n{conf_matrix}")
    print(f"{mode} F1 Score (Macro): {f1_macro:.4f}")
    print(f"{mode} F1 Score (Micro): {f1_micro:.4f}")

    return epoch_loss, epoch_acc, conf_matrix, f1_macro, f1_micro
