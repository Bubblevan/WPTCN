import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def load_dataset(dataset, batch_size, validation_split):
    if dataset == 'UCI-HAR':
        from utils.ucihar_proc import create_dataloaders_ucihar
        train_loader, val_loader, test_loader = create_dataloaders_ucihar(
            data_dir='../../data/UCI-HAR-Dataset',
            batch_size=batch_size,
            normalize=True,
            validation_split=validation_split
        )
        input_length = 128  # 时间步长
        num_input_channels = 9  # 通道数量（传感器数量）
        num_classes = 6  # UCI-HAR 数据集的活动类别数
    elif dataset == 'WISDM':
        from utils.wisdm_proc import create_dataloaders_wisdm
        train_loader, val_loader, test_loader = create_dataloaders_wisdm(
            data_dir='../../data/WISDM_ar_v1.1',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True          # 是否进行标准化
        )
        input_length = 128  # 时间步长
        num_input_channels = 3  # 通道数量（x, y, z）
        num_classes = 6  # WISDM 数据集的活动类别数（根据映射）
    elif dataset == 'PAMAP2':
        from utils.pamap_proc import create_dataloaders_pamap2
        train_loader, val_loader, test_loader = create_dataloaders_pamap2(
            data_dir='../../data/PAMAP2_Dataset/Protocol',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True          # 是否进行标准化
        )
        input_length = 171  # 时间步长（根据 PAMAP2 的窗口大小）
        num_input_channels = 36  # 通道数量（PAMAP2 的传感器数据维度）
        num_classes = 12  # PAMAP2 数据集的活动类别数
    elif dataset == 'OPPORTUNITY':
        from utils.oppo_proc import create_dataloaders_oppo
        train_loader, val_loader, test_loader = create_dataloaders_oppo(
            data_dir='../../data/OpportunityUCIDataset/dataset',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True          # 是否进行标准化
        )
        input_length = 90  # 时间步长（根据 OPPORTUNITY 的窗口大小 * 3倍，不然卷积把时间步卷到小于1了）
        num_input_channels = 113  # 通道数量（OPPORTUNITY 的传感器数据维度）
        num_classes = 17  # OPPORTUNITY 数据集的活动类别数
    elif dataset == 'USC-HAD':
        from utils.uschad_proc import create_dataloaders_uschad
        train_loader, val_loader, test_loader, num_input_channels = create_dataloaders_uschad(
            data_dir='../../data/USC-HAD',
            batch_size=batch_size,
            validation_split=validation_split,
            normalize=True          # 是否进行标准化
        )
        input_length = 100  # 时间步长（根据 USC-HAD 的窗口大小）
        num_classes = 12  # USC-HAD 数据集的活动类别数
    elif dataset == 'DASA':
        from utils.dasa_proc import create_dataloaders_dasa
        train_loader, val_loader, test_loader = create_dataloaders_dasa(
            data_dir='../../data/Daily_and_Sports_Activities/data',
            batch_size=batch_size,
            validation_split=validation_split,
            window_size=125,           # DASA 的滑窗大小
            overlap_rate=0.4,          # DASA 的滑窗重叠率
            validation_subjects=None,  # 留一法验证集，设为 None 使用平均法
            z_score=True               # 是否进行标准化
        )
        input_length = 125  # 时间步长
        num_input_channels = 45  # 通道数量（DASA 的传感器数据维度）
        num_classes = 19  # DASA 数据集的活动类别数
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
