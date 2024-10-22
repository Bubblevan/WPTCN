import os
import sys
script_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
print(root_dir)
sys.path.append(root_dir)
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from models.WPTCN import WPTCN
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from utils.util import load_config
from utils.data_processing import create_dataloaders
from data_proc import get_processor
from utils.visualization import plot_results, calculate_latency, calculate_memory_usage, calculate_model_complexity



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


def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data_dir = '../../data/UCI-HAR-Dataset'  # 数据集路径
    batch_size = 32
    num_epochs = 20  # 根据需要调整
    learning_rate = 1e-3
    validation_split = 0.2  # 从训练集中划分20%作为验证集

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=True,
        validation_split=validation_split
    )

    # 获取数据维度信息
    input_length = 128  # 时间步长
    num_input_channels = 9  # 通道数量（传感器数量）
    num_classes = 6  # UCI-HAR 数据集的活动类别数

    # 初始化模型
    model_parameters = {
        'num_input_channels': num_input_channels,   # 输入通道数
        'input_length': input_length,               # 输入序列长度（当前未使用）
        'num_classes': num_classes,                 # 类别数
        'hidden_dim': 128,                          # 隐藏层维度
        'kernel_size': 3,                           # 卷积核大小
        'num_levels': 2,                            # 小波包分解层数
        'num_layers': 3,                            # 模型层数
        'wavelet_type': 'db1',                      # 小波类型
        'feedforward_ratio': 1,                     # 前馈网络扩展比例
        'group_type': 'channel',                    # 分组类型
        'normalization_eps': 1e-5,                  # 归一化的 epsilon
        'normalization_affine': True                # 是否使用仿射变换
    }
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 初始化模型
    model = WPTCN(**model_parameters)
    model.to(device)

    logging.info(f'Model:\n{model}')

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    best_model_path = 'best_model.pth'

    # 用于保存每个epoch的准确率、F1分数和混淆矩阵
    train_accuracies = []
    val_accuracies = []
    f1_macros = []
    f1_micros = []
    conf_matrices = []

    for epoch in range(1, num_epochs + 1):
        logging.info(f"\nEpoch {epoch}/{num_epochs}")

        # 训练
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, conf_matrix, f1_macro, f1_micro = evaluate(model, device, val_loader, criterion, mode='Validation')
        logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # 保存准确率、F1分数和混淆矩阵
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        f1_macros.append(f1_macro)
        f1_micros.append(f1_micro)
        conf_matrices.append(conf_matrix)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved to: {best_model_path}")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, _, _, _ = evaluate(model, device, test_loader, criterion, mode='Test')
    logging.info(f"\nTest - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # 绘制混淆矩阵、F1分数、准确率的图像
    plot_results(train_accuracies, val_accuracies, f1_macros, f1_micros, conf_matrices)

    # 计算Latency
    latency = calculate_latency(model, device, test_loader)

    # 计算Peak Memory Usage
    memory_usage = calculate_memory_usage(model, device, test_loader)

    # 计算模型参数量和FLOPs
    input_size = (1, model_parameters['num_input_channels'], model_parameters['input_length'])  # 1 表示 batch_size 为 1
    flops, params = calculate_model_complexity(model, input_size)

if __name__ == "__main__":
    main()