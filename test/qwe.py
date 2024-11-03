# 实验 FreTS 模型

import os
import sys
script_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
print(root_dir)
sys.path.append(root_dir)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------- 模型定义 -------------------- #

from models.frets import BackboneFreTS

# -------------------- 数据加载和预处理 -------------------- #

def load_inertial_signals(data_dir, dataset='train'):
    signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                    'total_acc_x', 'total_acc_y', 'total_acc_z']

    signals = []
    for signal_type in signal_types:
        filename = os.path.join(data_dir, dataset, 'Inertial Signals', f"{signal_type}_{dataset}.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件未找到: {filename}")
        signal_data = np.loadtxt(filename)
        if signal_data.shape[1] != 128:
            raise ValueError(f"每个样本应有128个时间步长，但在文件 {filename} 中找到 {signal_data.shape[1]} 个。")
        signals.append(signal_data)

    return np.transpose(np.array(signals), (1, 0, 2))  # Shape (n_samples, n_channels, n_timesteps)

def normalize_signals(X):
    # 按通道标准化
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    return (X - mean) / std

def create_dataloaders(data_dir, batch_size=32, normalize=True, validation_split=0.2):
    X_train = load_inertial_signals(data_dir, 'train')
    X_test = load_inertial_signals(data_dir, 'test')

    if normalize:
        X_train = normalize_signals(X_train)
        X_test = normalize_signals(X_test)

    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')) - 1
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.txt')) - 1

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建训练和验证集
    num_train = int((1 - validation_split) * len(X_train_tensor))
    train_dataset = TensorDataset(X_train_tensor[:num_train], y_train_tensor[:num_train])
    val_dataset = TensorDataset(X_train_tensor[num_train:], y_train_tensor[num_train:])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# -------------------- 训练和评估函数 -------------------- #

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)
    best_val_accuracy = 0.0

    # 初始化用于存储指标的列表
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_f1_micro_list = []
    val_f1_macro_list = []
    val_conf_matrices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 调整输入形状
            inputs = inputs.permute(0, 2, 1)  # (batch_size, n_timesteps, n_channels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # 调用修改后的 evaluate_model 函数
        val_accuracy, val_preds, val_labels = evaluate_model(model, val_loader, device)

        # 计算F1得分和混淆矩阵
        val_f1_micro = f1_score(val_labels, val_preds, average='micro')
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_conf_matrix = confusion_matrix(val_labels, val_preds)

        # 存储指标
        train_loss_list.append(epoch_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        val_f1_micro_list.append(val_f1_micro)
        val_f1_macro_list.append(val_f1_macro)
        val_conf_matrices.append(val_conf_matrix)

        # 打印指标
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Val F1 Micro: {val_f1_micro:.4f}, Val F1 Macro: {val_f1_macro:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_accuracy))

    # 将指标作为函数的返回值返回，以便在主函数中使用
    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'val_f1_micro': val_f1_micro_list,
        'val_f1_macro': val_f1_macro_list,
        'val_conf_matrices': val_conf_matrices
    }

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 调整输入形状
            inputs = inputs.permute(0, 2, 1)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集所有的预测值和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels



def main():
    # 配置参数
    data_dir = '../../data/UCI-HAR-Dataset'  # 数据集路径
    batch_size = 64
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

    n_classes = 6  # UCI-HAR数据集的活动类别数
    model = BackboneFreTS(
        n_steps=128,
        n_features=9,
        embed_size=9,
        n_pred_steps=128,  # 这个参数不再使用，可以忽略或删除
        hidden_size=64,
        n_classes=n_classes,
        channel_independence=False,
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并获取指标
    num_epochs = 25
    metrics = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

    # 绘制准确率和F1得分随epoch的变化
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_acc'], label='Training Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    plt.plot(epochs, metrics['val_f1_micro'], label='Validation F1 Micro')
    plt.plot(epochs, metrics['val_f1_macro'], label='Validation F1 Macro')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.show()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 在测试集上评估模型
    test_accuracy, test_preds, test_labels = evaluate_model(model, test_loader, device)
    test_f1_micro = f1_score(test_labels, test_preds, average='micro')
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_conf_matrix = confusion_matrix(test_labels, test_preds)

    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Micro: {test_f1_micro:.4f}, Test F1 Macro: {test_f1_macro:.4f}")

    # 绘制测试集的混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Test Set')
    plt.show()

if __name__ == "__main__":
    main()