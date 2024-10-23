# baseline train
import os
import sys
script_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
print(root_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
import torch.optim as optim
# 导入你自己的模型，如 CNN, ResNet 等
from models.cnn import CNN
from models.resnet import ResNet
from models.lstm import LSTM
from models.channel_attention import ChannelAttentionNeuralNetwork
from utils.util import evaluate, train_one_epoch, load_dataset
from utils.visualization import plot_results

def main():
    # 配置超参数
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    validation_split = 0.2
    dataset_name = 'OPPORTUNITY'  # 根据需要选择数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = load_dataset(dataset_name, batch_size, validation_split)
    print(num_classes, num_input_channels, input_length)
    # 定义模型列表，传入 input_length 参数
    model_list = {
        # 'CNN': CNN(num_input_channels, num_classes, input_length),
        'ResNet': ResNet(num_input_channels, num_classes, input_length),
        # 'LSTM': LSTM(num_input_channels, num_classes, input_length),
        'CANet': ChannelAttentionNeuralNetwork(num_input_channels, num_classes, input_length)
    }

    # 用于保存每个epoch的准确率、F1分数和混淆矩阵
    train_accuracies = []
    val_accuracies = []
    f1_macros = []
    f1_micros = []
    conf_matrices = []

    # 逐个模型进行训练和评估
    for model_name, model in model_list.items():
        print(f"\n===== Training and evaluating {model_name} =====\n")
        
        # 将模型放到设备上
        model = model.to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_val_acc = 0.0
        best_model_path = f'best_{model_name}.pth'

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs} for {model_name}")
            
            # 遍历 train_loader 并对 inputs 进行 unsqueeze 处理
            # train_loader_with_unsqueeze = [(inputs, labels) for inputs, labels in train_loader]
            train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # 同样对 val_loader 进行 unsqueeze 处理
            # val_loader_with_unsqueeze = [(inputs, labels) for inputs, labels in val_loader]
            val_loss, val_acc, conf_matrix, f1_macro, f1_micro = evaluate(model, device, val_loader, criterion, mode='Validation')
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # 保存准确率、F1分数和混淆矩阵
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            f1_macros.append(f1_macro)
            f1_micros.append(f1_micro)
            conf_matrices.append(conf_matrix)
            # 更新学习率
            scheduler.step()

        # 测试最佳模型
        # test_loader_with_unsqueeze = [(inputs, labels) for inputs, labels in test_loader]
        # model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, _, _, _ = evaluate(model, device, test_loader, criterion, mode='Test')
        print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # 绘制混淆矩阵、F1分数、准确率的图像
        plot_results(train_accuracies, val_accuracies, f1_macros, f1_micros, conf_matrices)


if __name__ == "__main__":
    main()
