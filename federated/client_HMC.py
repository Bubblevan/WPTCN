# 本想着用HMC来优化服务端不确定度的
# 结果整的好像我模型训练了两遍一样

import os
import sys
script_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
print(root_dir)
sys.path.append(root_dir)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from models.WPTCN import WPTCN
from utils.util import load_dataset, train_one_epoch, evaluate
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, eta=0.01, K=10, T=5):
        # 初始化客户端，包括客户端ID、模型、数据加载器和FA-HMC参数
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta = eta      # 学习率
        self.K = K          # Leapfrog步数
        self.T = T          # 本地更新步数

    def sample_momentum(self):
        """从高斯分布中采样动量"""
        return {k: torch.randn_like(v) if v.dtype in [torch.float32, torch.float64] else torch.zeros_like(v) 
                for k, v in self.model.state_dict().items()}

    def compute_gradients(self):
        """计算当前模型的梯度"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.eta)  # 使用小学习率避免影响
        criterion = nn.CrossEntropyLoss()
        
        # 获取一个训练样本并计算损失
        for data, target in self.train_loader:
            data, target = data.to(next(self.model.parameters()).device), target.to(next(self.model.parameters()).device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            break  # 只计算一次梯度
        
        gradients = {name: param.grad for name, param in self.model.named_parameters()}
        return gradients

    def leapfrog_update(self, params, momentum, gradients):
        """执行Leapfrog积分器更新"""
        for k in params.keys():
            # 使用梯度更新动量
            if k in gradients:
                momentum[k] -= 0.5 * self.eta * gradients[k]
                params[k] += self.eta * momentum[k]
                # 重新计算梯度
                new_gradients = self.compute_gradients()
                momentum[k] -= 0.5 * self.eta * new_gradients[k]
        return params, momentum

    def get_parameters(self, config=None):
        """返回模型参数，格式为NumPy数组列表"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """从NumPy数组列表设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """使用HMC进行模型训练"""
        self.set_parameters(parameters)
        
        # 初始动量采样
        momentum = self.sample_momentum()
        gradients = self.compute_gradients()  # 初始梯度
        
        # Leapfrog积分更新
        for _ in range(self.K):
            params, momentum = self.leapfrog_update(self.model.state_dict(), momentum, gradients)

        # 优化器设置
        optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练T步
        for _ in range(self.T):
            train_loss, train_acc = train_one_epoch(
                self.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                self.train_loader, criterion, optimizer
            )
        
        return self.get_parameters(config=config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """使用验证数据评估模型"""
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, _, _, _ = evaluate(
            self.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            self.val_loader, criterion, mode='Validation'
        )
        return test_acc, len(self.val_loader.dataset), {"accuracy": test_acc}

def main():
    logging.basicConfig(level=logging.INFO)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Flower client')
    parser.add_argument('--cid', type=str, required=True, help='Client ID (e.g., client_1)')
    parser.add_argument('--num_clients', type=int, required=True, help='Total number of clients')
    args = parser.parse_args()

    client_id = args.cid
    num_clients = args.num_clients

    # 加载数据集
    dataset = 'UCI-HAR'  # 可根据实际情况更改
    batch_size = 32
    validation_split = 0.2
    train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = load_dataset(
        dataset, batch_size, validation_split, client_id=client_id, num_clients=num_clients
    )

    # 初始化模型
    model_parameters = {
        'num_input_channels': num_input_channels,
        'input_length': input_length,
        'num_classes': num_classes,
        'hidden_dim': 128,
        'kernel_size': 3,
        'num_levels': 2,
        'num_layers': 3,
        'wavelet_type': 'db1',
        'feedforward_ratio': 1,
        'group_type': 'channel',
        'normalization_eps': 1e-5,
        'normalization_affine': True
    }

    model = WPTCN(**model_parameters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建客户端实例并使用 .to_client() 启动
    client = FlowerClient(client_id, model, train_loader, val_loader, eta=0.01, K=10, T=5)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
