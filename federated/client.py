# federated/client.py
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config=None):  # 接受 config 参数
        """返回模型参数"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """训练模型"""
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_loss, train_acc = train_one_epoch(
            self.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            self.train_loader, criterion, optimizer
        )

        return self.get_parameters(config=config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """评估模型"""
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

    # 加载数据
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

    # 创建并启动客户端
    client = FlowerClient(client_id, model, train_loader, val_loader)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()

# federated/server.py

import flwr as fl
import logging

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 配置联邦学习策略
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,            # 每轮选择50%的客户端参与训练
        fraction_evaluate=0.5,       # 每轮选择50%的客户端参与评估
        min_fit_clients=3,           # 每轮最少训练客户端数
        min_evaluate_clients=3,      # 每轮最少评估客户端数
        min_available_clients=3,     # 最少可用客户端数
    )

    # 创建 ServerConfig 对象
    config = fl.server.ServerConfig(num_rounds=10)

    # 启动服务器
    fl.server.start_server(
        server_address="[::]:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
        main()
