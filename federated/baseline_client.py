import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
project_root = os.path.abspath(os.path.join(script_dir, '..'))  # 获取项目根目录
sys.path.insert(0, project_root)  # 将项目根目录添加到Python路径

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import time
from models.WPTCN import WPTCN
from models.partitioned_wptcn import PartitionedWPTCN
from utils.util import load_dataset, train_one_epoch, evaluate
from utils.resource_monitor import ResourceMonitor
import requests

class BaselineClient(fl.client.NumPyClient):
    def __init__(self, cid, model_parameters, baseline_type, train_loader, val_loader, config=None):
        """
        初始化不同baseline的客户端
        
        Args:
            cid: 客户端ID
            model_parameters: WPTCN模型参数
            baseline_type: baseline类型 ('TFL', 'HFL', 'FedMEC')
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 客户端配置
        """
        self.cid = cid
        self.model_parameters = model_parameters
        self.baseline_type = baseline_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 资源监控
        self.resource_monitor = ResourceMonitor()
        
        # 根据baseline类型创建模型
        if baseline_type == 'TFL':
            # 传统联邦学习：完整模型在设备上
            self.model = WPTCN(**model_parameters).to(self.device)
            self.partitioned_model = None
        else:
            # HFL或FedMEC：分区模型
            partition_point = None
            if baseline_type == 'HFL':
                # 分层联邦学习：固定在小波变换后分割
                partition_point = 'wavelet'
            elif baseline_type == 'FedMEC':
                # 经验分区：例如在第1个TCN层后分割
                partition_point = 'tcn_1'
                
            self.partitioned_model = PartitionedWPTCN(
                model_parameters=model_parameters,
                partition_point=partition_point
            )
            self.model = self.partitioned_model.get_device_model().to(self.device)
        
        # 训练性能统计
        self.train_stats = {
            'train_time': [],
            'communication_size': [],
        }

        self.config = config

    def get_parameters(self, config=None):
        """返回模型参数"""
        if self.baseline_type == 'TFL':
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        else:
            # 对于分区模型，返回完整模型参数
            return [val.cpu().numpy() for _, val in self.partitioned_model.state_dict().items()]

    def set_parameters(self, parameters):
        """设置模型参数"""
        if self.baseline_type == 'TFL':
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        else:
            # 对于分区模型，更新完整模型参数
            params_dict = zip(self.partitioned_model.full_model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.partitioned_model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """训练模型"""
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 启动资源监控
        self.resource_monitor.start()
        
        start_time = time.time()
        
        # 根据baseline类型进行训练
        if self.baseline_type == 'TFL':
            # 传统联邦学习：直接在设备上训练完整模型
            train_loss, train_acc = train_one_epoch(
                self.model, self.device, self.train_loader, criterion, optimizer
            )
        elif self.baseline_type in ['HFL', 'FedMEC']:
            # 分区模型训练：与边缘服务器通信
            train_loss, train_acc = self._train_with_edge_server(optimizer)
        
        end_time = time.time()
        training_time = end_time - start_time
        self.train_stats['train_time'].append(training_time)
        
        # 停止资源监控
        self.resource_monitor.stop()
        resource_stats = self.resource_monitor.get_statistics()
        
        # 准备返回的指标
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'training_time': training_time,
            'cpu_usage_mean': float(resource_stats['cpu']['mean']),
            'memory_usage_max': float(resource_stats['memory']['max']),
            'bandwidth_usage_mean': float(resource_stats['bandwidth']['mean']),
        }
        
        if 'gpu_memory' in resource_stats:
            metrics['gpu_memory_max'] = float(resource_stats['gpu_memory']['max'])
            
        if self.baseline_type != 'TFL':
            metrics['communication_size_total'] = sum(self.train_stats['communication_size'])
        
        return self.get_parameters(config=config), len(self.train_loader.dataset), metrics

    def _train_with_edge_server(self, optimizer):
        """与边缘服务器通信进行训练"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 获取当前轮次
        round_num = self.train_stats.get('round', 0)
        self.train_stats['round'] = round_num + 1
        
        # 定义边缘服务器URL
        edge_server_url = self.config.get("edge_server_url", "http://localhost:8000")
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 记录通信开始时间
            comm_start = time.time()
            
            # 设备端前向传播
            intermediate_output = self.model(data)
            
            # 将中间输出发送到边缘服务器
            features_list = intermediate_output.cpu().detach().numpy().tolist()
            labels_list = target.cpu().numpy().tolist()
            
            try:
                # 发送前向传播请求
                forward_response = requests.post(
                    f"{edge_server_url}/forward",
                    json={
                        "client_id": self.cid,
                        "round_num": round_num,
                        "features": features_list,
                        "labels": labels_list,
                        "shapes": list(intermediate_output.shape)
                    }
                ).json()
                
                # 获取输出和指标
                outputs = torch.tensor(forward_response["outputs"]).to(self.device)
                batch_loss = forward_response["metrics"]["loss"]
                batch_accuracy = forward_response["metrics"]["accuracy"]
                
                # 计算本批次大小和准确样本数
                batch_size = target.size(0)
                correct = batch_accuracy * batch_size
                
                # 累加损失和准确率
                total_loss += batch_loss * batch_size
                total_correct += correct
                total_samples += batch_size
                
                # 反向传播（从边缘服务器获取梯度）
                # 在实际场景中，我们需要将输出的梯度传回给边缘服务器，
                # 然后边缘服务器返回输入的梯度，这里简化处理
                backward_response = requests.post(
                    f"{edge_server_url}/backward",
                    json={
                        "client_id": self.cid,
                        "round_num": round_num,
                        "features": features_list,
                        "shapes": list(intermediate_output.shape)
                    }
                ).json()
                
                # 获取输入梯度
                input_gradients = torch.tensor(backward_response["input_gradients"]).to(self.device)
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 设置中间输出的梯度
                intermediate_output.backward(input_gradients)
                
                # 更新模型参数
                optimizer.step()
                
                # 记录通信结束时间和大小
                comm_end = time.time()
                comm_time = comm_end - comm_start
                
                # 计算通信大小（假设每个浮点数4字节）
                features_size = len(str(features_list)) * 4 / (1024 * 1024)  # MB
                gradients_size = len(str(backward_response["input_gradients"])) * 4 / (1024 * 1024)  # MB
                communication_size = features_size + gradients_size
                
                # 记录通信大小
                if len(self.train_stats['communication_size']) < len(self.train_loader):
                    self.train_stats['communication_size'].append(communication_size)
                else:
                    self.train_stats['communication_size'][-1] += communication_size
                
            except Exception as e:
                logger.error(f"Error communicating with edge server: {str(e)}")
                # 如果通信失败，跳过这个批次
                continue
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_accuracy

    def evaluate(self, parameters, config):
        """评估模型"""
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        
        if self.baseline_type == 'TFL':
            # 传统联邦学习：直接在设备上评估完整模型
            test_loss, test_acc, _, _, _ = evaluate(
                self.model, self.device, self.val_loader, criterion, mode='Validation'
            )
        else:
            # 分区模型评估：只评估设备部分，模拟与边缘服务器通信
            # 在实际部署中，这部分需要与服务器端协作完成
            # 这里简化模拟
            test_loss, test_acc = 0.0, 0.0
            
        return float(test_loss), len(self.val_loader.dataset), {"accuracy": float(test_acc)}

def main():
    logging.basicConfig(level=logging.INFO)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Baseline Flower client')
    parser.add_argument('--cid', type=str, required=True, help='Client ID (e.g., client_1)')
    parser.add_argument('--num_clients', type=int, required=True, help='Total number of clients')
    parser.add_argument('--baseline', type=str, choices=['TFL', 'HFL', 'FedMEC'], required=True, 
                      help='Baseline type: TFL, HFL or FedMEC')
    parser.add_argument('--server_address', type=str, default="localhost:8080", 
                      help='Server address (host:port)')
    args = parser.parse_args()

    client_id = args.cid
    num_clients = args.num_clients
    baseline_type = args.baseline
    server_address = args.server_address

    # 加载数据
    dataset = 'USC-HAD'  # 使用USC-HAD数据集
    batch_size = 32
    validation_split = 0.2

    try:
        train_loader, val_loader, test_loader, input_length, num_input_channels, num_classes = load_dataset(
            dataset, batch_size, validation_split, client_id=client_id, num_clients=num_clients
        )
    except FileNotFoundError as e:
        # 如果找不到数据集，使用模拟数据
        logging.warning(f"数据集加载失败: {e}，使用模拟数据代替")
        # 创建模拟数据
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader
        
        # 模拟输入数据：[batch_size, channels, sequence_length]
        input_length = 100  # USC-HAD数据集特征长度
        num_input_channels = 6  # USC-HAD数据集特征通道数
        num_classes = 12  # USC-HAD数据集类别数
        
        # 生成随机数据
        num_samples = 100
        X_train = torch.tensor(np.random.rand(num_samples, num_input_channels, input_length), dtype=torch.float32)
        y_train = torch.tensor(np.random.randint(0, num_classes, num_samples), dtype=torch.long)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_train[:20], y_train[:20])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型参数
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

    # 创建并启动客户端
    client = BaselineClient(client_id, model_parameters, baseline_type, train_loader, val_loader)
    fl.client.start_client(server_address=server_address, client=client)
    
if __name__ == "__main__":
    main()