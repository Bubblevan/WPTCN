import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
project_root = os.path.abspath(os.path.join(script_dir, '..'))  # 获取项目根目录
sys.path.insert(0, project_root)  # 将项目根目录添加到Python路径

import flwr as fl
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    """保存训练指标的联邦学习策略"""
    
    def __init__(self, baseline_type: str, **kwargs):
        super().__init__(**kwargs)
        self.baseline_type = baseline_type
        self.round_metrics = []
        self.client_metrics = {}
        
    def aggregate_fit(self, server_round, results, failures):
        """聚合训练结果并保存指标"""
        # 保存每轮的客户端指标
        round_client_metrics = {}
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            round_client_metrics[client_id] = metrics
            
            # 更新客户端历史指标
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = []
            self.client_metrics[client_id].append(metrics)
            
        # 计算聚合指标
        aggregated_metrics = {}
        for metric_name in ['train_loss', 'train_accuracy', 'training_time', 
                           'cpu_usage_mean', 'memory_usage_max', 'bandwidth_usage_mean']:
            if all(metric_name in metrics for metrics in round_client_metrics.values()):
                values = [metrics[metric_name] for metrics in round_client_metrics.values()]
                aggregated_metrics[f'mean_{metric_name}'] = sum(values) / len(values)
                aggregated_metrics[f'max_{metric_name}'] = max(values)
        
        self.round_metrics.append({
            'round': server_round,
            'aggregated_metrics': aggregated_metrics,
            'client_metrics': round_client_metrics
        })
        
        # 保存指标到文件
        self._save_metrics()
        
        # 继续正常的聚合过程
        return super().aggregate_fit(server_round, results, failures)
    
    def _save_metrics(self):
        """保存指标到文件"""
        # 确保result目录存在
        result_dir = os.path.join(project_root, "result")
        os.makedirs(result_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(result_dir, f"metrics_{self.baseline_type}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump({
                'baseline_type': self.baseline_type,
                'round_metrics': self.round_metrics,
                'client_metrics': self.client_metrics
            }, f, indent=2)
        
        logging.info(f"Metrics saved to {filename}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Baseline Flower server')
    parser.add_argument('--baseline', type=str, choices=['TFL', 'HFL', 'FedMEC'], required=True, 
                      help='Baseline type: TFL, HFL or FedMEC')
    parser.add_argument('--rounds', type=int, default=10, help='Number of training rounds')
    parser.add_argument('--min_clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 根据baseline类型调整配置
    if args.baseline == 'TFL':
        # 传统联邦学习：所有客户端直接与服务器通信
        fraction_fit = 1.0  # 使用所有可用客户端
    elif args.baseline == 'HFL':
        # 分层联邦学习：部分客户端作为边缘节点
        fraction_fit = 0.8  # 使用80%的客户端
    elif args.baseline == 'FedMEC':
        # 经验分区：固定分区策略
        fraction_fit = 0.8  # 类似HFL
    
    # 配置联邦学习策略
    strategy = SaveMetricsStrategy(
        baseline_type=args.baseline,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    # 创建 ServerConfig 对象
    config = fl.server.ServerConfig(num_rounds=args.rounds)

    # 启动服务器
    fl.server.start_server(
        server_address=f"[::]:{args.port}",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

