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
