import os
import sys
import subprocess
import time
import argparse
import logging
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_comparison_results(metrics_files, result_dir):
    """绘制不同baseline的比较结果"""
    baseline_data = {}
    
    # 加载指标数据
    for file_path in metrics_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            baseline_type = data['baseline_type']
            baseline_data[baseline_type] = data
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('WPTCN Baseline Comparison')
    
    # 1. 训练时间对比
    ax = axs[0, 0]
    for baseline, data in baseline_data.items():
        rounds = [m['round'] for m in data['round_metrics']]
        times = [m['aggregated_metrics'].get('mean_training_time', 0) for m in data['round_metrics']]
        ax.plot(rounds, times, 'o-', label=baseline)
    ax.set_xlabel('Round')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Average Training Time per Round')
    ax.legend()
    ax.grid(True)
    
    # 2. 内存使用对比
    ax = axs[0, 1]
    for baseline, data in baseline_data.items():
        rounds = [m['round'] for m in data['round_metrics']]
        memory = [m['aggregated_metrics'].get('max_memory_usage_max', 0) for m in data['round_metrics']]
        ax.plot(rounds, memory, 'o-', label=baseline)
    ax.set_xlabel('Round')
    ax.set_ylabel('Memory Usage (%)')
    ax.set_title('Maximum Memory Usage per Round')
    ax.legend()
    ax.grid(True)
    
    # 3. 准确率对比
    ax = axs[1, 0]
    for baseline, data in baseline_data.items():
        rounds = [m['round'] for m in data['round_metrics']]
        accuracy = [m['aggregated_metrics'].get('mean_train_accuracy', 0) for m in data['round_metrics']]
        ax.plot(rounds, accuracy, 'o-', label=baseline)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True)
    
    # 4. 带宽使用对比 (HFL和FedMEC才有)
    ax = axs[1, 1]
    for baseline, data in baseline_data.items():
        if baseline in ['HFL', 'FedMEC']:
            rounds = [m['round'] for m in data['round_metrics']]
            bandwidth = [m['aggregated_metrics'].get('mean_bandwidth_usage_mean', 0) for m in data['round_metrics']]
            ax.plot(rounds, bandwidth, 'o-', label=baseline)
    ax.set_xlabel('Round')
    ax.set_ylabel('Bandwidth Usage (MB/s)')
    ax.set_title('Average Bandwidth Usage')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    # 保存到result目录
    output_path = os.path.join(result_dir, 'baseline_comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def run_baseline(baseline_type, num_clients=4, num_rounds=10):
    """运行指定的baseline测试"""
    # 为不同baseline分配不同端口
    server_port_mapping = {
        'TFL': 8080,
        'HFL': 8081,
        'FedMEC': 8082
    }
    
    edge_port_mapping = {
        'HFL': 9000,
        'FedMEC': 9001
    }
    
    server_port = server_port_mapping.get(baseline_type, 8080)
    
    logging.info(f"Running {baseline_type} baseline with {num_clients} clients and {num_rounds} rounds on port {server_port}")
    
    edge_processes = []
    
    # 对于HFL和FedMEC，启动边缘服务器
    if baseline_type in ['HFL', 'FedMEC']:
        edge_port = edge_port_mapping.get(baseline_type, 9000)
        partition_point = 'wavelet' if baseline_type == 'HFL' else 'tcn_1'
        
        edge_cmd = [
            "python", "federated/start_edge_server.py",
            "--edge_id", f"edge_1",
            "--port", str(edge_port),
            "--partition_point", partition_point,
            "--central_server", f"http://localhost:{server_port}",
            "--num_clients", str(num_clients)
        ]
        
        edge_process = subprocess.Popen(edge_cmd)
        edge_processes.append(edge_process)
        
        # 等待边缘服务器启动
        time.sleep(5)
    
    # 启动服务器进程
    server_cmd = [
        "python", "federated/baseline_server.py", 
        "--baseline", baseline_type,
        "--rounds", str(num_rounds),
        "--min_clients", str(num_clients // 2),
        "--port", str(server_port)
    ]
    server_process = subprocess.Popen(server_cmd)
    
    # 等待服务器启动
    time.sleep(5)
    
    # 启动客户端进程
    client_processes = []
    for i in range(num_clients):
        client_cmd = [
            "python", "federated/baseline_client.py",
            "--cid", f"client_{i+1}",
            "--num_clients", str(num_clients),
            "--baseline", baseline_type,
            "--server_address", f"localhost:{server_port}"
        ]
        
        # 对于HFL和FedMEC，添加边缘服务器参数
        if baseline_type in ['HFL', 'FedMEC']:
            edge_port = edge_port_mapping.get(baseline_type, 9000)
            client_cmd.extend(["--edge_server", f"http://localhost:{edge_port}"])
            
        client_process = subprocess.Popen(client_cmd)
        client_processes.append(client_process)
    
    # 等待服务器进程完成
    server_process.wait()
    
    # 终止所有客户端进程
    for process in client_processes:
        process.terminate()
    
    # 终止所有边缘服务器进程
    for process in edge_processes:
        process.terminate()
    
    # 给端口一些释放时间
    time.sleep(2)
    
    logging.info(f"Completed {baseline_type} baseline")

def main():
    parser = argparse.ArgumentParser(description='Run WPTCN baseline comparison')
    parser.add_argument('--clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of training rounds')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 确保result目录存在
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(result_dir, exist_ok=True)
    
    baselines = ['TFL', 'HFL', 'FedMEC']
    metrics_files = []
    
    for baseline in baselines:
        run_baseline(baseline, num_clients=args.clients, num_rounds=args.rounds)
        
        # 找到最新的指标文件 - 现在在result目录中查找
        files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) 
                if f.startswith(f'metrics_{baseline}_')]
        if files:
            latest_file = max(files, key=os.path.getctime)
            metrics_files.append(latest_file)
    
    # 绘制比较结果
    if metrics_files:
        output_path = plot_comparison_results(metrics_files, result_dir)
        logging.info(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()