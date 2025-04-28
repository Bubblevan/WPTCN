# federated/start_edge_server.py
import argparse
import requests
import json
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description="Start Edge Server")
    parser.add_argument('--edge_id', type=str, required=True, help='Edge server ID (e.g., edge_1)')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the edge server on')
    parser.add_argument('--partition_point', type=str, default='wavelet', 
                      help='Model partition point (wavelet or tcn_x)')
    parser.add_argument('--central_server', type=str, default='http://localhost:8080',
                      help='URL of the central parameter server')
    parser.add_argument('--num_clients', type=int, default=4, 
                      help='Number of clients this edge server will handle')
    args = parser.parse_args()
    
    # 启动边缘服务器进程
    import subprocess
    import time
    
    edge_server_cmd = [
        "python", "-m", "federated.edge_server",
        "--host", "0.0.0.0",
        "--port", str(args.port)
    ]
    
    edge_process = subprocess.Popen(edge_server_cmd)
    
    # 等待边缘服务器启动
    time.sleep(2)
    
    # 初始化模型参数（从模型配置中读取）
    model_parameters = {
        'num_input_channels': 6,  # USC-HAD数据集
        'input_length': 100,
        'num_classes': 12,
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
    
    # 初始化边缘服务器
    try:
        response = requests.post(
            f"http://localhost:{args.port}/init",
            json={
                "edge_id": args.edge_id,
                "model_parameters": model_parameters,
                "partition_point": args.partition_point,
                "central_server_url": args.central_server
            }
        )
        
        if response.status_code == 200:
            print(f"Edge server {args.edge_id} initialized successfully")
            print(f"Waiting for clients to connect to http://localhost:{args.port}")
            
            # 等待边缘服务器进程
            edge_process.wait()
        else:
            print(f"Failed to initialize edge server: {response.text}")
            edge_process.terminate()
    except Exception as e:
        print(f"Error initializing edge server: {str(e)}")
        edge_process.terminate()

if __name__ == "__main__":
    main()