import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import json
import time
import logging
import requests
import asyncio
from models.partitioned_wptcn import PartitionedWPTCN

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("edge_server")

app = FastAPI()

# 边缘服务器状态
class EdgeServerState:
    def __init__(self, edge_id, model_parameters, partition_point, central_server_url):
        self.edge_id = edge_id
        self.model_parameters = model_parameters
        self.partition_point = partition_point
        self.central_server_url = central_server_url
        
        # 创建分区模型
        self.partitioned_model = PartitionedWPTCN(
            model_parameters=model_parameters,
            partition_point=partition_point
        )
        
        # 初始化边缘服务器模型
        self.server_model = self.partitioned_model.get_server_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model.to(self.device)
        
        # 性能统计
        self.performance_metrics = {
            "computation_time": [],
            "communication_time": [],
            "accuracy": [],
            "loss": []
        }
        
        # 客户端缓存
        self.client_features = {}
        self.client_labels = {}
        self.clients_ready = set()
        
        # 训练参数
        self.criterion = nn.CrossEntropyLoss()
        self.current_round = 0
        self.aggregated_grads = None
        
        logger.info(f"Edge Server {edge_id} initialized with partition point {partition_point}")

# 边缘服务器状态实例
edge_state = None

# 模型输入数据模型
class FeatureData(BaseModel):
    client_id: str
    round_num: int
    features: List[List[float]]
    labels: Optional[List[int]] = None
    shapes: List[int]

# 模型参数数据模型
class ModelUpdate(BaseModel):
    parameters: List[List[float]]
    shapes: List[List[int]]

# 初始化请求模型
class InitRequest(BaseModel):
    edge_id: str
    model_parameters: Dict
    partition_point: str
    central_server_url: str

@app.post("/init")
async def initialize(request: InitRequest):
    """初始化边缘服务器"""
    global edge_state
    
    edge_state = EdgeServerState(
        edge_id=request.edge_id,
        model_parameters=request.model_parameters,
        partition_point=request.partition_point,
        central_server_url=request.central_server_url
    )
    
    return {"status": "success", "message": f"Edge server {request.edge_id} initialized"}

@app.post("/forward")
async def forward_pass(data: FeatureData):
    """处理客户端的前向传播请求"""
    global edge_state
    
    if edge_state is None:
        raise HTTPException(status_code=400, detail="Edge server not initialized")
    
    # 记录开始时间
    start_time = time.time()
    
    client_id = data.client_id
    round_num = data.round_num
    
    # 转换特征为Tensor
    features_array = np.array(data.features, dtype=np.float32)
    features_tensor = torch.tensor(features_array).reshape(data.shapes).to(edge_state.device)
    
    # 存储标签（如果有）
    if data.labels is not None:
        labels_tensor = torch.tensor(data.labels, dtype=torch.long).to(edge_state.device)
        edge_state.client_labels[client_id] = labels_tensor
    
    # 使用边缘服务器模型计算前向传播
    with torch.no_grad():
        outputs = edge_state.server_model(features_tensor)
    
    # 计算损失和准确率（如果有标签）
    metrics = {}
    if data.labels is not None:
        loss = edge_state.criterion(outputs, edge_state.client_labels[client_id])
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == edge_state.client_labels[client_id]).float().mean().item()
        
        metrics["loss"] = loss.item()
        metrics["accuracy"] = accuracy
    
    # 记录计算时间
    computation_time = time.time() - start_time
    edge_state.performance_metrics["computation_time"].append(computation_time)
    
    # 将输出转换为列表并返回
    outputs_list = outputs.cpu().detach().numpy().tolist()
    
    logger.info(f"Forward pass completed for client {client_id} in round {round_num}")
    
    return {
        "outputs": outputs_list,
        "metrics": metrics,
        "computation_time": computation_time
    }

@app.post("/backward")
async def backward_pass(data: FeatureData, background_tasks: BackgroundTasks):
    """处理客户端的反向传播请求"""
    global edge_state
    
    if edge_state is None:
        raise HTTPException(status_code=400, detail="Edge server not initialized")
    
    # 记录开始时间
    start_time = time.time()
    
    client_id = data.client_id
    round_num = data.round_num
    
    # 转换特征为Tensor
    features_array = np.array(data.features, dtype=np.float32)
    features_tensor = torch.tensor(features_array).reshape(data.shapes).to(edge_state.device)
    
    # 边缘服务器模型需要计算梯度
    features_tensor.requires_grad_(True)
    
    # 前向传播
    outputs = edge_state.server_model(features_tensor)
    
    # 计算损失
    if client_id not in edge_state.client_labels:
        raise HTTPException(status_code=400, detail="Labels not found for client")
    
    loss = edge_state.criterion(outputs, edge_state.client_labels[client_id])
    
    # 反向传播
    loss.backward()
    
    # 获取输入梯度
    input_gradients = features_tensor.grad.cpu().detach().numpy().tolist()
    
    # 标记客户端已完成
    edge_state.clients_ready.add(client_id)
    
    # 如果所有客户端都已完成，聚合梯度并发送至中央服务器
    background_tasks.add_task(check_aggregation_needed, round_num)
    
    # 记录计算时间
    computation_time = time.time() - start_time
    edge_state.performance_metrics["computation_time"].append(computation_time)
    
    logger.info(f"Backward pass completed for client {client_id} in round {round_num}")
    
    return {
        "input_gradients": input_gradients,
        "computation_time": computation_time
    }

@app.post("/update_model")
async def update_model(update: ModelUpdate):
    """从中央服务器更新模型参数"""
    global edge_state
    
    if edge_state is None:
        raise HTTPException(status_code=400, detail="Edge server not initialized")
    
    # 将参数列表转换为numpy数组，然后转换为PyTorch张量
    parameters = []
    for param, shape in zip(update.parameters, update.shapes):
        param_array = np.array(param, dtype=np.float32).reshape(shape)
        parameters.append(torch.tensor(param_array))
    
    # 更新边缘服务器模型参数
    with torch.no_grad():
        for param, new_param in zip(edge_state.server_model.parameters(), parameters):
            param.copy_(new_param)
    
    edge_state.current_round += 1
    # 清空客户端缓存
    edge_state.client_features = {}
    edge_state.client_labels = {}
    edge_state.clients_ready = set()
    
    logger.info(f"Model updated for round {edge_state.current_round}")
    
    return {"status": "success", "round": edge_state.current_round}

@app.get("/metrics")
async def get_metrics():
    """获取边缘服务器性能指标"""
    global edge_state
    
    if edge_state is None:
        raise HTTPException(status_code=400, detail="Edge server not initialized")
    
    return edge_state.performance_metrics

async def check_aggregation_needed(round_num):
    """检查是否需要聚合梯度并发送至中央服务器"""
    global edge_state
    
    # 实际生产环境中，这里应该有更复杂的逻辑来确定何时执行聚合
    # 当足够多的客户端完成时进行聚合
    
    # 简单延迟，以模拟等待其他客户端
    await asyncio.sleep(2)
    
    logger.info(f"Aggregating updates from {len(edge_state.clients_ready)} clients in round {round_num}")
    
    # 这里应该实现聚合逻辑，然后发送至中央服务器
    # 仅作为示例，这里省略了实际的聚合逻辑
    # 实际应用中，您需要实现梯度或参数的聚合
    
    try:
        # 通知中央服务器本轮边缘聚合已完成
        requests.post(
            f"{edge_state.central_server_url}/edge_aggregation_complete",
            json={
                "edge_id": edge_state.edge_id,
                "round_num": round_num,
                "clients_count": len(edge_state.clients_ready)
            }
        )
        logger.info(f"Notified central server about round {round_num} completion")
    except Exception as e:
        logger.error(f"Error notifying central server: {str(e)}")

def start():
    """启动边缘服务器"""
    parser = argparse.ArgumentParser(description="Edge Server")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    logger.info(f"Starting Edge Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    import argparse
    start()
