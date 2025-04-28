
# WPTCN: 基于小波包卷积变换的联邦异构模型

try.py: 在不同数据集下实验WPTCN
train.py: 使用baseline实验
```
cd test
python try.py
```


federated/server.py: 启动Flower服务端
federated/client.py: 启动Flower客户端
```
cd federated
python server.py
python client.py --num_clients 4 --cid client_1
python client.py --num_clients 4 --cid client_2
python client.py --num_clients 4 --cid client_3
python client.py --num_clients 4 --cid client_4
python run_baseline_comparison.py --clients 4 --rounds 5

```

当运行python run_baseline_comparison.py --clients 4 --rounds 5时，脚本会执行以下操作：
1. 初始化设置
解析参数：使用4个客户端，进行5轮训练
创建WPTCN/result目录，用于保存所有输出结果
设置日志格式

2. 顺序测试三种基线方法
对于每种基线方法（TFL、HFL、FedMEC），逐个执行：
传统联邦学习 (TFL)
- 分配端口8080
- 启动服务器进程
- 启动4个客户端进程，每个客户端：
    - 加载模拟数据（USC-HAD数据集）
    - 在设备上训练完整的WPTCN模型
    - 收集训练时间、内存使用等指标
- 服务器收集并聚合所有客户端的指标
- 将指标保存为JSON文件到result目录
- 结束服务器和客户端进程

分层联邦学习 (HFL)
- 分配端口8081
- 启动服务器进程
- 启动4个客户端进程，每个客户端：
    - 在设备上只运行小波变换前的部分
    - 模拟与边缘服务器通信
    - 收集指标（包括通信开销）
- 保存聚合指标到JSON文件

经验分区 (FedMEC)
- 分配端口8082
- 类似于HFL，但分区点不同（在第1个TCN层后）
- 运行测试并保存指标

3. 生成比较结果
检索三个基线方法的指标JSON文件
生成四个对比图表：
训练时间对比
内存使用对比
准确率对比
通信带宽对比（仅HFL和FedMEC）