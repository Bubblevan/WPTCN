
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
python client.py --cid client_1 --num_clients 3
python client.py --cid client_2 --num_clients 3
python client.py --cid client_3 --num_clients 3
```