
# WPTCN: ����С��������任�������칹ģ��

try.py: �ڲ�ͬ���ݼ���ʵ��WPTCN
train.py: ʹ��baselineʵ��
```
cd test
python try.py
```


federated/server.py: ����Flower�����
federated/client.py: ����Flower�ͻ���
```
cd federated
python server.py
python client.py --cid client_1 --num_clients 3
python client.py --cid client_2 --num_clients 3
python client.py --cid client_3 --num_clients 3
```