
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
python client.py --num_clients 4 --cid client_1
python client.py --num_clients 4 --cid client_2
python client.py --num_clients 4 --cid client_3
python client.py --num_clients 4 --cid client_4
python run_baseline_comparison.py --clients 4 --rounds 5

```

������python run_baseline_comparison.py --clients 4 --rounds 5ʱ���ű���ִ�����²�����
1. ��ʼ������
����������ʹ��4���ͻ��ˣ�����5��ѵ��
����WPTCN/resultĿ¼�����ڱ�������������
������־��ʽ

2. ˳��������ֻ��߷���
����ÿ�ֻ��߷�����TFL��HFL��FedMEC�������ִ�У�
��ͳ����ѧϰ (TFL)
- ����˿�8080
- ��������������
- ����4���ͻ��˽��̣�ÿ���ͻ��ˣ�
    - ����ģ�����ݣ�USC-HAD���ݼ���
    - ���豸��ѵ��������WPTCNģ��
    - �ռ�ѵ��ʱ�䡢�ڴ�ʹ�õ�ָ��
- �������ռ����ۺ����пͻ��˵�ָ��
- ��ָ�걣��ΪJSON�ļ���resultĿ¼
- �����������Ϳͻ��˽���

�ֲ�����ѧϰ (HFL)
- ����˿�8081
- ��������������
- ����4���ͻ��˽��̣�ÿ���ͻ��ˣ�
    - ���豸��ֻ����С���任ǰ�Ĳ���
    - ģ�����Ե������ͨ��
    - �ռ�ָ�꣨����ͨ�ſ�����
- ����ۺ�ָ�굽JSON�ļ�

������� (FedMEC)
- ����˿�8082
- ������HFL���������㲻ͬ���ڵ�1��TCN���
- ���в��Բ�����ָ��

3. ���ɱȽϽ��
�����������߷�����ָ��JSON�ļ�
�����ĸ��Ա�ͼ��
ѵ��ʱ��Ա�
�ڴ�ʹ�öԱ�
׼ȷ�ʶԱ�
ͨ�Ŵ���Աȣ���HFL��FedMEC��