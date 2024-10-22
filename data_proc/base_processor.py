# data_proc/base_processor.py
import abc
from torch.utils.data import DataLoader

class BaseProcessor(abc.ABC):
    @abc.abstractmethod
    def load_data(self):
        """加载原始数据"""
        pass

    @abc.abstractmethod
    def preprocess(self):
        """预处理数据，包括清洗、分割、标准化等"""
        pass

    @abc.abstractmethod
    def create_dataloaders(self, batch_size: int):
        """创建训练、验证和测试的DataLoader"""
        pass
