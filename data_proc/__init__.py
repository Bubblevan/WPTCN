# data_proc/__init__.py
from .uci_har_proc import UCIHARProcessor
from .wisdm_proc import WISDMProcessor

def get_processor(dataset_name, data_dir, batch_size, **kwargs):
    """
    工厂函数，根据数据集名称返回相应的处理器实例。

    参数:
        dataset_name (str): 数据集名称，例如 'uci-har' 或 'wisdm'。
        data_dir (str): 数据集的根目录。
        batch_size (int): 批量大小。
        **kwargs: 其他可选参数。

    返回:
        BaseProcessor: 相应的数据处理器实例。
    """
    if dataset_name.lower() == 'uci-har':
        return UCIHARProcessor(data_dir, batch_size, **kwargs)
    elif dataset_name.lower() == 'wisdm':
        return WISDMProcessor(data_dir, batch_size, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
