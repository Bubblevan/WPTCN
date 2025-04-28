import torch
import torch.nn as nn
from models.WPTCN import WPTCN, WTTCNBlock, CustomNormalization

# 添加一个扁平化层类
class Flatten(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)

class PartitionedWPTCN:
    """可分区的WPTCN模型，支持在设备和边缘服务器间分割"""
    
    def __init__(self, model_parameters, partition_point=None):
        """
        初始化可分区的WPTCN模型
        
        Args:
            model_parameters: WPTCN的参数
            partition_point: 模型分区点(None=完整模型, 'wavelet'=小波变换后分割, 'tcn_x'=第x个TCN层后分割)
        """
        self.full_model = WPTCN(**model_parameters)
        self.partition_point = partition_point
        self.device_model = None
        self.server_model = None
        # 保存模型参数，以便后续重新创建分区模型
        self.model_parameters = model_parameters
        
        if partition_point is not None:
            self._create_partitioned_models()

    def _create_partitioned_models(self):
        """根据分区点创建设备端和服务器端模型"""
        if self.partition_point == 'wavelet':
            # 设备端只保留归一化和初始卷积
            self.device_model = nn.Sequential(
                CustomNormalization(eps=self.model_parameters['normalization_eps'], 
                                affine=self.model_parameters['normalization_affine']),
                self.full_model.initial_conv
            )
            
            # 服务器端包含TCN骨干网络和分类层
            backbone_modules = list(self.full_model.backbone.children())
            self.server_model = nn.Sequential(
                *backbone_modules,
                self.full_model.global_pool,
                Flatten(),  # 使用自定义模块替代lambda
                self.full_model.fc
            )
            
        elif self.partition_point.startswith('tcn_'):
            # 提取TCN层索引
            tcn_layer_idx = int(self.partition_point.split('_')[1])
            
            # 设备端包含归一化、初始卷积和部分TCN层
            backbone_modules = list(self.full_model.backbone.children())
            device_modules = [
                CustomNormalization(eps=self.model_parameters['normalization_eps'], 
                                affine=self.model_parameters['normalization_affine']),
                self.full_model.initial_conv
            ]
            device_modules.extend(backbone_modules[:tcn_layer_idx])
            self.device_model = nn.Sequential(*device_modules)
            
            # 服务器端包含剩余TCN层和分类层
            server_modules = backbone_modules[tcn_layer_idx:]
            self.server_model = nn.Sequential(
                *server_modules,
                self.full_model.global_pool,
                Flatten(),  # 使用自定义模块替代lambda
                self.full_model.fc
            )
    def get_device_model(self):
        """获取设备端模型"""
        return self.device_model if self.partition_point else self.full_model
    
    def get_server_model(self):
        """获取服务器端模型"""
        return self.server_model if self.partition_point else None
    
    def device_forward(self, x):
        """设备端前向传播"""
        if self.partition_point:
            return self.device_model(x)
        else:
            return self.full_model(x)
            
    def server_forward(self, x):
        """服务器端前向传播"""
        if self.partition_point:
            return self.server_model(x)
        else:
            return x  # 完整模型在设备端运行，服务器不处理
    
    def load_state_dict(self, state_dict):
        """加载模型参数"""
        self.full_model.load_state_dict(state_dict)
        if self.partition_point:
            # 重新创建分区模型，使用之前保存的模型参数
            self._create_partitioned_models()
    
    def state_dict(self):
        """获取完整模型参数"""
        return self.full_model.state_dict()