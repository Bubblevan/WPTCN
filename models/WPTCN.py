import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class CustomNormalization(nn.Module):
    def __init__(self, eps=1e-5, affine=True):
        super(CustomNormalization, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1))
        
    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

class WTTCNBlock(nn.Module):
    def __init__(self, D, kernel_size=3, num_levels=2, wavelet_type='db1', r=1, group_type='channel'):
        super(WTTCNBlock, self).__init__()
        self.D = D
        self.num_levels = num_levels
        self.wavelet_type = wavelet_type
        
        # 小波包滤波器参数
        wavelet = pywt.Wavelet(wavelet_type)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        
        # 深度卷积
        self.dw_conv = nn.Conv1d(
            in_channels=D, 
            out_channels=D, 
            kernel_size=kernel_size, 
            groups=D, 
            padding='same'
        )
        self.bn = nn.BatchNorm1d(D)
        
        # 前馈网络
        if group_type == 'channel':
            groups_num = D
        else:
            groups_num = 1
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(D, D * r, kernel_size=1, groups=groups_num),
            nn.GELU(),
            nn.Conv1d(D * r, D, kernel_size=1, groups=groups_num)
        )
        
        # 可学习的尺度因子
        self.scale_factor = nn.Parameter(torch.ones(1), requires_grad=True)
        
    def wavelet_packet_decompose(self, x, level):
        if level == 0:
            return [x]
        else:
            # 低频和高频分解
            x_lo = F.conv1d(x, self.dec_lo.unsqueeze(0).unsqueeze(0).repeat(self.D, 1, 1), stride=2, padding=0, groups=self.D)
            x_hi = F.conv1d(x, self.dec_hi.unsqueeze(0).unsqueeze(0).repeat(self.D, 1, 1), stride=2, padding=0, groups=self.D)
            # 递归分解
            coeffs_lo = self.wavelet_packet_decompose(x_lo, level - 1)
            coeffs_hi = self.wavelet_packet_decompose(x_hi, level - 1)
            return coeffs_lo + coeffs_hi
    
    def forward(self, x):
        # 小波包分解
        coeffs = self.wavelet_packet_decompose(x, self.num_levels)
        # 处理各个子频带
        processed_coeffs = [self.dw_conv(c) for c in coeffs]
        # 融合子频带
        x_fused = sum(processed_coeffs) / len(processed_coeffs)
        x_fused = self.scale_factor * x_fused
        
        # BatchNorm 和 激活
        x_fused = self.bn(x_fused)
        x_fused = F.relu(x_fused)
        
        # 前馈网络
        ffn_out = self.conv_ffn(x_fused)
        
        # 残差连接
        return ffn_out + x_fused

class WPTCN(nn.Module):
    def __init__(self, num_input_channels, input_length, num_classes, hidden_dim, kernel_size, num_levels, num_layers, wavelet_type, feedforward_ratio, group_type, normalization_eps, normalization_affine):
        super(WPTCN, self).__init__()
        self.normalization = CustomNormalization(eps=normalization_eps, affine=normalization_affine)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(num_input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.backbone = nn.ModuleList([
            WTTCNBlock(D=hidden_dim, kernel_size=kernel_size, num_levels=num_levels, wavelet_type=wavelet_type, r=feedforward_ratio, group_type=group_type) 
            for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, input_length)
    
    def forward(self, x):
        x = self.normalization(x)
        x = self.initial_conv(x)
        for layer in self.backbone:
            x = layer(x)
        x = self.global_pool(x).squeeze(-1)
        output = self.fc(x)
        return output
