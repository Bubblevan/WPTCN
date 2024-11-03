import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneFITS(nn.Module):
    def __init__(self, n_steps, n_features, n_pred_steps, cut_freq, individual, num_classes):
        super(BackboneFITS, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.individual = individual
        self.num_classes = num_classes

        self.dominance_freq = cut_freq
        self.length_ratio = (n_steps + n_pred_steps) / n_steps

        # 频率上采样层
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.n_features):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio), bias=False)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio), bias=False
            )

        # 分类器输入维度
        classifier_input_dim = (n_steps + n_pred_steps) * n_features

        # 分类器设计
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 计算输入信号的FFT，沿时间步长维度 (dim=2)
        low_specx = torch.fft.rfft(x, dim=2)

        # 低通滤波：保留前dominance_freq个频率分量，其他置零
        low_specx[:, :, self.dominance_freq:] = 0
        low_specx = low_specx[:, :, :self.dominance_freq]

        # 使用幅度代替复数数据
        low_specx_mag = torch.abs(low_specx)
        
        # 频率上采样
        if self.individual:
            low_specxy_ = torch.zeros(
                (low_specx_mag.size(0), self.n_features, int(self.dominance_freq * self.length_ratio)),
                dtype=low_specx_mag.dtype,
                device=low_specx_mag.device
            )
            for i in range(self.n_features):
                upsampled = self.freq_upsampler[i](low_specx_mag[:, i])
                low_specxy_[:, i, :] = upsampled
        else:
            batch_size, n_features, dominance_freq = low_specx_mag.shape
            low_specx_mag_reshaped = low_specx_mag.view(batch_size * n_features, dominance_freq)
            low_specxy_reshaped = self.freq_upsampler(low_specx_mag_reshaped)
            low_specxy_ = low_specxy_reshaped.view(batch_size, n_features, -1)

        # 频域零填充
        expected_freq_length = int((self.n_steps + self.n_pred_steps) / 2 + 1)
        low_specxy = torch.zeros(
            (low_specxy_.size(0), self.n_features, expected_freq_length),
            dtype=low_specxy_.dtype,
            device=low_specxy_.device
        )
        low_specxy[:, :, :low_specxy_.size(2)] = low_specxy_

        # 计算逆FFT得到时域信号
        low_xy = torch.fft.irfft(low_specxy, n=self.n_steps + self.n_pred_steps, dim=2)
        low_xy = low_xy * self.length_ratio

        # 将特征展平为一维向量
        low_xy_flat = low_xy.view(low_xy.size(0), -1)

        # 输入分类器得到输出
        output = self.classifier(low_xy_flat)
        return output
    
