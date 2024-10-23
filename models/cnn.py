import torch.nn as nn

'''Convolutional Neural Network'''
class CNN(nn.Module):
    def __init__(self, num_input_channels, num_classes, input_length):
        super(CNN, self).__init__()
        '''
            num_input_channels: 输入通道数
            num_classes: 类别数
            input_length: 输入序列长度
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, input_length))
        self.fc = nn.Linear(512*input_length, num_classes)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        if x.dim() == 3:
            x = x.unsqueeze(2)  # 在第三个维度增加一个维度，使形状变为 [batch_size, num_input_channels, 1, input_length]
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
