import torch.nn as nn

'''Channel Attention Neural Network: 通道注意力'''
class ChannelAttentionModule(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.att_fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // 4),
            nn.ReLU(),
            nn.Linear(inchannel // 4, inchannel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        att = nn.AdaptiveAvgPool2d((1, x.size(-1)))(x)  # [b, c, series, modal] -> [b, c, 1, modal]
        att = att.permute(0, 3, 1, 2).squeeze(-1)  # [b, c, 1, modal] -> [b, modal, c]
        att = self.att_fc(att)  # [b, modal, c]
        att = att.permute(0, 2, 1).unsqueeze(-2)  # [b, modal, c] -> [b, c, modal] -> [b, c, 1, modal]
        out = x * att
        return out
    

class ChannelAttentionNeuralNetwork(nn.Module):
    def __init__(self, num_input_channels, num_classes, input_length):
        super(ChannelAttentionNeuralNetwork, self).__init__()
        '''
            num_input_channels: 输入通道数
            num_classes: 类别数
            input_length: 输入序列长度
        '''
        # Adjusted first Conv2d layer to take `num_input_channels` as input
        self.layer = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adjusted pooling to (1, 1)
        self.fc = nn.Linear(512, num_classes)  # Reduced final FC layer input size to match output channels

    def forward(self, x):
        '''
            x.shape: [batch_size, num_input_channels, input_length]
        '''
        # Ensure input is properly reshaped
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [batch_size, num_input_channels, 1, input_length]

        # Apply layers
        x = self.layer(x)

        # Adaptive pooling
        x = self.ada_pool(x)

        # Flatten before fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x