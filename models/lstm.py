import torch.nn as nn

'''Long Short Term Memory Network'''
class LSTM(nn.Module):
    def __init__(self, num_input_channels, num_classes, input_length):
        super().__init__()
        '''
            num_input_channels: 输入通道数
            num_classes: 类别数
            input_length: 输入序列长度
        '''
        self.lstm = nn.LSTM(input_length, 512, 2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x, _ = self.lstm(x.squeeze(1))
        x = x[:, -1, :]
        x = self.fc(x)
        return x
