import torch.nn as nn

'''Residual Neural Network'''
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 1, 1, 0),
            nn.BatchNorm2d(outchannel)
        )
        self.short = nn.Sequential()
        # Fix: Adjust shortcut to ensure matching channel dimensions
        if (inchannel != outchannel or stride != 1):
            self.short = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride),  # Changed kernel size to 1 to match channel dimensions properly
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        out = self.block(x) + self.short(x)
        return nn.ReLU()(out)

    
class ResNet(nn.Module):
    def __init__(self, num_input_channels, num_classes, input_length):
        super().__init__()
        '''
            num_input_channels: 输入通道数
            num_classes: 类别数
            input_length: 输入序列长度
        '''
        # Define the layers, starting from num_input_channels as the first input
        self.layer1 = self.make_layers(num_input_channels, 64, 1, 2)  # Updated stride for first layer
        self.layer2 = self.make_layers(64, 128, 1, 2)
        self.layer3 = self.make_layers(128, 256, 1, 2)
        self.layer4 = self.make_layers(256, 512, 1, 2)
        
        # Update Adaptive Pooling to ensure it works with the reshaped input length
        self.ada_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to [batch, 512, 1, 1]
        self.fc = nn.Linear(512, num_classes)  # Fully connected to output num_classes

    def forward(self, x):
        '''
            x.shape: [batch_size, num_input_channels, input_length]
        '''
        # Ensure input is properly reshaped
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [batch_size, num_input_channels, 1, input_length]

        # Pass through layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive Pooling
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        
        # Fully connected layer
        x = self.fc(x)
        return x
    
    def make_layers(self, inchannel, outchannel, stride, blocks):
        # Define a block with correct channel dimensions
        layers = [Block(inchannel, outchannel, stride)]
        for _ in range(1, blocks):
            layers.append(Block(outchannel, outchannel, 1))  # Keep stride=1 for subsequent blocks
        return nn.Sequential(*layers)
