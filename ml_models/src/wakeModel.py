import torch
import torch.nn as nn

class LightweightKWS(nn.Module):
    def __init__(self, num_classes=3):
        super(LightweightKWS, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(13, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64 ,bias=False),
            nn.ReLU(),
            nn.Linear(64, num_classes, bias=False)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1) 
        x = self.classifier(x)  # 直接进入分类器
        return x

class SimpleEffectiveKWS(nn.Module):
    def __init__(self, num_classes=1, input_channels=13):
        super(SimpleEffectiveKWS, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            # Block 3 - 减少时间维度
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # 分类器 - 输出[batch, 1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, kernel_size=1, bias=False)  # 输出通道为1
        )

    def forward(self, x):
        x = self.features(x)  # [batch, 128, 1]
        x = self.classifier(x)  # [batch, 1, 1]
        x = x.squeeze(-1)  # [batch, 1]
        return x
    
"""滑动窗口处理 - 保持时间维度"""
class SlidingWindowKWS(nn.Module):
    def __init__(self, num_classes=2, window_size=100):
        super().__init__()
        self.window_size = window_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(13, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # 计算卷积后的时间维度
        self.time_reduction = 4  # 两次池化，每次/2
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * (window_size // self.time_reduction), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 13, window_size)
        batch_size = x.shape[0]
        
        x = self.conv_layers(x)                    # (batch, 64, window_size/4)
        x = x.view(batch_size, -1)                 # 展平时间维度
        x = self.classifier(x)
        return x

class FrameBasedStreamingKWS(nn.Module):  
    def __init__(self, num_classes=3):
        super().__init__()
        
        # 处理单帧的轻量网络
        self.frame_net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 13, 1) 或 (batch, 13, T)
        if x.dim() == 3 and x.shape[-1] > 1:
            # 多帧输入，逐帧处理
            batch_size, features, time_steps = x.shape
            x = x.transpose(1, 2)  # (batch, time_steps, features)
            x = x.reshape(-1, features)  # (batch*time_steps, features)
            x = self.frame_net(x)
            x = x.view(batch_size, time_steps, -1)  # (batch, time_steps, num_classes)
            return x
        else:
            # 单帧输入
            if x.shape[-1] == 1:
                x = x.squeeze(-1)  # (batch, 13)
            return self.frame_net(x)

class StreamableGRU_KWS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(StreamableGRU_KWS, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 核心GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 分类器
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid() # 二分类，关键词 vs 非关键词

    def forward(self, x, h_prev=None):
        # x: (batch_size, sequence_length, input_size)
        # h_prev: (num_layers, batch_size, hidden_size)，初始隐藏状态
        
        out, h_next = self.gru(x, h_prev) # out: (batch, seq_len, hidden_size)
        # 我们通常只取最后一个时间步的输出做分类
        # 但对于流式，我们其实关心每一个时间步
        out = self.fc(out[:, -1, :]) # 这里取最后一个时间步，用于训练
        out = self.sigmoid(out)
        return out, h_next

    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
