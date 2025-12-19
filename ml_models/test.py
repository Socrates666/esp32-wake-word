import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import librosa

class CTCKeywordSpotter(nn.Module):
    """基于CTC的关键词识别模型"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(CTCKeywordSpotter, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Encoder: GRU + Linear
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, bidirectional=True, dropout=dropout)
        
        # 双向GRU输出维度是 2 * hidden_size
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
        
        # Softmax for CTC (输出每个时间步的字符概率)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, x, input_lengths=None):
        # x: (batch_size, seq_len, input_size)
        
        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, 2*hidden_size)
        
        # Classification
        logits = self.classifier(gru_out)  # (batch_size, seq_len, num_classes)
        
        # Log softmax for CTC loss
        log_probs = self.softmax(logits)  # (batch_size, seq_len, num_classes)
        
        return log_probs

class CTCDataset(Dataset):
    """CTC训练数据集"""
    def __init__(self, audio_files, transcripts, char_to_idx, sample_rate=16000):
        self.audio_files = audio_files
        self.transcripts = transcripts  # 文本标签，如 ["xiaoa", "other", ...]
        self.char_to_idx = char_to_idx
        self.sample_rate = sample_rate
        self.blank_idx = 0  # CTC空白符号索引
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 加载音频并提取特征
        waveform, sr = torchaudio.load(self.audio_files[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # 提取MFCC特征
        features = self.extract_mfcc_features(waveform.squeeze())  # (seq_len, n_mfcc)
        
        # 将文本标签转换为索引序列
        transcript = self.transcripts[idx]
        target = torch.tensor([self.char_to_idx[char] for char in transcript], dtype=torch.long)
        
        # 输入长度 (序列长度)
        input_length = torch.tensor([features.shape[0]], dtype=torch.long)
        
        # 目标长度 (标签长度)
        target_length = torch.tensor([len(target)], dtype=torch.long)
        
        return features, input_length, target, target_length
    
    def extract_mfcc_features(self, waveform, n_mfcc=13):
        """提取MFCC特征"""
        waveform_np = waveform.numpy()
        mfcc_features = librosa.feature.mfcc(
            y=waveform_np,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=256,
            n_mels=40
        )
        return torch.FloatTensor(mfcc_features.T)  # (seq_len, n_mfcc)

def train_ctc_model(model, train_loader, val_loader, char_to_idx, num_epochs=10, device='cpu'):
    """CTC模型训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=char_to_idx['_'], zero_infinity=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        for batch_idx, (features, input_lengths, targets, target_lengths) in enumerate(train_loader):
            features = features.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            log_probs = model(features)  # (batch, seq_len, num_classes)
            
            # CTC损失计算
            log_probs = log_probs.transpose(0, 1)  # CTC要求 (seq_len, batch, num_classes)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        val_loss = evaluate_ctc_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss == min(val_losses):
            torch.save(model.state_dict(), 'best_ctc_model.pth')
    
    return train_losses, val_losses

def evaluate_ctc_model(model, data_loader, criterion, device):
    """评估CTC模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, input_lengths, targets, target_lengths in data_loader:
            features = features.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            log_probs = model(features)
            log_probs = log_probs.transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


class CTCKeywordDetector:
    """基于CTC的关键词检测器"""
    def __init__(self, model, char_to_idx, keywords, threshold=0.8):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        self.keywords = keywords  # 要检测的关键词列表，如 ['xiaoa']
        self.threshold = threshold
        
    def detect_keywords(self, audio_stream):
        """流式关键词检测"""
        self.model.eval()
        buffer = []
        detected_keywords = []
        
        for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            # 当缓冲区足够长时进行处理
            if len(buffer):  # 例如10个chunk
                # 提取特征
                features = self.extract_mfcc_features(buffer)
                
                # 模型预测
                with torch.no_grad():
                    log_probs = self.model(features.unsqueeze(0))  # (1, seq_len, num_classes)
                
                # CTC解码
                decoded_text = self.ctc_greedy_decode(log_probs.squeeze(0), self.idx_to_char)
                print(f"log_probs's shape: {log_probs.shape}")
                print(f"decoded_text's shape: {len(decoded_text)}")
                # 检测关键词
                for keyword in self.keywords:
                    if keyword in decoded_text:
                        confidence = self.calculate_confidence(decoded_text, keyword, log_probs)
                        if confidence > self.threshold:
                            detected_keywords.append((keyword, confidence))
                            print(f"检测到关键词: {keyword}, 置信度: {confidence:.3f}")
                
                # 清空缓冲区
                buffer = buffer[5:]  # 滑动窗口
        
        return detected_keywords
    def ctc_greedy_decode(self, log_probs, char_list):
        """CTC贪心解码"""
        # log_probs: (seq_len, num_classes)
        if len(log_probs.shape) == 3: 
            log_probs = log_probs.squeeze(0)  # 取第一个样本
        print(log_probs.shape)
        _, max_indices = torch.max(log_probs, dim=1)
        
        # 移除重复字符和空白符
        decoded = []
        prev_char = -1
        for idx in max_indices.tolist():
            if idx != prev_char and idx != 0:  # 0是空白符
                decoded.append(char_list[idx])
            #prev_char = idx
        
        return ''.join(decoded)
    def extract_mfcc_features(self, waveform, n_mfcc=13):
        """提取MFCC特征"""
        waveform_np = waveform[0].numpy()
        mfcc_features = librosa.feature.mfcc(
            y=waveform_np,
            sr=16000,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=256,
            n_mels=40
        )
        return torch.FloatTensor(mfcc_features.T)  # (seq_len, n_mfcc)
    def calculate_confidence(self, decoded_text, keyword, log_probs):
        """计算关键词检测置信度"""
        # 简化的置信度计算
        if keyword in decoded_text:
            return 0.9  # 可以根据实际概率调整
        return 0.0
import os
def main():
    # 字符映射表
    char_to_idx = {
        '_': 0,  # CTC空白符
        'k': 1,
        'n': 2,
        #'x': 1, 'i': 2, 'a': 3, 'o': 4,  # 小啊的拼音

    }
    
    # 准备数据
    # 假设我们有音频文件列表和对应的文本标签
    audio_files = ["./audio_data/xiaoa/" + p for p in os.listdir("./audio_data/xiaoa")]
    audio_files += ["./audio_data/others/" + p for p in os.listdir("./audio_data/others")]
    transcripts = ["k"] * len(os.listdir("./audio_data/xiaoa")) + ["n"] * len(os.listdir("./audio_data/others"))
    
    # 创建数据集
    dataset = CTCDataset(audio_files, transcripts, char_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=ctc_collate_fn)
    
    # 创建模型
    model = CTCKeywordSpotter(
        input_size=13,  # MFCC特征维度
        hidden_size=128,
        num_layers=2,
        num_classes=len(char_to_idx)
    ).to('cpu')
    
    # 训练
    train_losses, val_losses = train_ctc_model(model, train_loader, val_loader, char_to_idx, num_epochs=5)
    
    # 创建检测器
    detector = CTCKeywordDetector(model, char_to_idx, keywords=['xiaoa'])
    audio_stream = [torchaudio.load(s)[0] for s in audio_files]
    detector.detect_keywords(audio_stream[0])
    print("CTC关键词识别模型训练完成！")

def ctc_collate_fn(batch):
    """CTC数据collate函数"""
    features, input_lengths, targets, target_lengths = zip(*batch)
    
    # 填充特征序列
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    # 填充目标序列
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    
    input_lengths = torch.cat(input_lengths)
    target_lengths = torch.cat(target_lengths)
    
    return features_padded, input_lengths, targets_padded, target_lengths

#if __name__ == "__main__":
#   main()
# 字符映射表


def extract_mfcc_features(waveform, n_mfcc=13):
    """提取MFCC特征"""
    waveform_np = waveform.numpy()
    mfcc_features = librosa.feature.mfcc(
        y=waveform_np,
        sr=16000,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=256,
        n_mels=40
    )
    return torch.FloatTensor(mfcc_features.T)  # (seq_len, n_mfcc)
char_to_idx = {
    '_': 1,  # CTC空白符
    'k': 2,
    #'x': 1, 'i': 2, 'a': 3, 'o': 4,  # 小啊的拼音
    'n': 0,
}

# 准备数据
# 假设我们有音频文件列表和对应的文本标签
audio_files = ["./audio_data/xiaoa/" + p for p in os.listdir("./audio_data/xiaoa")]
audio_files += ["./audio_data/others/" + p for p in os.listdir("./audio_data/others")]
transcripts = ["k"] * len(os.listdir("./audio_data/xiaoa")) + ["n"] * len(os.listdir("./audio_data/others"))

# 创建数据集
dataset = CTCDataset(audio_files, transcripts, char_to_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=ctc_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=ctc_collate_fn)

# 创建模型
model = CTCKeywordSpotter(
    input_size=13,  # MFCC特征维度
    hidden_size=128,
    num_layers=2,
    num_classes=len(char_to_idx)
).to('cpu')

# 训练
train_losses, val_losses = train_ctc_model(model, train_loader, val_loader, char_to_idx, num_epochs=20)

# 创建检测器
detector = CTCKeywordDetector(model, char_to_idx, keywords=['xiaoa'])
audio_stream = [torchaudio.load(s)[0] for s in audio_files]

for s in audio_stream:
    d = detector.detect_keywords(s)
    if d.count != 0:
        print(d)