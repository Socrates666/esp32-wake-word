import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import requests
import tarfile
import librosa
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """配置参数类"""
    # 数据参数
    data_dir: str = "./thchs30_data"
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_length: float = 8.0  # 最大音频长度（秒）
    
    # 模型参数
    hidden_size: int = 128  # 减小规模以便快速训练
    gru_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    
    # 训练参数
    batch_size: int = 8  # 减小批次大小
    learning_rate: float = 0.001
    num_epochs: int = 30
    early_stopping_patience: int = 5

class THCHS30Dataset(Dataset):
    """THCHS-30数据集"""
    
    def __init__(self, config: Config, audio_files: List[str], transcripts: List[str], vocab: Dict[str, int], is_train: bool = True):
        self.config = config
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.vocab = vocab
        self.is_train = is_train
        
        print(f"{'训练' if is_train else '验证'}数据集大小: {len(self.audio_files)}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        transcript = self.transcripts[idx]
        
        try:
            # 加载音频
            waveform, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # 提取特征
            features = self.extract_features(waveform)
            
            # 文本转索引
            text_indices = self.text_to_indices(transcript)
            
            return features, text_indices, audio_path
        
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            # 返回空数据
            empty_features = torch.zeros(10, self.config.n_mels)
            empty_text = torch.tensor([self.vocab["<unk>"]], dtype=torch.long)
            return empty_features, empty_text, "error"
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """提取Mel特征"""
        # 截断或填充音频
        max_samples = int(self.config.max_audio_length * self.config.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Mel频谱
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-8)
        
        # 标准化
        if mel.std() > 0:
            mel = (mel - mel.mean()) / mel.std()
        
        return mel.squeeze(0).transpose(0, 1)
    
    def text_to_indices(self, text: str) -> torch.Tensor:
        """文本转索引"""
        indices = []
        for char in text.strip():
            if char in self.vocab:
                indices.append(self.vocab[char])
            else:
                indices.append(self.vocab["<unk>"])
        return torch.tensor(indices, dtype=torch.long)

class GRU_CTC_Model(nn.Module):
    """GRU + CTC 语音识别模型"""
    
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.n_mels, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # GRU层
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # 输出层
        gru_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.output_layer = nn.Linear(gru_output_size, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.audio_encoder(x)
        gru_out, _ = self.gru(x)
        logits = self.output_layer(gru_out)
        return torch.nn.functional.log_softmax(logits, dim=-1)

class THCHS30Trainer:
    """THCHS-30训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.vocab = None
        self.idx_to_char = None
        self.model = None
        
    def download_thchs30(self) -> bool:
        """下载THCHS-30数据集"""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(exist_ok=True)
        
        tar_path = data_dir / "data_thchs30.tgz"
        extracted_path = data_dir / "data_thchs30"
        print(extracted_path)
        # 检查是否已经下载
        if 1:
            print("THCHS-30数据集已存在")
            return True
        
        # 下载数据集
        url = "http://www.openslr.org/resources/18/data_thchs30.tgz"
        
        try:
            print("下载THCHS-30数据集...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print("解压数据集...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            
            # 重命名解压后的文件夹
            if (data_dir / "data").exists():
                (data_dir / "data").rename(extracted_path)
            
            print("数据集下载完成!")
            return True
            
        except Exception as e:
            print(f"下载数据集失败: {e}")
            print("请手动下载: http://www.openslr.org/resources/18/data_thchs30.tgz")
            print(f"并解压到: {self.config.data_dir}")
            return False
    
    def load_thchs30_data(self) -> Tuple[List[str], List[str]]:
        """加载THCHS-30数据 - 修正版本"""
        data_dir = Path(self.config.data_dir) / "data_thchs30"
        audio_files = []
        transcripts = []
        
        # THCHS-30的实际文件结构
        scp_files = [
            data_dir / "train" / ".wav.scp",
            data_dir / "dev" / ".wav.scp", 
            data_dir / "test" / ".wav.scp"
        ]
        
        trn_files = [
            data_dir / "train" / "word.trn",
            data_dir / "dev" / "word.trn",
            data_dir / "test" / "word.trn"
        ]
        
        # 首先构建音频路径映射
        audio_path_map = {}
        for scp_file in scp_files:
            if scp_file.exists():
                with open(scp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            audio_id = parts[0]
                            audio_path = parts[1]
                            audio_path_map[audio_id] = audio_path
        
        # 然后加载转录文本
        for trn_file in trn_files:
            if trn_file.exists():
                with open(trn_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # .trn文件格式：每3行一组（拼音、汉字、音素）
                    for i in range(0, len(lines), 3):
                        if i + 1 < len(lines):
                            audio_id = lines[i].strip().split()[0] if i == 0 else lines[i].strip()
                            chinese_text = lines[i + 1].strip()  # 第二行是汉字文本
                            
                            if audio_id in audio_path_map:
                                audio_path = audio_path_map[audio_id]
                                if Path(audio_path).exists():
                                    audio_files.append(audio_path)
                                    transcripts.append(chinese_text)
        
        print(f"成功加载 {len(audio_files)} 个音频文件")
        return audio_files, transcripts
    
    def build_vocab(self, transcripts: List[str]) -> Dict[str, int]:
        """构建词汇表"""
        char_counter = Counter()
        for text in transcripts:
            char_counter.update(text.strip())
        
        # 创建词汇表
        vocab = {"<blank>": 0, "<unk>": 1}
        
        # 添加所有出现的字符
        for char, count in char_counter.most_common():
            if char not in vocab and char.strip():  # 跳过空格字符
                vocab[char] = len(vocab)
        
        print(f"词汇表大小: {len(vocab)}")
        print(f"前20个字符: {list(vocab.keys())[:20]}")
        
        return vocab
    
    def collate_fn(self, batch):
        """整理批次数据"""
        features, labels, audio_paths = zip(*batch)
        
        # 过滤无效数据
        valid_indices = [i for i, feat in enumerate(features) if feat.shape[0] > 1]
        if len(valid_indices) == 0:
            # 返回最小批次
            empty_feat = torch.zeros(10, self.config.n_mels)
            empty_label = torch.tensor([1], dtype=torch.long)
            return empty_feat.unsqueeze(0), empty_label.unsqueeze(0), torch.tensor([10]), torch.tensor([1])
        
        features = [features[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        # 填充特征
        feature_lengths = [feat.shape[0] for feat in features]
        max_feature_len = max(feature_lengths)
        padded_features = torch.zeros(len(features), max_feature_len, self.config.n_mels)
        
        for i, feat in enumerate(features):
            padded_features[i, :len(feat)] = feat
        
        # 填充标签
        label_lengths = [len(label) for label in labels]
        max_label_len = max(label_lengths)
        padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
        
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        return (
            padded_features,
            padded_labels,
            torch.tensor(feature_lengths, dtype=torch.long),
            torch.tensor(label_lengths, dtype=torch.long)
        )
    
    def train(self):
        """训练模型"""
        print("=== THCHS-30 中文语音识别训练 ===")
        
        # 下载数据
        if not self.download_thchs30():
            print("无法下载数据集，退出训练")
            return
        
        # 加载数据
        audio_files, transcripts = self.load_thchs30_data()
        
        if len(audio_files) == 0:
            print("没有找到数据，退出训练")
            return
        
        # 构建词汇表
        self.vocab = self.build_vocab(transcripts)
        self.idx_to_char = {idx: char for char, idx in self.vocab.items()}
        
        # 分割数据集 (80% 训练, 20% 验证)
        split_idx = int(0.8 * len(audio_files))
        train_audio = audio_files[:split_idx]
        train_text = transcripts[:split_idx]
        val_audio = audio_files[split_idx:]
        val_text = transcripts[split_idx:]
        
        # 创建数据集
        train_dataset = THCHS30Dataset(self.config, train_audio, train_text, self.vocab, is_train=True)
        val_dataset = THCHS30Dataset(self.config, val_audio, val_text, self.vocab, is_train=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # 初始化模型
        self.model = GRU_CTC_Model(self.config, len(self.vocab))
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CTCLoss(blank=0)
        
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("开始训练...")
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.num_epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                features, labels, feature_lengths, label_lengths = batch
                
                features = features.to(self.device)
                labels = labels.to(self.device)
                feature_lengths = feature_lengths.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                # 前向传播
                log_probs = self.model(features)
                log_probs = log_probs.transpose(0, 1)  # (seq_len, batch, num_classes)
                
                loss = criterion(log_probs, labels, feature_lengths, label_lengths)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 验证
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    features, labels, feature_lengths, label_lengths = batch
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    feature_lengths = feature_lengths.to(self.device)
                    label_lengths = label_lengths.to(self.device)
                    
                    log_probs = self.model(features)
                    log_probs = log_probs.transpose(0, 1)
                    
                    loss = criterion(log_probs, labels, feature_lengths, label_lengths)
                    val_loss += loss.item()
            
            # 计算平均损失
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1:02d}/{self.config.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("thchs30_best_model.pth")
                print("保存最佳模型!")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                print("早停触发!")
                break
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses)
        print("训练完成!")
    
    def decode_predictions(self, log_probs: torch.Tensor) -> List[str]:
        """解码预测结果"""
        _, predictions = torch.max(log_probs, dim=2)
        decoded_texts = []
        
        for batch_idx in range(predictions.shape[0]):
            pred_sequence = []
            prev_token = 0  # blank_idx
            
            for token in predictions[batch_idx]:
                token = token.item()
                if token != 0 and token != prev_token:
                    pred_sequence.append(token)
                prev_token = token
            
            text = "".join([self.idx_to_char.get(idx, "<unk>") for idx in pred_sequence])
            decoded_texts.append(text)
        
        return decoded_texts
    
    def predict_audio(self, audio_path: str) -> str:
        """预测单个音频"""
        if self.model is None:
            if not self.load_model("thchs30_best_model.pth"):
                print("请先训练模型!")
                return ""
        
        self.model.eval()
        
        try:
            # 加载和预处理音频
            waveform, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # 提取特征
            dataset = THCHS30Dataset(self.config, [], [], self.vocab)
            features = dataset.extract_features(waveform)
            features = features.unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                log_probs = self.model(features)
                predictions = self.decode_predictions(log_probs)
                
            return predictions[0] if predictions else "预测失败"
            
        except Exception as e:
            print(f"预测时出错: {e}")
            return "预测失败"
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'idx_to_char': self.idx_to_char,
            'config': self.config,
        }, path)
    
    def load_model(self, path: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.config = checkpoint['config']
            self.vocab = checkpoint['vocab']
            self.idx_to_char = checkpoint['idx_to_char']
            
            self.model = GRU_CTC_Model(self.config, len(self.vocab))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(val_losses, label='验证损失', linewidth=2)
        plt.title('THCHS-30 训练曲线')
        plt.xlabel('Epoch')
        plt.ylabel('CTC Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('thchs30_training.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    config = Config()
    trainer = THCHS30Trainer(config)
    
    # 训练模型
    trainer.train()
    
    # 演示预测
    print("\n=== 演示预测 ===")
    
    # 从验证集选择一个音频进行预测
    data_dir = Path(config.data_dir) / "data_thchs30"
    test_audio_files = list(data_dir.glob("**/*.wav"))
    
    if test_audio_files:
        test_audio = test_audio_files[0]  # 选择第一个音频
        print(f"测试音频: {test_audio.name}")
        
        prediction = trainer.predict_audio(str(test_audio))
        print(f"预测结果: {prediction}")
        
        # 显示真实文本（如果可用）
        audio_id = test_audio.stem
        for folder in ["train", "dev", "test"]:
            txt_file = data_dir / folder / "word.txt"
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith(audio_id):
                            true_text = ' '.join(line.strip().split()[1:])
                            print(f"真实文本: {true_text}")
                            break

if __name__ == "__main__":
    main()