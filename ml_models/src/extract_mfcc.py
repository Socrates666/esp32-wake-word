import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

def pad_audio(audio, target_length, add_noise_to_pad=True, noise_level=0.005):
    """
    填充音频到目标长度
    """
    current_length = audio.shape[1]
    if current_length < target_length:
        pad_length = target_length - current_length
        
        if add_noise_to_pad:
            pad_noise = torch.randn(audio.shape[0], pad_length) * noise_level
            audio = torch.cat([audio, pad_noise], dim=1)
        else:
            audio = torch.nn.functional.pad(audio, (0, pad_length))
    elif current_length > target_length:
        audio = audio[:, :target_length]
    
    return audio

def add_random_noise(waveform, noise_level=0.01, snr_range=(5, 20)):
    """
    为音频添加随机噪声
    """
    noise = torch.randn_like(waveform) * noise_level
    
    min_snr, max_snr = snr_range
    snr_db = torch.rand(1) * (max_snr - min_snr) + min_snr
    snr = 10 ** (snr_db / 20)
    
    signal_power = torch.mean(waveform ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power > 0:
        scale_factor = torch.sqrt(signal_power / (noise_power * snr))
        noise = noise * scale_factor
    
    noisy_audio = waveform + noise
    noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
    
    return noisy_audio

def normalize_mfcc(mfcc, method='standardization'):
    """
    对MFCC特征进行归一化
    Args:
        mfcc: MFCC特征 tensor (n_mfcc, time_steps)
        method: 归一化方法
                'standardization': Z-score标准化 (推荐)
                'minmax': min-max归一化
                'cmvn': 倒谱均值方差归一化 (更好)
    Returns:
        normalized_mfcc: 归一化后的MFCC
    """
    if method == 'standardization':
        # Z-score标准化: (x - mean) / std
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        # 防止除以0
        std = torch.where(std == 0, torch.ones_like(std), std)
        normalized = (mfcc - mean) / (std + 1e-8)
        
    elif method == 'minmax':
        # Min-Max归一化: (x - min) / (max - min)
        min_val = mfcc.min(dim=1, keepdim=True)[0]
        max_val = mfcc.max(dim=1, keepdim=True)[0]
        normalized = (mfcc - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'cmvn':
        # 倒谱均值方差归一化 (Cepstral Mean and Variance Normalization)
        # 对每一帧的所有MFCC系数进行归一化
        mean = mfcc.mean(dim=1, keepdim=True)  # (n_mfcc, 1)
        std = mfcc.std(dim=1, keepdim=True)     # (n_mfcc, 1)
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        normalized = (mfcc - mean) / (std + 1e-8)
        
        # 对时间维度也做一个平滑
        # 这样可以捕捉MFCC在时间上的变化模式
        
    else:
        normalized = mfcc
    
    return normalized

def augment_audio_waveform(audio, augment_factor=3):
    """
    在音频波形级别进行数据增强（不依赖sox）
    Args:
        audio: 原始音频波形 (1, samples)
        augment_factor: 增强倍数
    Returns:
        augmented_audios: 增强后的音频列表
    """
    augmented_audios = [audio]  # 保留原始音频
    
    # 语速变化（使用线性插值代替sox）
    speeds = [0.8, 1.2]
    for speed in speeds:
        target_length = int(audio.shape[1] * speed)
        augmented = torch.nn.functional.interpolate(
            audio.unsqueeze(0), 
            size=target_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        augmented = pad_audio(augmented, 16000)
        augmented_audios.append(augmented)
    
    # 音量变化
    volumes = [0.7, 1.3]
    for volume in volumes:
        augmented = audio * volume
        augmented = torch.clamp(augmented, -1.0, 1.0)
        augmented_audios.append(augmented)
    
    return augmented_audios

def extract_features(audio_path='./audio_data/train_data/xiaoa', label=0, is_noise=False, 
                    add_noise_to_pad=True, augment_audio=True, normalize_method='cmvn'):
    """
    提取MFCC特征，支持音频级别的数据增强和MFCC归一化
    Args:
        audio_path: 音频文件路径
        label: 标签
        is_noise: 是否添加背景噪声
        add_noise_to_pad: 是否在填充部分添加噪声
        augment_audio: 是否进行音频增强
        normalize_method: MFCC归一化方法 ('cmvn' 或 'standardization')
    """
    features = []
    labels = []
    mfcc_transform = T.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        log_mels=True,
        melkwargs={
            'n_fft': 512,
            'win_length': 320,
            'hop_length': 256,
            'n_mels': 40,
            'window_fn': torch.hamming_window
        }
    )
    
    file_count = 0
    for file in os.listdir(audio_path):
        if file.endswith('.wav'):
            file_count += 1
            audio, sr = torchaudio.load(os.path.join(audio_path, file))
            
            # 填充时添加噪声到填充部分
            audio = pad_audio(audio, 16000, add_noise_to_pad=add_noise_to_pad, noise_level=0.005)
            
            # 音频级别的数据增强
            if augment_audio:
                augmented_audios = augment_audio_waveform(audio, augment_factor=3)
            else:
                augmented_audios = [audio]
            
            # 对每个增强版本提取MFCC
            for aug_audio in augmented_audios:
                # 如果需要，添加背景噪声增强
                if is_noise:
                    aug_audio = add_random_noise(aug_audio, noise_level=0.01)
                
                preemphasized = torchaudio.functional.preemphasis(aug_audio, coeff=0.97)
                mfcc = mfcc_transform(preemphasized)[0]
                
                # ==================== 关键：MFCC归一化 ====================
                mfcc = normalize_mfcc(mfcc, method=normalize_method)
                
                features.append(mfcc)
                labels.append(torch.tensor(label))
    
    print(f"✅ 从 {file_count} 个文件提取了 {len(features)} 个MFCC特征 (归一化方法: {normalize_method})")
    
    return features, labels