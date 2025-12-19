#!/usr/bin/env python3
"""
生成随机噪声音频的 PyTorch 脚本
支持白噪声、粉红噪声、棕色噪声等
"""

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
from datetime import datetime


class AudioNoiseGenerator:
    """使用 PyTorch 生成各种噪声音频"""
    
    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        """
        初始化音频生成器
        
        Args:
            sample_rate: 采样率 (Hz)
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.sample_rate = sample_rate
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # 支持的噪声类型
        self.noise_types = {
            'white': self.generate_white_noise,
            'pink': self.generate_pink_noise,
            'brown': self.generate_brown_noise,
            'blue': self.generate_blue_noise,
            'violet': self.generate_violet_noise,
            'grey': self.generate_grey_noise,
            'gaussian': self.generate_gaussian_noise,
            'uniform': self.generate_uniform_noise,
            'impulse': self.generate_impulse_noise,
            'periodic': self.generate_periodic_noise,
            'mixed': self.generate_mixed_noise
        }
        
    def generate_white_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成白噪声（所有频率能量相等）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        noise = torch.randn(num_samples, device=self.device)
        return amplitude * noise / torch.max(torch.abs(noise))
    
    def generate_pink_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成粉红噪声（频率与能量成反比）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        
        # 生成白噪声
        white_noise = torch.randn(num_samples, device=self.device)
        
        # 应用粉红噪声滤波器（1/f 滤波器）
        # 使用快速傅里叶变换
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # 避免除零
        frequencies[0] = 1
        
        # 1/f 滤波器
        pink_filter = 1 / torch.sqrt(frequencies)
        pink_filter = pink_filter / torch.max(pink_filter)
        
        # 应用滤波器
        pink_fft = fft * pink_filter
        pink_noise = torch.fft.irfft(pink_fft, n=num_samples)
        
        # 归一化
        pink_noise = amplitude * pink_noise / torch.max(torch.abs(pink_noise))
        return pink_noise
    
    def generate_brown_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成棕色噪声（布朗噪声，1/f²）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        
        # 生成白噪声
        white_noise = torch.randn(num_samples, device=self.device)
        
        # 应用棕色噪声滤波器（1/f²）
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # 避免除零
        frequencies[0] = 1
        
        # 1/f² 滤波器
        brown_filter = 1 / frequencies
        brown_filter = brown_filter / torch.max(brown_filter)
        
        # 应用滤波器
        brown_fft = fft * brown_filter
        brown_noise = torch.fft.irfft(brown_fft, n=num_samples)
        
        # 归一化
        brown_noise = amplitude * brown_noise / torch.max(torch.abs(brown_noise))
        return brown_noise
    
    def generate_blue_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成蓝色噪声（频率与能量成正比）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        
        # 生成白噪声
        white_noise = torch.randn(num_samples, device=self.device)
        
        # 应用蓝色噪声滤波器（f 滤波器）
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # 蓝色噪声滤波器（f）
        blue_filter = frequencies
        blue_filter = blue_filter / torch.max(blue_filter)
        
        # 应用滤波器
        blue_fft = fft * blue_filter
        blue_noise = torch.fft.irfft(blue_fft, n=num_samples)
        
        # 归一化
        blue_noise = amplitude * blue_noise / torch.max(torch.abs(blue_noise))
        return blue_noise
    
    def generate_violet_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成紫色噪声（f²）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        
        # 生成白噪声
        white_noise = torch.randn(num_samples, device=self.device)
        
        # 应用紫色噪声滤波器（f²）
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # 紫色噪声滤波器（f²）
        violet_filter = frequencies ** 2
        violet_filter = violet_filter / torch.max(violet_filter)
        
        # 应用滤波器
        violet_fft = fft * violet_filter
        violet_noise = torch.fft.irfft(violet_fft, n=num_samples)
        
        # 归一化
        violet_noise = amplitude * violet_noise / torch.max(torch.abs(violet_noise))
        return violet_noise
    
    def generate_grey_noise(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """
        生成灰色噪声（等响度曲线补偿）
        
        Args:
            duration: 音频时长（秒）
            amplitude: 振幅 (0-1)
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        
        # 生成白噪声
        white_noise = torch.randn(num_samples, device=self.device)
        
        # 等响度曲线滤波器（简化版本）
        fft = torch.fft.rfft(white_noise)
        frequencies = torch.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # 等响度曲线滤波器（A-weighting 近似）
        # 在1kHz时为0dB，在其他频率有衰减
        f1 = 20.6  # Hz
        f2 = 107.7  # Hz
        f3 = 737.9  # Hz
        f4 = 12194.0  # Hz
        
        # A-weighting 滤波器响应
        numerator = (frequencies**2) * (frequencies**2 + f4**2)
        denominator = (frequencies**2 + f1**2) * torch.sqrt(frequencies**2 + f2**2) * torch.sqrt(frequencies**2 + f3**2) * (frequencies**2 + f4**2)
        a_weighting = 1.2589 * numerator / denominator
        
        # 归一化
        a_weighting = a_weighting / torch.max(a_weighting)
        
        # 应用滤波器
        grey_fft = fft * a_weighting
        grey_noise = torch.fft.irfft(grey_fft, n=num_samples)
        
        # 归一化
        grey_noise = amplitude * grey_noise / torch.max(torch.abs(grey_noise))
        return grey_noise
    
    def generate_gaussian_noise(self, duration: float, mean: float = 0.0, std: float = 0.3) -> torch.Tensor:
        """
        生成高斯噪声
        
        Args:
            duration: 音频时长（秒）
            mean: 均值
            std: 标准差
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        noise = torch.normal(mean=mean, std=std, size=(num_samples,), device=self.device)
        return noise / torch.max(torch.abs(noise))
    
    def generate_uniform_noise(self, duration: float, low: float = -0.5, high: float = 0.5) -> torch.Tensor:
        """
        生成均匀分布噪声
        
        Args:
            duration: 音频时长（秒）
            low: 下界
            high: 上界
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        noise = torch.rand(num_samples, device=self.device) * (high - low) + low
        return noise
    
    def generate_impulse_noise(self, duration: float, probability: float = 0.01, amplitude: float = 1.0) -> torch.Tensor:
        """
        生成脉冲噪声
        
        Args:
            duration: 音频时长（秒）
            probability: 脉冲概率
            amplitude: 脉冲振幅
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        noise = torch.zeros(num_samples, device=self.device)
        
        # 生成随机脉冲位置
        impulse_mask = torch.rand(num_samples, device=self.device) < probability
        num_impulses = impulse_mask.sum().item()
        
        if num_impulses > 0:
            # 随机脉冲幅度和极性
            impulse_values = (torch.rand(num_impulses, device=self.device) * 2 - 1) * amplitude
            noise[impulse_mask] = impulse_values
        
        return noise
    
    def generate_periodic_noise(self, duration: float, frequency: float = 50.0, amplitude: float = 0.3) -> torch.Tensor:
        """
        生成周期性噪声（如电源干扰）
        
        Args:
            duration: 音频时长（秒）
            frequency: 噪声频率 (Hz)
            amplitude: 振幅
            
        Returns:
            torch.Tensor: 音频数据
        """
        num_samples = int(duration * self.sample_rate)
        t = torch.arange(num_samples, device=self.device) / self.sample_rate
        noise = amplitude * torch.sin(2 * torch.pi * frequency * t)
        return noise
    
    def generate_mixed_noise(self, duration: float) -> torch.Tensor:
        """
        生成混合噪声
        
        Args:
            duration: 音频时长（秒）
            
        Returns:
            torch.Tensor: 音频数据
        """
        # 组合多种噪声
        white = self.generate_white_noise(duration, 0.3)
        pink = self.generate_pink_noise(duration, 0.2)
        impulse = self.generate_impulse_noise(duration, 0.005, 0.5)
        periodic = self.generate_periodic_noise(duration, 60.0, 0.1)
        
        # 混合
        mixed = white + pink + impulse + periodic
        
        # 归一化防止削波
        mixed = mixed / torch.max(torch.abs(mixed)) * 0.9
        
        return mixed
    
    def generate_noise(self, noise_type: str, **kwargs) -> torch.Tensor:
        """
        生成指定类型的噪声
        
        Args:
            noise_type: 噪声类型
            **kwargs: 传递给噪声生成函数的参数
            
        Returns:
            torch.Tensor: 音频数据
        """
        if noise_type not in self.noise_types:
            raise ValueError(f"未知的噪声类型: {noise_type}. 可用类型: {list(self.noise_types.keys())}")
        
        return self.noise_types[noise_type](**kwargs)
    
    def save_audio(self, audio: torch.Tensor, filepath: str, 
                   format: str = "wav", bits_per_sample: int = 16):
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            filepath: 保存路径
            format: 音频格式
            bits_per_sample: 比特深度
        """
        # 确保音频是2D的 [channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # 转换为单声道
        
        # 保存音频
        torchaudio.save(
            filepath,
            audio.cpu(),
            self.sample_rate,
            bits_per_sample=bits_per_sample,
            format=format
        )
    
    def analyze_audio(self, audio: torch.Tensor) -> Dict:
        """
        分析音频统计信息
        
        Args:
            audio: 音频数据
            
        Returns:
            Dict: 统计信息
        """
        audio_np = audio.cpu().numpy()
        
        return {
            'duration': len(audio) / self.sample_rate,
            'samples': len(audio),
            'sample_rate': self.sample_rate,
            'max_amplitude': float(torch.max(torch.abs(audio))),
            'rms': float(torch.sqrt(torch.mean(audio**2))),
            'mean': float(torch.mean(audio)),
            'std': float(torch.std(audio)),
            'dynamic_range': float(20 * torch.log10(torch.max(torch.abs(audio)) / torch.std(audio))) if torch.std(audio) > 0 else 0
        }


def generate_noise_dataset(output_dir: str = "./noise_dataset",
                           num_samples: int = 100,
                           durations: List[float] = [1.0, 2.0, 5.0, 10.0],
                           noise_types: List[str] = None,
                           sample_rate: int = 16000,
                           use_gpu: bool = False,
                           save_metadata: bool = True,
                           visualize: bool = True):
    """
    生成噪声音频数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 总样本数
        durations: 音频时长列表（秒）
        noise_types: 噪声类型列表，如果为None则使用所有类型
        sample_rate: 采样率
        use_gpu: 是否使用GPU
        save_metadata: 是否保存元数据
        visualize: 是否生成可视化
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 音频文件子目录
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    if visualize:
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
    
    # 初始化生成器
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    generator = AudioNoiseGenerator(sample_rate=sample_rate, device=device)
    
    # 使用所有噪声类型（如果未指定）
    if noise_types is None:
        noise_types = list(generator.noise_types.keys())
    
    print(f"开始生成噪声音频数据集")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print(f"噪声类型: {noise_types}")
    print(f"采样率: {sample_rate} Hz")
    print(f"音频时长: {durations} 秒")
    print(f"总样本数: {num_samples}")
    print("-" * 50)
    
    # 存储元数据
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'sample_rate': sample_rate,
        'device': device,
        'num_samples': num_samples,
        'durations': durations,
        'noise_types': noise_types,
        'samples': []
    }
    
    # 生成音频样本
    for i in range(num_samples):
        # 随机选择参数
        noise_type = np.random.choice(noise_types)
        duration = np.random.choice(durations)
        amplitude = np.random.uniform(0.1, 0.8)
        
        # 根据噪声类型调整参数
        params = {'duration': duration, 'amplitude': amplitude}
        
        if noise_type == 'gaussian':
            params['mean'] = np.random.uniform(-0.2, 0.2)
            params['std'] = np.random.uniform(0.1, 0.5)
        elif noise_type == 'uniform':
            params['low'] = np.random.uniform(-0.8, -0.2)
            params['high'] = np.random.uniform(0.2, 0.8)
        elif noise_type == 'impulse':
            params['probability'] = np.random.uniform(0.001, 0.02)
            params['amplitude'] = np.random.uniform(0.5, 1.0)
        elif noise_type == 'periodic':
            params['frequency'] = np.random.choice([50.0, 60.0, 100.0, 120.0, 150.0])
            params['amplitude'] = np.random.uniform(0.1, 0.4)
        
        # 生成噪声
        try:
            audio = generator.generate_noise(noise_type, **params)
            
            # 分析音频
            stats = generator.analyze_audio(audio)
            
            # 生成文件名
            filename = f"noise_{noise_type}_{i:04d}_{int(duration)}s.wav"
            filepath = audio_dir / filename
            
            # 保存音频
            generator.save_audio(audio, str(filepath))
            
            # 保存样本信息
            sample_info = {
                'id': i,
                'filename': filename,
                'noise_type': noise_type,
                'duration': duration,
                'parameters': params,
                'statistics': stats,
                'filepath': str(filepath.relative_to(output_path))
            }
            metadata['samples'].append(sample_info)
            
            print(f"生成 [{i+1:04d}/{num_samples}] {filename} - {noise_type} ({duration}s)")
            
            # 可视化（每10个样本生成一个）
            if visualize and i % 10 == 0:
                create_visualization(audio, noise_type, duration, sample_rate, viz_dir, i)
                
        except Exception as e:
            print(f"生成样本 {i} 时出错: {e}")
            continue
    
    # 保存元数据
    if save_metadata:
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存为 CSV 格式（方便分析）
        csv_file = output_path / "metadata.csv"
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'filename', 'noise_type', 'duration', 'max_amplitude', 'rms', 'mean', 'std', 'dynamic_range']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in metadata['samples']:
                writer.writerow({
                    'id': sample['id'],
                    'filename': sample['filename'],
                    'noise_type': sample['noise_type'],
                    'duration': sample['duration'],
                    'max_amplitude': sample['statistics']['max_amplitude'],
                    'rms': sample['statistics']['rms'],
                    'mean': sample['statistics']['mean'],
                    'std': sample['statistics']['std'],
                    'dynamic_range': sample['statistics']['dynamic_range']
                })
    
    # 生成汇总报告
    generate_summary_report(metadata, output_path)
    
    print("\n" + "=" * 50)
    print(f"数据集生成完成!")
    print(f"音频文件: {audio_dir}")
    print(f"元数据: {output_path}/metadata.json")
    print(f"总计: {len(metadata['samples'])} 个音频文件")
    print("=" * 50)


def create_visualization(audio: torch.Tensor, noise_type: str, duration: float,
                         sample_rate: int, output_dir: Path, sample_id: int):
    """
    创建音频可视化
    
    Args:
        audio: 音频数据
        noise_type: 噪声类型
        duration: 音频时长
        sample_rate: 采样率
        output_dir: 输出目录
        sample_id: 样本ID
    """
    try:
        audio_np = audio.cpu().numpy()
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 时域图
        time = np.arange(len(audio_np)) / sample_rate
        axes[0].plot(time, audio_np, alpha=0.7)
        axes[0].set_title(f"{noise_type.capitalize()} Noise - Sample {sample_id} ({duration}s)")
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 频谱图
        from scipy import signal
        f, Pxx = signal.welch(audio_np, sample_rate, nperseg=1024)
        axes[1].semilogx(f, 10 * np.log10(Pxx))
        axes[1].set_title('Power Spectral Density')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power/Frequency (dB/Hz)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        viz_file = output_dir / f"viz_{noise_type}_{sample_id:04d}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"可视化生成失败: {e}")


def generate_summary_report(metadata: Dict, output_dir: Path):
    """
    生成汇总报告
    
    Args:
        metadata: 元数据
        output_dir: 输出目录
    """
    try:
        import pandas as pd
        
        # 创建统计摘要
        df = pd.DataFrame(metadata['samples'])
        
        # 噪声类型统计
        type_counts = df['noise_type'].value_counts()
        
        # 时长统计
        duration_stats = df['duration'].describe()
        
        # 振幅统计
        max_amp_stats = pd.Series([s['max_amplitude'] for s in df['statistics']]).describe()
        rms_stats = pd.Series([s['rms'] for s in df['statistics']]).describe()
        
        # 生成报告
        report = f"""
        =========================================
        噪声音频数据集统计报告
        =========================================
        生成时间: {metadata['generated_at']}
        设备: {metadata['device']}
        采样率: {metadata['sample_rate']} Hz
        总样本数: {len(df)} / {metadata['num_samples']} 计划
        
        噪声类型分布:
        {type_counts.to_string()}
        
        音频时长统计 (秒):
        {duration_stats.to_string()}
        
        最大振幅统计:
        {max_amp_stats.to_string()}
        
        RMS 振幅统计:
        {rms_stats.to_string()}
        
        数据集包含以下文件:
        - 音频文件: {len(df)} 个 WAV 文件
        - 元数据: metadata.json (完整信息)
        - 元数据: metadata.csv (表格格式)
        - 可视化: 部分样本的时域/频域图
        
        =========================================
        """
        
        # 保存报告
        report_file = output_dir / "dataset_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        
    except Exception as e:
        print(f"生成报告失败: {e}")


def test_noise_generation():
    """测试各种噪声生成"""
    print("测试噪声生成...")
    
    generator = AudioNoiseGenerator(sample_rate=16000)
    
    # 测试每种噪声
    test_duration = 3.0
    output_dir = Path("./test_noises")
    output_dir.mkdir(exist_ok=True)
    
    for noise_type in generator.noise_types.keys():
        try:
            print(f"生成 {noise_type} 噪声...")
            audio = generator.generate_noise(noise_type, duration=test_duration, amplitude=0.5)
            
            # 保存
            filename = output_dir / f"test_{noise_type}.wav"
            generator.save_audio(audio, str(filename))
            
            # 分析
            stats = generator.analyze_audio(audio)
            print(f"  ✓ 时长: {stats['duration']:.1f}s, 最大振幅: {stats['max_amplitude']:.3f}, RMS: {stats['rms']:.3f}")
            
        except Exception as e:
            print(f"  ✗ {noise_type}: {e}")
    
    print(f"\n测试完成！文件保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 PyTorch 生成噪声音频数据集")
    parser.add_argument("--output-dir", type=str, default="./noise_dataset",
                       help="输出目录路径")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="生成的音频样本数量")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       choices=[8000, 16000, 22050, 44100, 48000],
                       help="音频采样率")
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 2.0, 5.0],
                       help="音频时长列表（秒）")
    parser.add_argument("--noise-types", type=str, nargs="+",
                       help="噪声类型列表（默认：全部）")
    parser.add_argument("--use-gpu", action="store_true",
                       help="使用 GPU 加速（如果可用）")
    parser.add_argument("--no-metadata", action="store_true",
                       help="不生成元数据文件")
    parser.add_argument("--no-visualize", action="store_true",
                       help="不生成可视化图像")
    parser.add_argument("--test", action="store_true",
                       help="运行测试模式，生成各种噪声的示例")
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        test_noise_generation()
        return
    
    # 生成数据集
    generate_noise_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        durations=args.durations,
        noise_types=args.noise_types,
        sample_rate=args.sample_rate,
        use_gpu=args.use_gpu,
        save_metadata=not args.no_metadata,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    # 检查 PyTorch 和 torchaudio
    try:
        import torch
        import torchaudio
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"torchaudio 版本: {torchaudio.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"错误: 缺少必要的库 - {e}")
        print("请安装: pip install torch torchaudio numpy matplotlib scipy pandas")
        exit(1)
    
    # 运行主函数
    main()