import os
import shutil
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.silence import split_on_silence
import logging
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_white_noise(audio_segment, noise_level=-20):
    """
    添加白噪声
    :param audio_segment: 原始音频段
    :param noise_level: 噪声级别(dB)，越小噪声越大，典型值：-40到-20
    :return: 添加噪声后的音频
    """
    # 生成白噪声，长度与原始音频相同
    noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))
    
    # 调整噪声级别
    noise = noise - (noise.dBFS - noise_level)
    
    # 混合噪声和原始音频
    return audio_segment.overlay(noise)

def split_large_audio_file(file_path, output_folder, max_duration_ms=2000, min_chunk_duration_ms=500):
    """
    切割大音频文件为小片段
    :param file_path: 原始音频文件路径
    :param output_folder: 输出文件夹
    :param max_duration_ms: 最大片段时长(毫秒)
    :param min_chunk_duration_ms: 最小片段时长(毫秒)
    :return: 切割后的文件列表
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取音频文件
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        audio = AudioSegment.from_file(file_path, format=file_extension[1:])
    except Exception as e:
        logging.error(f"无法读取文件 {file_path}: {e}")
        return []
    
    original_duration = len(audio)
    logging.info(f"切割文件: {os.path.basename(file_path)}, 时长: {original_duration}ms")
    
    if original_duration <= max_duration_ms:
        logging.info(f"文件无需切割，时长: {original_duration}ms <= {max_duration_ms}ms")
        return []
    
    # 切割音频
    chunk_files = []
    start_ms = 0
    chunk_index = 0
    
    while start_ms < original_duration:
        # 计算当前片段的结束位置
        end_ms = min(start_ms + max_duration_ms, original_duration)
        
        # 提取片段
        chunk = audio[start_ms:end_ms]
        chunk_duration = len(chunk)
        
        # 确保片段有足够长度
        if chunk_duration >= min_chunk_duration_ms:
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            chunk_filename = f"{base_name}_chunk{chunk_index:03d}.wav"
            chunk_path = os.path.join(output_folder, chunk_filename)
            
            # 标准化参数并导出
            chunk = chunk.set_frame_rate(16000)
            chunk = chunk.set_channels(1)
            chunk = chunk.set_sample_width(2)
            chunk.export(chunk_path, format="wav")
            
            chunk_files.append(chunk_path)
            logging.info(f"  创建片段 {chunk_index}: {start_ms}-{end_ms}ms ({chunk_duration}ms)")
            chunk_index += 1
        
        start_ms += max_duration_ms
    
    logging.info(f"切割完成: 创建了 {len(chunk_files)} 个片段")
    return chunk_files

def remove_silence_from_audio(audio_segment, min_silence_len=300, silence_thresh=-40, keep_silence=100):
    """
    去除音频中的静音部分
    :param audio_segment: 音频段
    :param min_silence_len: 最小静音长度(ms)
    :param silence_thresh: 静音阈值(dB)
    :param keep_silence: 保留的静音(ms)
    :return: 处理后的音频
    """
    # 分割静音
    chunks = split_on_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=10
    )
    
    if not chunks:
        logging.warning("没有检测到有效音频片段")
        return audio_segment
    
    # 合并有效片段
    processed_audio = AudioSegment.silent(duration=0)
    for chunk in chunks:
        processed_audio += chunk
    
    original_duration = len(audio_segment)
    processed_duration = len(processed_audio)
    
    logging.info(f"去静音: {original_duration}ms -> {processed_duration}ms "
                 f"(删除了 {original_duration - processed_duration}ms)")
    
    return processed_audio

def process_audio_file(input_path, output_folder, prefix, add_noise=False, 
                       noise_level=-45, remove_silence=True, min_audio_duration=500):
    """
    处理单个音频文件：标准化、去静音、加噪声
    :param input_path: 输入文件路径
    :param output_folder: 输出文件夹
    :param prefix: 文件名前缀
    :param add_noise: 是否添加噪声
    :param noise_level: 噪声级别
    :param remove_silence: 是否去除静音
    :param min_audio_duration: 最小音频时长(ms)
    :return: 处理后的文件路径
    """
    if not os.path.exists(input_path):
        logging.error(f"文件不存在: {input_path}")
        return None
    
    try:
        # 读取音频文件
        file_extension = os.path.splitext(input_path)[1].lower()
        audio = AudioSegment.from_file(input_path, format=file_extension[1:])
        original_duration = len(audio)
        
        # 检查音频长度
        if original_duration < min_audio_duration:
            logging.warning(f"音频过短: {input_path} ({original_duration}ms < {min_audio_duration}ms)")
            return None
        
        # 去除静音
        if remove_silence:
            audio = remove_silence_from_audio(audio)
        
        # 再次检查处理后的长度
        processed_duration = len(audio)
        if processed_duration < min_audio_duration:
            logging.warning(f"处理后音频过短: {input_path} ({processed_duration}ms)")
            return None
        
        # 添加噪声
        if add_noise:
            audio = add_white_noise(audio, noise_level=noise_level)
        
        # 标准化参数
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)
        
        # 生成输出文件名
        random_num = random.randint(1, 1000)
        output_filename = f"{prefix}_{random_num:03d}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # 避免文件名冲突
        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{prefix}_{random_num:03d}_{counter:02d}.wav"
            output_path = os.path.join(output_folder, output_filename)
            counter += 1
        
        # 导出文件
        audio.export(output_path, format="wav")
        
        logging.info(f"处理完成: {os.path.basename(input_path)} -> {output_filename} "
                    f"({original_duration}ms -> {processed_duration}ms)")
        
        return output_path
    
    except Exception as e:
        logging.error(f"处理失败 {input_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prefix, 
                   max_duration_to_split=2000, process_small_files=True,
                   add_noise=False, remove_silence=True):
    """
    处理整个文件夹的音频文件
    :param input_folder: 输入文件夹
    :param output_folder: 输出文件夹
    :param prefix: 文件名前缀
    :param max_duration_to_split: 超过此时长则切割(ms)
    :param process_small_files: 是否处理小文件
    :param add_noise: 是否添加噪声
    :param remove_silence: 是否去除静音
    """
    if not os.path.exists(input_folder):
        logging.error(f"输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有音频文件
    audio_extensions = ('.m4a', '.wav', '.mp3', '.flac', '.aac', '.ogg')
    audio_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(audio_extensions)]
    
    if not audio_files:
        logging.warning(f"文件夹中没有音频文件: {input_folder}")
        return
    
    logging.info(f"开始处理文件夹: {input_folder}")
    logging.info(f"找到 {len(audio_files)} 个音频文件")
    
    # 临时文件夹用于存放切割后的文件
    temp_split_folder = os.path.join(output_folder, "temp_split")
    if os.path.exists(temp_split_folder):
        shutil.rmtree(temp_split_folder)
    os.makedirs(temp_split_folder)
    
    processed_count = 0
    error_count = 0
    split_files = []
    
    # 第一步：切割大文件
    logging.info("=== 第一步：切割大音频文件 ===")
    for filename in audio_files:
        input_path = os.path.join(input_folder, filename)
        
        try:
            # 检查文件大小和时长
            file_extension = os.path.splitext(filename)[1].lower()
            audio = AudioSegment.from_file(input_path, format=file_extension[1:])
            duration = len(audio)
            
            if duration > max_duration_to_split:
                # 切割大文件
                logging.info(f"切割大文件: {filename} ({duration}ms)")
                split_result = split_large_audio_file(
                    input_path, 
                    temp_split_folder, 
                    max_duration_ms=max_duration_to_split
                )
                split_files.extend(split_result)
            else:
                # 小文件直接复制到临时文件夹处理
                if process_small_files:
                    temp_path = os.path.join(temp_split_folder, filename)
                    shutil.copy2(input_path, temp_path)
                    split_files.append(temp_path)
        
        except Exception as e:
            logging.error(f"处理文件 {filename} 失败: {e}")
            error_count += 1
    
    logging.info(f"切割完成: 总共 {len(split_files)} 个文件待处理")
    
    # 第二步：处理所有文件（包括切割后的小文件）
    logging.info("=== 第二步：处理所有音频文件 ===")
    for i, file_path in enumerate(split_files):
        logging.info(f"处理文件 [{i+1}/{len(split_files)}]: {os.path.basename(file_path)}")
        
        result = process_audio_file(
            file_path, 
            output_folder, 
            prefix,
            add_noise=add_noise,
            remove_silence=remove_silence
        )
        
        if result:
            processed_count += 1
        else:
            error_count += 1
    
    # 第三步：清理临时文件
    logging.info("=== 第三步：清理临时文件 ===")
    if os.path.exists(temp_split_folder):
        shutil.rmtree(temp_split_folder)
        logging.info(f"清理临时文件夹: {temp_split_folder}")
    
    # 第四步：删除原始文件夹中的文件（可选）
    logging.info("=== 第四步：清理原始文件（可选）===")
    delete_original = True  # 设置为False以保留原始文件
    if delete_original:
        for filename in audio_files:
            try:
                original_path = os.path.join(input_folder, filename)
                os.remove(original_path)
                logging.info(f"删除原始文件: {filename}")
            except Exception as e:
                logging.error(f"删除文件失败 {filename}: {e}")
    
    logging.info(f"处理完成: 成功 {processed_count} 个, 失败 {error_count} 个")
    logging.info(f"输出文件夹: {output_folder}")
    
    return processed_count

def main():
    """主函数"""
    # 定义要处理的文件夹配置
    # 格式: (输入文件夹, 输出文件夹, 文件名前缀, 是否添加噪声, 是否去静音)
    folders_to_process = [
        {
            "input": "audio_data/others",
            "output": "audio_data/others",
            "prefix": "others",
            "add_noise": False,
            "remove_silence": False,
            "max_duration_to_split": 1000  # 超过2秒则切割
        },
        # {
        #     "input": "audio_data/xiaoa",
        #     "output": "audio_data/xiaoa", 
        #     "prefix": "xiaoa",
        #     "add_noise": False,
        #     "remove_silence": True,
        #     "max_duration_to_split": 1000
        # }
    ]
    
    total_processed = 0
    
    for config in folders_to_process:
        input_folder = config["input"]
        output_folder = config["output"]
        prefix = config["prefix"]
        add_noise = config["add_noise"]
        remove_silence = config["remove_silence"]
        max_duration = config["max_duration_to_split"]
        
        if not os.path.exists(input_folder):
            logging.warning(f"跳过不存在的文件夹: {input_folder}")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"处理配置:")
        logging.info(f"  输入文件夹: {input_folder}")
        logging.info(f"  输出文件夹: {output_folder}")
        logging.info(f"  文件名前缀: {prefix}")
        logging.info(f"  添加噪声: {add_noise}")
        logging.info(f"  去除静音: {remove_silence}")
        logging.info(f"  切割阈值: {max_duration}ms")
        logging.info(f"{'='*60}")
        
        try:
            processed = process_folder(
                input_folder=input_folder,
                output_folder=output_folder,
                prefix=prefix,
                max_duration_to_split=max_duration,
                add_noise=add_noise,
                remove_silence=remove_silence
            )
            total_processed += processed
        except Exception as e:
            logging.error(f"处理文件夹 {input_folder} 时出错: {e}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"所有处理完成!")
    logging.info(f"总共成功处理: {total_processed} 个文件")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()