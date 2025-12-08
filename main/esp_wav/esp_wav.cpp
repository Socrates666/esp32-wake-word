#include "esp_wav.hpp"
#include <stdexcept>
#include <stdio.h>
#include <cstring>
#include <esp_log.h>
namespace wav{

wav::WavHeader::WavHeader(std::string file_path){
    f = fopen(file_path.c_str(), "r+b");
    if (f == NULL) {
        ESP_LOGE("WAV", "Failed to open file: %s", file_path.c_str());
        return;
    }
    status = WAV_OPEND;
    
    fseek(f, 0, SEEK_SET);
    
    // 逐字节读取并验证
    char tag[5] = {0};  // 留一个字节给字符串结束符
    
    // 读取 RIFF tag
    if (fread(tag, 1, 4, f) != 4) {
        ESP_LOGE("WAV", "Failed to read RIFF tag");
        return;
    }
    tag[4] = '\0';
    memcpy(riff_tag.data(), tag, 4);
    
    if (strncmp(tag, "RIFF", 4) != 0) {
        ESP_LOGE("WAV", "Invalid RIFF tag: %.4s, expected: RIFF", tag);
    }
    
    // 读取文件大小
    if (fread(&riff_length, sizeof(uint32_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read RIFF length");
        return;
    }
    
    // 读取 WAVE tag
    if (fread(tag, 1, 4, f) != 4) {
        ESP_LOGE("WAV", "Failed to read WAVE tag");
        return;
    }
    tag[4] = '\0';
    memcpy(wave_tag.data(), tag, 4);
    
    if (strncmp(tag, "WAVE", 4) != 0) {
        ESP_LOGE("WAV", "Invalid WAVE tag: %.4s, expected: WAVE", tag);
    }
    
    // 读取 fmt tag
    if (fread(tag, 1, 4, f) != 4) {
        ESP_LOGE("WAV", "Failed to read fmt tag");
        return;
    }
    tag[4] = '\0';
    memcpy(fmt_tag.data(), tag, 4);
    
    if (strncmp(tag, "fmt ", 4) != 0) {
        ESP_LOGE("WAV", "Invalid fmt tag: %.4s, expected: fmt ", tag);
    }
    
    // 继续读取其他字段...
    if (fread(&fmt_length, sizeof(uint32_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read fmt length");
        return;
    }
    if (fread(&audio_format, sizeof(uint16_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read audio format");
        return;
    }
    if (fread(&num_channels, sizeof(uint16_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read number of channels");
        return;
    }
    if (fread(&sample_rate, sizeof(uint32_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read sample rate");
        return;
    }
    if (fread(&byte_rate, sizeof(uint32_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read byte rate");
        return;
    }
    if (fread(&block_align, sizeof(uint16_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read block align");
        return;
    }
    if (fread(&bits_per_sample, sizeof(uint16_t), 1, f) != 1) {
        ESP_LOGE("WAV", "Failed to read bits per sample");
        return;
    }
    
    // 寻找 data chunk
    long data_pos = -1;
    
    while (1) {
        if (fread(tag, 1, 4, f) != 4) {
            ESP_LOGE("WAV", "Failed to read chunk tag while searching for data chunk");
            break;
        }
        tag[4] = '\0';
        
        uint32_t chunk_size;
        if (fread(&chunk_size, sizeof(uint32_t), 1, f) != 1) {
            ESP_LOGE("WAV", "Failed to read chunk size while searching for data chunk");
            break;
        }
        
        if (strncmp(tag, "data", 4) == 0) {
            memcpy(data_tag.data(), tag, 4);
            data_length = chunk_size;
            data_pos = ftell(f);
            break;
        } else {
            ESP_LOGW("WAV", "Skipping unknown chunk: %.4s, size: %" PRIu32 "", tag, chunk_size);
            // 跳过未知 chunk
            if (fseek(f, chunk_size, SEEK_CUR) != 0) {
                ESP_LOGE("WAV", "Failed to skip unknown chunk: %.4s", tag);
                break;
            }
        }
    }
    
    if (data_pos == -1) {
        ESP_LOGE("WAV", "Data chunk not found in WAV file");
        return;
    }
    
    // 计算要读取的样本数
    size_t samples_to_read = data_length / sizeof(int16_t);
    if (samples_to_read > 16000) {
        samples_to_read = 16000;
        ESP_LOGW("WAV", "Truncating to %zu samples", samples_to_read);
    }
    raw_data_pos = ftell(f);
    
    ESP_LOGI("WAV", "WAV file parsed successfully: %d channels, %" PRIu32 " Hz, %d bits, data length: %" PRIu32 "", 
             num_channels, sample_rate, bits_per_sample, data_length);
}
}
