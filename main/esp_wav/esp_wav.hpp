#pragma once

#include "esp_log.h"
#include "esp_err.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include <cstdint>
#include <string>
#include <cinttypes>  // 如果还没有添加的话
#include <array>
namespace wav {

#define data_size_field_pos_ 0x28
#define WAV_OPEND 1
#define EMPTY_WAV -1
#define WAV_CLOSED 0 


class WavHeader {
private:
    std::array<char, 4> riff_tag;        // "RIFF"
    uint32_t riff_length;                // Length of the rest of the file
    std::array<char, 4> wave_tag;        // "WAVE"
    std::array<char, 4> fmt_tag;         // "fmt "
    uint32_t fmt_length;                 // Length of the fmt data (16 for PCM)
    uint16_t audio_format;               // Audio format (1 for PCM)
    uint16_t num_channels;               // Number of channels
    uint32_t sample_rate;                // Sampling rate
    uint32_t byte_rate;                  // Byte rate
    uint16_t block_align;                // Block align
    uint16_t bits_per_sample;            // Bits per sample
    std::array<char, 4> data_tag;        // "data"
    uint32_t data_length;                // Length of the data section
    uint32_t raw_data_pos;               // Pointer to the raw audio data
    std::string file_path;
    FILE* f = nullptr;
    esp_err_t status = EMPTY_WAV;
public:
    // 构造函数
    WavHeader() = default;
    
    // 使用文件的构造函数
    WavHeader(std::string file_path);
    ~WavHeader(){
        if(status >= ESP_OK)
        {
            if(f)   fclose(f);
        }
    }
    // 带参数的构造函数
    WavHeader(std::string file_path, uint16_t channels, uint32_t sample_rate, uint16_t bits_per_sample, 
              uint32_t data_size = 0) {
        status = WAV_CLOSED;
        initialize(file_path, channels, sample_rate, bits_per_sample, data_size);
    }
    
    // 初始化函数
    void initialize(std::string file_path, uint16_t channels, uint32_t sr, uint16_t bps, uint32_t data_size = 0) {
        // 设置固定标签
        setString(riff_tag, "RIFF");
        setString(wave_tag, "WAVE");
        setString(fmt_tag, "fmt ");
        setString(data_tag, "data");
        
        // 计算各种参数
        fmt_length = 16;
        audio_format = 1;  // PCM format
        num_channels = channels;
        sample_rate = sr;
        bits_per_sample = bps;
        block_align = channels * bps / 8;
        byte_rate = sample_rate * block_align;
        data_length = data_size;
        riff_length = 36 + data_length;  // 4 + 24 + 8 + data_length
        raw_data_pos = 44;  // Standard WAV header size
        this->file_path = file_path;
    }
    
    // 设置字符串到数组的辅助函数
    void setString(std::array<char, 4>& arr, const char* str) {
        for (int i = 0; i < 4 && str[i] != '\0'; ++i) {
            arr[i] = str[i];
        }
    }
    
    // Getter 方法
    FILE* getFilePtr() const { return f; }
    esp_err_t getFileStatus() const { return status; }
    uint16_t getNumChannels() const { return num_channels; }
    uint32_t getSampleRate() const { return sample_rate; }
    uint16_t getBitsPerSample() const { return bits_per_sample; }
    uint32_t getDataLength() const { return data_length; }
    uint32_t getByteRate() const { return byte_rate; }
    uint16_t getBlockAlign() const { return block_align; }
    uint32_t getRawDataPosition() const { return raw_data_pos; }
    
    // Setter 方法
    void setDataLength(uint32_t length) { 
        data_length = length; 
        riff_length = 36 + data_length;
    }
    
    // 验证头文件有效性
    bool isValid() const {
        return (std::string(riff_tag.data(), 4) == "RIFF" &&
                std::string(wave_tag.data(), 4) == "WAVE" &&
                std::string(fmt_tag.data(), 4) == "fmt " &&
                std::string(data_tag.data(), 4) == "data" &&
                audio_format == 1 &&  // PCM
                num_channels > 0 &&
                sample_rate > 0 &&
                bits_per_sample > 0);
    }
    
    // 获取头文件总大小
    static constexpr size_t getHeaderSize() { return 44; }
    
    // 转换为字节数组（用于文件写入）
    std::array<uint8_t, 44> toByteArray() const {
        std::array<uint8_t, 44> buffer{};
        size_t pos = 0;
        
        // 复制各个字段到缓冲区
        copyToBuffer(buffer, pos, riff_tag);
        copyToBuffer(buffer, pos, riff_length);
        copyToBuffer(buffer, pos, wave_tag);
        copyToBuffer(buffer, pos, fmt_tag);
        copyToBuffer(buffer, pos, fmt_length);
        copyToBuffer(buffer, pos, audio_format);
        copyToBuffer(buffer, pos, num_channels);
        copyToBuffer(buffer, pos, sample_rate);
        copyToBuffer(buffer, pos, byte_rate);
        copyToBuffer(buffer, pos, block_align);
        copyToBuffer(buffer, pos, bits_per_sample);
        copyToBuffer(buffer, pos, data_tag);
        copyToBuffer(buffer, pos, data_length);
        
        return buffer;
    }
    bool write_info_to_file(void) {
        if (status == WAV_CLOSED) {
            f = fopen(file_path.c_str(), "wb");
            status = WAV_OPEND;
        }else if(status == EMPTY_WAV){
            ESP_LOGE("WavHeader", "WAVHEADER Is Empty");
            return false;
        }

        // 写入 WAV 头
        auto header_bytes = toByteArray();
        size_t written = fwrite(header_bytes.data(), 1, header_bytes.size(), f);
        if (written != header_bytes.size()) {
            ESP_LOGE("WavHeader", "Failed to write WAV header");
            return false;
        }
        return true;
    }
    bool write_data_to_file(const int16_t* audio_data = nullptr, size_t audio_size = 0) {
                // 写入音频数据（如果提供）
        
        if (audio_data != nullptr && audio_size > 0) {
            size_t written = fwrite(audio_data, sizeof(int16_t), audio_size, f);
            if (written != audio_size) {
                ESP_LOGE("WavHeader", "Failed to write audio data: %zu/%zu", written, audio_size);
                return false;
            }
            data_length+=written;
            //ESP_LOGI("WavHeader", "Wrote %zu audio samples", audio_size);
        }else{
            return false;
        }
        return true;
    }
    bool finalize_wav_file(void) {
        if (f == nullptr) {
            ESP_LOGE("WavHeader", "文件未打开");
            return false;
        }
        
        // 1. 更新 RIFF 头部的文件总大小
        fseek(f, 4, SEEK_SET);
        uint32_t file_total_size = 36 + data_length;  // 36 = RIFF头(12) + fmt块(24) + data头(8)
        size_t written = fwrite(&file_total_size, sizeof(uint32_t), 1, f);
        if (written != 1) {
            ESP_LOGE("WavHeader", "更新RIFF大小失败");
            fclose(f);
            f = nullptr;
            return false;
        }
        
        // 2. 更新 data 块的大小
        fseek(f, 40, SEEK_SET);  // data块大小在文件偏移40字节处
        written = fwrite(&data_length, sizeof(uint32_t), 1, f);
        if (written != 1) {
            ESP_LOGE("WavHeader", "更新data大小失败");
            fclose(f);
            f = nullptr;
            return false;
        }
        
        // 3. 刷新并关闭文件
        fflush(f);
        fclose(f);
        f = nullptr;
        
        ESP_LOGI("WavHeader", "WAV文件完成: 数据大小=%" PRIu32 " bytes", data_length);
        return true;
    }
private:
    // 辅助函数：复制数据到缓冲区
    template<typename T>
    void copyToBuffer(std::array<uint8_t, 44>& buffer, size_t& pos, const T& value) const {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(T); ++i) {
            buffer[pos++] = bytes[i];
        }
    }
    
    // 特化版本用于字符数组
    void copyToBuffer(std::array<uint8_t, 44>& buffer, size_t& pos, 
                     const std::array<char, 4>& arr) const {
        for (size_t i = 0; i < 4; ++i) {
            buffer[pos++] = static_cast<uint8_t>(arr[i]);
        }
    }
};

}
