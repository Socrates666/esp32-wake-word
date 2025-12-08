#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "esp_dsp.h"

static const char *TAG = "MFCC";

// 统一的释放函数
static void safe_free(void* ptr) {
    if (ptr) {
        heap_caps_free(ptr);
    }
}

// 改进的DCT-II实现，支持流式处理
static void dct_ii(const float *input, float *output, int n)
{
    if (!input || !output || n <= 0) return;
    
    // 对于小尺寸，使用简单的计算
    if (n <= 32) {
        for (int k = 0; k < n; k++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                float angle = M_PI * k * (2 * i + 1) / (2.0f * n);
                sum += input[i] * cosf(angle);
            }
            float scale = (k == 0) ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
            output[k] = scale * sum;
        }
    } else {
        // 对于大尺寸，使用预计算的表格
        static float *cos_table = NULL;
        static int prev_n = 0;
        
        if (cos_table == NULL || prev_n != n) {
            safe_free(cos_table);
            cos_table = (float*)heap_caps_malloc(n * n * sizeof(float), MALLOC_CAP_32BIT);
            if (!cos_table) return;
            
            for (int k = 0; k < n; k++) {
                for (int i = 0; i < n; i++) {
                    float angle = M_PI * k * (2 * i + 1) / (2.0f * n);
                    cos_table[k * n + i] = cosf(angle);
                }
            }
            prev_n = n;
        }
        
        for (int k = 0; k < n; k++) {
            float sum = 0.0f;
            const float *cos_row = &cos_table[k * n];
            for (int i = 0; i < n; i++) {
                sum += input[i] * cos_row[i];
            }
            float scale = (k == 0) ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
            output[k] = scale * sum;
        }
    }
}

static void pre_emphasis(const float* signal, int signal_len, float alpha, float* emphasized_signal)
{   
    if (!signal || !emphasized_signal || signal_len <= 0) return;
    
    emphasized_signal[0] = signal[0];
    for (int i = 1; i < signal_len; i++) {
        emphasized_signal[i] = signal[i] - alpha * signal[i-1];
    } 
}

static float* frame_division(const float* signal, int signal_len, int frame_size, int hop_size, int num_frames, float* frames)
{
    if (!signal || !frames || signal_len <= 0 || frame_size <= 0 || hop_size <= 0 || num_frames <= 0) {
        ESP_LOGE(TAG, "Invalid parameters in frame_division");
        return frames;
    }
    
    // 初始化为0
    memset(frames, 0, num_frames * frame_size * sizeof(float));
    
    for (int i = 0; i < num_frames; i++) {
        int frame_start = i * frame_size;
        int signal_start = i * hop_size;
        
        // 检查信号索引范围
        if (signal_start >= signal_len) {
            break;  // 超出信号长度
        }
        
        // 计算实际要复制的长度
        int copy_length = frame_size;
        if (signal_start + frame_size > signal_len) {
            copy_length = signal_len - signal_start;
        }
        
        // 复制数据
        for (int j = 0; j < copy_length; j++) {
            frames[frame_start + j] = signal[signal_start + j];
        }
    }
    
    return frames;
}

static void apply_window(float* frames, int num_frames, int frame_size, float alpha)
{
    if (!frames || num_frames <= 0 || frame_size <= 0) return;
    
    // 生成汉明窗
    float* window = (float*)heap_caps_malloc(frame_size * sizeof(float), MALLOC_CAP_32BIT);
    if (!window) return;
    
    for (int i = 0; i < frame_size; i++) {
        window[i] = alpha - (1.0f - alpha) * cosf(2.0f * M_PI * i / (frame_size - 1));
    }
    
    // 应用窗函数
    for (int i = 0; i < num_frames; i++) {
        int frame_start = i * frame_size;
        for (int j = 0; j < frame_size; j++) {
            frames[frame_start + j] *= window[j];
        }
    }

    safe_free(window);
}

static float hz_to_mel(float freq)
{
    if(freq == 0) freq = 1;
    return 1127.0f * log1pf(freq / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static float* create_mel_filterbank(int sampling_rate, int n_filters, int n_fft, int low_freq, int high_freq)
{
    if (high_freq <= 0) {
        high_freq = sampling_rate / 2;
    }
    
    int n_freq_bins = n_fft / 2 + 1;
    float* fbank = (float*)heap_caps_calloc(n_filters * n_freq_bins, sizeof(float), MALLOC_CAP_32BIT);
    if (!fbank) return NULL;
    
    // 计算mel频率范围
    float low_mel = hz_to_mel(low_freq);
    float high_mel = hz_to_mel(high_freq);
    
    // 在mel尺度上均匀分布的点
    float* mel_points = (float*)heap_caps_malloc((n_filters + 2) * sizeof(float), MALLOC_CAP_32BIT);
    if (!mel_points) {
        safe_free(fbank);
        return NULL;
    }
    
    for (int i = 0; i < n_filters + 2; i++) {
        mel_points[i] = low_mel + i * (high_mel - low_mel) / (n_filters + 1);
    }
    
    // 转换回Hz
    float* hz_points = (float*)heap_caps_malloc((n_filters + 2) * sizeof(float), MALLOC_CAP_32BIT);
    if (!hz_points) {
        safe_free(fbank);
        safe_free(mel_points);
        return NULL;
    }
    
    for (int i = 0; i < n_filters + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // 转换为FFT bin索引
    int* bin_points = (int*)heap_caps_malloc((n_filters + 2) * sizeof(int), MALLOC_CAP_32BIT);
    if (!bin_points) {
        safe_free(fbank);
        safe_free(mel_points);
        safe_free(hz_points);
        return NULL;
    }
    
    float bin_width = (float)sampling_rate / n_fft;
    for (int i = 0; i < n_filters + 2; i++) {
        bin_points[i] = (int)floorf(hz_points[i] / bin_width);
    }
    
    // 创建三角滤波器
    for (int i = 0; i < n_filters; i++) {
        int left = bin_points[i];
        int center = bin_points[i + 1];
        int right = bin_points[i + 2];
        
        // 确保索引有效
        left = (left < 0) ? 0 : left;
        center = (center < 0) ? 0 : center;
        right = (right < 0) ? 0 : right;
        left = (left >= n_freq_bins) ? n_freq_bins - 1 : left;
        center = (center >= n_freq_bins) ? n_freq_bins - 1 : center;
        right = (right >= n_freq_bins) ? n_freq_bins - 1 : right;
        
        // 确保三角滤波器有效
        if (left >= center) center = left + 1;
        if (center >= right) right = center + 1;
        if (right >= n_freq_bins) right = n_freq_bins - 1;
        
        // 上升部分
        for (int j = left; j <= center; j++) {
            if (j >= 0 && j < n_freq_bins) {
                fbank[i * n_freq_bins + j] = (float)(j - left) / (center - left);
            }
        }
        
        // 下降部分
        for (int j = center; j <= right; j++) {
            if (j >= 0 && j < n_freq_bins) {
                fbank[i * n_freq_bins + j] = (float)(right - j) / (right - center);
            }
        }
    }
    
    safe_free(mel_points);
    safe_free(hz_points);
    safe_free(bin_points);
    
    return fbank;
}

static void compute_power_spectrum(const float* frames, int num_frames, int frame_size, int n_fft, float* power_spectra)
{
    if (!frames || !power_spectra || num_frames <= 0 || frame_size <= 0 || n_fft <= 0) {
        ESP_LOGE(TAG, "Invalid parameters in compute_power_spectrum");
        return;
    }
    
    int n_freq_bins = n_fft / 2 + 1;
    
    // 临时缓冲区
    float* fft_input = (float*)heap_caps_calloc(n_fft * 2, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_32BIT);
    if (!fft_input) return;
    
    for (int i = 0; i < num_frames; i++) {
        memset(fft_input, 0, n_fft * 2 * sizeof(float));
        
        // 复制帧数据到实部，虚部设为0
        for (int j = 0; j < frame_size && j < n_fft; j++) {
            fft_input[2 * j] = frames[i * frame_size + j];
            fft_input[2 * j + 1] = 0.0f;
        }
        
        // 执行FFT
        dsps_fft2r_fc32_ansi(fft_input, n_fft);
        dsps_bit_rev_fc32(fft_input, n_fft);
        dsps_cplx2reC_fc32(fft_input, n_fft);
        
        // 计算功率谱，添加小常数避免除0
        for (int k = 0; k < n_freq_bins; k++) {
            float real = fft_input[2 * k];
            float imag = fft_input[2 * k + 1];
            float power = (real * real + imag * imag) / n_fft + 1e-12f;  // 添加小常数
            power_spectra[i * n_freq_bins + k] = power;
        }
    }
    
    safe_free(fft_input);
}

static void apply_mel_filterbank(const float* power_spectra, const float* mel_fbank, 
                                int num_frames, int n_freq_bins, int n_filters, float* mel_energies)
{
    if (!power_spectra || !mel_fbank || !mel_energies) return;
    
    for (int i = 0; i < num_frames; i++) {
        const float* spectrum = &power_spectra[i * n_freq_bins];
        
        for (int j = 0; j < n_filters; j++) {
            const float* filter = &mel_fbank[j * n_freq_bins];
            float energy = 0.0f;
            
            for (int k = 0; k < n_freq_bins; k++) {
                energy += spectrum[k] * filter[k];
            }
            
            // 确保能量不为0
            mel_energies[i * n_filters + j] = fmaxf(energy, 1e-12f);
        }
    }
}

// 改进的流式MFCC提取 - 处理单帧数据
float* flow_extract_mfcc_single_frame(const float* frame, int frame_size, int sampling_rate,
                   int n_fft, int n_filters, int n_mfcc)
{
    if (!frame || frame_size <= 0 || frame_size > n_fft) {
        ESP_LOGE(TAG, "Invalid frame parameters");
        return NULL;
    }
    
    // 1. 复制并加窗
    float* windowed_frame = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_32BIT);
    if (!windowed_frame) {
        ESP_LOGE(TAG, "Windowed frame allocation failed");
        return NULL;
    }
    
    memset(windowed_frame, 0, n_fft * sizeof(float));
    
    // 生成并应用汉明窗
    float alpha = 0.53836f;
    for (int i = 0; i < frame_size; i++) {
        float window = alpha - (1.0f - alpha) * cosf(2.0f * M_PI * i / (frame_size - 1));
        windowed_frame[i] = frame[i] * window;
    }
    
    // 2. 计算功率谱
    int n_freq_bins = n_fft / 2 + 1;
    float* power_spectrum = (float*)heap_caps_calloc(n_freq_bins, sizeof(float), MALLOC_CAP_32BIT);
    if (!power_spectrum) {
        ESP_LOGE(TAG, "Power spectrum allocation failed");
        safe_free(windowed_frame);
        return NULL;
    }
    
    float* fft_input = (float*)heap_caps_calloc(n_fft * 2, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_32BIT);
    if (!fft_input) {
        ESP_LOGE(TAG, "FFT input allocation failed");
        safe_free(windowed_frame);
        safe_free(power_spectrum);
        return NULL;
    }
    
    // 准备FFT输入
    for (int j = 0; j < n_fft; j++) {
        fft_input[2 * j] = windowed_frame[j];
        fft_input[2 * j + 1] = 0.0f;
    }
    
    // 执行FFT
    dsps_fft2r_fc32_ansi(fft_input, n_fft);
    dsps_bit_rev_fc32(fft_input, n_fft);
    dsps_cplx2reC_fc32(fft_input, n_fft);
    
    // 计算功率谱
    for (int k = 0; k < n_freq_bins; k++) {
        float real = fft_input[2 * k];
        float imag = fft_input[2 * k + 1];
        float power = (real * real + imag * imag) / n_fft + 1e-12f;
        power_spectrum[k] = power;
    }
    
    safe_free(fft_input);
    safe_free(windowed_frame);
    
    // 3. 创建梅尔滤波器组（可以缓存以减少重复计算）
    static float* mel_fbank = NULL;
    static int prev_params[3] = {0, 0, 0}; // sampling_rate, n_filters, n_fft
    
    if (!mel_fbank || prev_params[0] != sampling_rate || 
        prev_params[1] != n_filters || prev_params[2] != n_fft) {
        safe_free(mel_fbank);
        mel_fbank = create_mel_filterbank(sampling_rate, n_filters, n_fft, 0, -1);
        if (!mel_fbank) {
            ESP_LOGE(TAG, "Mel filterbank creation failed");
            safe_free(power_spectrum);
            return NULL;
        }
        prev_params[0] = sampling_rate;
        prev_params[1] = n_filters;
        prev_params[2] = n_fft;
    }
    
    // 4. 应用梅尔滤波器组
    float* mel_energies = (float*)heap_caps_calloc(n_filters, sizeof(float), MALLOC_CAP_32BIT);
    if (!mel_energies) {
        ESP_LOGE(TAG, "Mel energies allocation failed");
        safe_free(power_spectrum);
        return NULL;
    }
    
    for (int j = 0; j < n_filters; j++) {
        float energy = 0.0f;
        const float* filter = &mel_fbank[j * n_freq_bins];
        
        for (int k = 0; k < n_freq_bins; k++) {
            energy += power_spectrum[k] * filter[k];
        }
        mel_energies[j] = fmaxf(energy, 1e-12f);
    }
    
    safe_free(power_spectrum);
    
    // 5. 取对数
    for (int i = 0; i < n_filters; i++) {
        mel_energies[i] = logf(mel_energies[i]);
    }
    
    // 6. DCT 计算 MFCC 系数
    float* mfcc = (float*)heap_caps_calloc(n_mfcc, sizeof(float), MALLOC_CAP_32BIT);
    if (!mfcc) {
        ESP_LOGE(TAG, "MFCC allocation failed");
        safe_free(mel_energies);
        return NULL;
    }
    
    float* dct_temp = (float*)heap_caps_malloc(n_filters * sizeof(float), MALLOC_CAP_32BIT);
    if (dct_temp) {        
        // 执行 DCT
        dct_ii(mel_energies, dct_temp, n_filters);
        
        // 取前 n_mfcc 个系数
        for (int j = 0; j < n_mfcc && j < n_filters; j++) {
            mfcc[j] = dct_temp[j];
        }
        
        safe_free(dct_temp);
    }
    
    safe_free(mel_energies);
    
    return mfcc;
}

// 批量MFCC提取
float* extract_mfcc(const float* signal, int signal_len, int sampling_rate,
                   int frame_size, int hop_size, int n_fft, int n_filters, int n_mfcc)
{
    if (!signal || signal_len < frame_size) {
        ESP_LOGE(TAG, "Invalid signal parameters");
        return NULL;
    }
    
    // 1. 预加重
    float* emphasized_signal = (float*)heap_caps_calloc(signal_len, sizeof(float), MALLOC_CAP_32BIT);
    if (!emphasized_signal) {
        ESP_LOGE(TAG, "Pre-emphasis allocation failed");
        return NULL;
    }
    pre_emphasis(signal, signal_len, 0.97f, emphasized_signal);
    
    // 2. 分帧
    int num_frames = (signal_len - frame_size) / hop_size + 1;
    float* frames = (float*)heap_caps_calloc(num_frames * frame_size, sizeof(float), MALLOC_CAP_32BIT);
    if (!frames) {
        ESP_LOGE(TAG, "Frame division allocation failed");
        safe_free(emphasized_signal);
        return NULL;
    }
    
    frame_division(emphasized_signal, signal_len, frame_size, hop_size, num_frames, frames);
    safe_free(emphasized_signal);
    
    // 3. 加窗
    apply_window(frames, num_frames, frame_size, 0.53836f);
    
    // 4. 计算功率谱
    int n_freq_bins = n_fft / 2 + 1;
    float* power_spectra = (float*)heap_caps_calloc(num_frames * n_freq_bins, sizeof(float), MALLOC_CAP_32BIT);
    if (!power_spectra) {
        ESP_LOGE(TAG, "Power spectra allocation failed");
        safe_free(frames);
        return NULL;
    }
    
    compute_power_spectrum(frames, num_frames, frame_size, n_fft, power_spectra);
    safe_free(frames);
    
    // 5. 创建梅尔滤波器组
    float* mel_fbank = create_mel_filterbank(sampling_rate, n_filters, n_fft, 0, -1);
    if (!mel_fbank) {
        ESP_LOGE(TAG, "Mel filterbank creation failed");
        safe_free(power_spectra);
        return NULL;
    }
    
    // 6. 应用梅尔滤波器组
    float* mel_energies = (float*)heap_caps_calloc(num_frames * n_filters, sizeof(float), MALLOC_CAP_32BIT);
    if (!mel_energies) {
        ESP_LOGE(TAG, "Mel energies allocation failed");
        safe_free(power_spectra);
        safe_free(mel_fbank);
        return NULL;
    }
    
    apply_mel_filterbank(power_spectra, mel_fbank, num_frames, n_freq_bins, n_filters, mel_energies);
    safe_free(power_spectra);
    safe_free(mel_fbank);
    
    // 7. 取对数
    for (int i = 0; i < num_frames * n_filters; i++) {
        mel_energies[i] = logf(mel_energies[i]);
    }
    
    // 8. DCT 计算 MFCC 系数
    float* mfcc = (float*)heap_caps_calloc(num_frames * n_mfcc, sizeof(float), MALLOC_CAP_32BIT);
    if (!mfcc) {
        ESP_LOGE(TAG, "MFCC allocation failed");
        safe_free(mel_energies);
        return NULL;
    }
    
    float* dct_temp = (float*)heap_caps_malloc(n_filters * sizeof(float), MALLOC_CAP_32BIT);
    if (dct_temp) {
        for (int i = 0; i < num_frames; i++) {
            const float* mel_frame = &mel_energies[i * n_filters];
            
            // 执行 DCT
            dct_ii(mel_frame, dct_temp, n_filters);
            
            // 取前 n_mfcc 个系数
            for (int j = 0; j < n_mfcc && j < n_filters; j++) {
                mfcc[i * n_mfcc + j] = dct_temp[j];
            }
        }
        safe_free(dct_temp);
    }
    
    safe_free(mel_energies);
    
    return mfcc;
}

// 分析MFCC范围
void analyze_mfcc_range(float* mfcc, int size, const char* label) {
    if (!mfcc || size <= 0) return;
    
    float min_val = INFINITY;
    float max_val = -INFINITY;
    float sum = 0;
    int valid_count = 0;
    
    for (int i = 0; i < size; i++) {
        if (!isnan(mfcc[i]) && !isinf(mfcc[i])) {
            if (mfcc[i] < min_val) min_val = mfcc[i];
            if (mfcc[i] > max_val) max_val = mfcc[i];
            sum += mfcc[i];
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        ESP_LOGI(TAG, "%s MFCC Range: min=%.6f, max=%.6f, avg=%.6f, valid=%d/%d", 
                 label, min_val, max_val, sum/valid_count, valid_count, size);
    } else {
        ESP_LOGE(TAG, "%s MFCC: No valid values", label);
    }
}

// 清理MFCC缓存
void cleanup_mfcc_cache() {
    // 这里可以清理静态变量，如果需要的话
}

void free_mfcc(float* mfcc)
{
    safe_free(mfcc);
}