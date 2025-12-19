#include "esp_wake_word_detector.hpp"

#define MFCC_LEN 13*63

// 任务句柄
static TaskHandle_t wake_word_task_handle = NULL;
static TaskHandle_t record_audio_task_handle = NULL;

// MFCC 缓冲区
static int8_t static_buffer[MFCC_LEN];
static int8_t mfcc_buffer[MFCC_LEN];
static uint16_t buffer_head = 0;
static uint16_t buffer_tail = 0;
static int shared_counter = 0;

static SemaphoreHandle_t mutex;  
static wakeWord_detection_config_t g_wake_config;

// ==================== MFCC 缓冲区管理 ====================

static int8_t* read_whole_mfcc_buffer(void){
    xSemaphoreTake(mutex, portMAX_DELAY);
    memcpy(mfcc_buffer, static_buffer+buffer_head, (MFCC_LEN-buffer_head)*sizeof(int8_t));
    if(buffer_head>0) {
        memcpy(mfcc_buffer+(MFCC_LEN-buffer_head), static_buffer, (buffer_head)*sizeof(int8_t));
    }
    xSemaphoreGive(mutex);
    return mfcc_buffer;
}

static esp_err_t write_one_frame_mfcc_to_buffer(int8_t* mfcc_frame){
    if(mfcc_frame == NULL) return ESP_FAIL;
    
    xSemaphoreTake(mutex, portMAX_DELAY);
    
    memcpy(static_buffer+buffer_tail, mfcc_frame, 13*sizeof(int8_t));
    
    if(shared_counter < 64) shared_counter++;
    
    buffer_tail = (buffer_tail+13) >= MFCC_LEN ? (buffer_tail+13) % MFCC_LEN : buffer_tail+13;
    
    if(shared_counter == 64) {
        buffer_head = (buffer_head+13) >= MFCC_LEN ? (buffer_head+13) % MFCC_LEN : buffer_head+13;
    }
    
    xSemaphoreGive(mutex);
    return ESP_OK;
}

// ==================== 音频录制任务 ====================

void record_task(void* param){
    static const wakeWord_detection_config_t* config = (wakeWord_detection_config_t*)param;
    
    // 持续接收的音频缓冲区（不再每次重新分配）
    static int16_t signal_48k[3840];  // 48kHz * 4通道 * 20ms
    
    // 降采样后的数据
    static int16_t signal_16k[320];   // 16kHz * 20ms
    
    static float mfcc_float[13];
    static int8_t mfcc_int8[13];
    static int16_t prev = 0;

    // 初始化 MFCC
    static dl::audio::SpeechFeatureConfig wake_config;
    wake_config.sample_rate = 16000;
    wake_config.frame_length = 20;
    wake_config.frame_shift = 16;
    wake_config.num_mel_bins = 40;
    wake_config.num_ceps = 13;
    wake_config.preemphasis = 0.97f;
    wake_config.cepstral_lifter = 0.0f;
    wake_config.window_type = dl::audio::WinType::HAMMING;
    wake_config.low_freq = 0.0f;
    wake_config.high_freq = 8000.0f;
    wake_config.log_epsilon = 1e-6f;
    wake_config.use_log_fbank = 2;
    wake_config.raw_energy = false;
    wake_config.use_power = true;
    wake_config.use_energy = false;
    wake_config.use_int16_fft = true;
    wake_config.remove_dc_offset = false;
    dl::audio::MFCC wake_mfcc(wake_config);
    
    TickType_t last_wake_time = xTaskGetTickCount();
    
    while(1){
        memset(mfcc_float, 0, 13*sizeof(float));

        if(config->read_mic != NULL){
            size_t bytes_read = 0;
            esp_err_t res = config->read_mic(signal_48k, 3840, &bytes_read, 20);
            
            if (res != ESP_OK) {
                ESP_LOGE("record", "read_mic failed: %s", esp_err_to_name(res));
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
        }
        
        // 从 TDM 4 通道提取单通道（MIC-L 是 CH0）
        static int16_t signal_mono_48k[960];
        for (int i = 0; i < 960; i++) {
            int16_t mic_l  = signal_48k[i * 4 + 0];  // CH0: MIC-L (权重 40%)
            int16_t aec_ref = signal_48k[i * 4 + 1];  // CH1: AEC参考 (权重 20%)
            int16_t mic_r  = signal_48k[i * 4 + 2];  // CH2: MIC-R (权重 40%)
            
            int32_t weighted = ((int32_t)mic_l << 6) + ((int32_t)aec_ref << 5) + ((int32_t)mic_r << 6);
            signal_mono_48k[i] = (int16_t)(weighted >> 7);  // 除以 128 ≈ 除以 100
        }
        
        // 降采样 48kHz → 16kHz
        for (int i = 0; i < 320; i++) {
            // ✅ 改为加权平均（保留更多高频信息）
            // 权重：[1, 2, 1] / 4，比简单平均 [1,1,1]/3 更好
            int32_t weighted = (int32_t)signal_mono_48k[i*3 + 0] * 1
                            + (int32_t)signal_mono_48k[i*3 + 1] * 2
                            + (int32_t)signal_mono_48k[i*3 + 2] * 1;
            signal_16k[i] = (int16_t)(weighted >> 2);  // 除以 4
        }
        
        // 计算 MFCC
        wake_mfcc.process_frame(signal_16k, 320, mfcc_float, prev);
        prev = signal_16k[319];
        
        // 量化到 int8
        for (int i = 0; i < 13; i++) {
            int32_t quantized = (int32_t)lroundf(mfcc_float[i]);
            mfcc_int8[i] = (quantized > 127) ? 127 : (quantized < -128) ? -128 : (int8_t)quantized;
        }
        
        // 写入 MFCC 缓冲区
        esp_err_t res = write_one_frame_mfcc_to_buffer(mfcc_int8);
        if(res != ESP_OK) {
            ESP_LOGE("record", "write_one_frame failed");
            continue;
        }
        
        // 缓冲区满时通知检测任务
        if(shared_counter == 64){
            xTaskNotifyGive(wake_word_task_handle);
        }
        
        vTaskDelayUntil(&last_wake_time, pdMS_TO_TICKS(20));
        //printf("time delay : %lu ms\n", last_wake_time);
    }

    vTaskDelete(NULL);
}

// ==================== 推理任务 ====================

void detect_task(void* param){
    static const wakeWord_detection_config_t* config = (wakeWord_detection_config_t*)param;

    static dl::Model *model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    static std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    static dl::TensorBase *model_input = model_inputs.begin()->second;
    static std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    static dl::TensorBase *model_output = model_outputs.begin()->second;
    static std::vector<int> input_shape = model_input->get_shape();
    
    static int8_t* data_ptr;
    
    // ✅ CMVN 输出缓冲（量化后的 int8）
    static int8_t mfcc_cmvn_buffer[13 * 64];
    
    int inference_count = 0;
    
    while(1){
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        data_ptr = read_whole_mfcc_buffer();

        if (data_ptr != NULL) {
            uint32_t cmvn_start = esp_timer_get_time() / 1000;
            
            // ✅ Step 1: 计算每个维度的均值（用浮点数计算统计）
            float mean[13] = {0.0f};
            for (int dim = 0; dim < 13; dim++) {
                float sum = 0.0f;
                for (int frame = 0; frame < 63; frame++) {  // 63 帧
                    sum += (float)data_ptr[frame * 13 + dim];
                }
                mean[dim] = sum / 63.0f;
            }
            
            // ✅ Step 2: 计算每个维度的标准差
            float std_dev[13] = {0.0f};
            for (int dim = 0; dim < 13; dim++) {
                float variance = 0.0f;
                for (int frame = 0; frame < 63; frame++) {
                    float diff = (float)data_ptr[frame * 13 + dim] - mean[dim];
                    variance += diff * diff;
                }
                std_dev[dim] = sqrtf(variance / 63.0f);
            }
            
            // ✅ Step 3: CMVN 归一化（int8 输入 → 浮点计算 → int8 输出）
            float epsilon = 1e-8f;
            for (int i = 0; i < 13 * 63; i++) {
                int dim = i % 13;
                float normalized = ((float)data_ptr[i] - mean[dim]) / (std_dev[dim] + epsilon);
                
                // 直接量化为 int8
                int32_t quantized = (int32_t)lroundf(normalized);
                if (quantized > 127) quantized = 127;
                if (quantized < -128) quantized = -128;
                mfcc_cmvn_buffer[i] = (int8_t)quantized;
            }
            
            uint32_t cmvn_end = esp_timer_get_time() / 1000;
            
            // ✅ Step 4: 模型推理（使用 CMVN 后的 int8 数据）
            int input_exponent = model_input->exponent;
            dl::TensorBase mfcc_tensor(input_shape, mfcc_cmvn_buffer, 
                                       0, dl::DATA_TYPE_INT8, false);
            
            bool res = model_input->assign(&mfcc_tensor);
            
            uint32_t run_start = esp_timer_get_time() / 1000;
            model->run();
            uint32_t run_end = esp_timer_get_time() / 1000;
            
            int8_t raw_output = model_output->get_element<int8_t>(0);
            float output_float = raw_output * powf(2.0f, model_output->exponent);
            float sigmoid = 1 / (1 + expf(-output_float)) * 100;
            
            inference_count++;
            if (inference_count % 10 == 1) {
                ESP_LOGI("detect", "Raw: %d, Float: %.4f, Sigmoid: %.2f%% | CMVN: %lums, Run: %lums", 
                        raw_output, output_float, sigmoid, 
                        cmvn_end - cmvn_start, run_end - run_start);
                
                ESP_LOGI("detect", "MFCC[0..4] orig: %d,%d,%d,%d,%d", 
                        data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3], data_ptr[4]);
                ESP_LOGI("detect", "MFCC[0..4] cmvn: %d,%d,%d,%d,%d", 
                        mfcc_cmvn_buffer[0], mfcc_cmvn_buffer[1], mfcc_cmvn_buffer[2], 
                        mfcc_cmvn_buffer[3], mfcc_cmvn_buffer[4]);
                ESP_LOGI("detect", "Mean[0..4]: %.2f,%.2f,%.2f,%.2f,%.2f", 
                        mean[0], mean[1], mean[2], mean[3], mean[4]);
            }
            
            if(sigmoid >= 80){
                ESP_LOGI("wake", "🎙️ 检测到唤醒词！信置度: %.2f%%", sigmoid);
                config->callback(WAKE_WORD_DETECTED, (void*)&sigmoid);
                vTaskDelay(pdMS_TO_TICKS(5000));
                
                // 重置缓冲区
                xSemaphoreTake(mutex, portMAX_DELAY);
                memset(mfcc_buffer, 0, MFCC_LEN * sizeof(int8_t));
                memset(static_buffer, 0, MFCC_LEN * sizeof(int8_t));
                shared_counter = 0;
                buffer_head = 0;
                buffer_tail = 0;
                xSemaphoreGive(mutex);
            }
        }
    }
    delete model;
    vTaskDelete(NULL);
}

// ==================== 外部接口 ====================

esp_err_t wakeWord_detection_open(wakeWord_detection_config_t* wake_config){
    if (!wake_config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    memcpy(&g_wake_config, wake_config, sizeof(g_wake_config));
    mutex = xSemaphoreCreateMutex();
    
    xTaskCreate(record_task, "record_audio", 32 * 1024, &g_wake_config, 3, &record_audio_task_handle);
    xTaskCreate(detect_task, "wake_word_recognization", 16 * 1024, &g_wake_config, 2, &wake_word_task_handle);
    
    return ESP_OK;
}

void wakeWord_detection_close(void){
    if(record_audio_task_handle) vTaskDelete(record_audio_task_handle);
    if(wake_word_task_handle) vTaskDelete(wake_word_task_handle);
    if(mutex) vSemaphoreDelete(mutex);
}