#include "esp_wake_word_detector.hpp"

#define MFCC_LEN 13*63

//任务句柄
static TaskHandle_t wake_word_task_handle = NULL;
static TaskHandle_t record_audio_task_handle = NULL;
// 静态分配内存
static int8_t static_buffer[MFCC_LEN];
static int8_t mfcc_buffer[MFCC_LEN];
static uint16_t buffer_head = 0;//指向static_buffer第一个数据
static uint16_t buffer_tail = 0;//指向static_buffer最后一个数据
//计算静态环形缓冲区是否满
static int shared_counter = 0;
// 互斥锁,防止共同访问static_buffer
static SemaphoreHandle_t mutex;  

static esp_err_t check_buffer_ptr(uint16_t ptr){
    if((ptr+1)%13==0)   return true;
    return false;
}

static int8_t* read_whole_mfcc_buffer(void){
    if((buffer_head+1)%13!=0)   return NULL;
    //memset(mfcc_buffer, 0, MFCC_LEN);
    memcpy(mfcc_buffer, static_buffer+buffer_head, (MFCC_LEN-buffer_head-1)*sizeof(int8_t));
    memcpy(mfcc_buffer+(MFCC_LEN-buffer_head-1), static_buffer, (buffer_head+1)*sizeof(int8_t));
    
    return mfcc_buffer;
}

static esp_err_t write_one_frame_mfcc_to_buffer(int8_t* mfcc_frame){
    if(mfcc_frame == NULL)  return ESP_FAIL;
    memcpy(static_buffer+buffer_tail, mfcc_frame, 13*sizeof(int8_t));
    buffer_tail = (buffer_tail+13)%MFCC_LEN;
    buffer_head = (buffer_head+13)%MFCC_LEN;
    return ESP_OK;
}

void record_task(void* param){

    const wakeWord_detection_config_t* config = (wakeWord_detection_config_t*)param;

    int16_t signal[320];
    size_t bytes = 0;
    float mfcc_float[13];
    int mfcc_int8[13];
    int16_t prev = 0;
    UBaseType_t res;

    //初始化前端算法
    dl::audio::SpeechFeatureConfig wake_config = {
        .sample_rate = 16000,
        .frame_length = 20,           // 20ms
        .frame_shift = 16,           // 16ms
        .num_mel_bins = 40,
        .num_ceps = 13,
        .preemphasis = 0.97f,
        .cepstral_lifter = 0.0f,
        .window_type = dl::audio::WinType::HAMMING,
        .low_freq = 0.0f,
        .high_freq = 8000.0f,
        .log_epsilon = 1e-6f,
        .use_log_fbank = 2,
        .raw_energy = false,
        .use_power = true,
        .use_energy = false,
        .use_int16_fft = true,
        .remove_dc_offset = false
    };
    audio::MFCC wake_mfcc(wake_config);

    while(1){
        //清空数据
        memset(signal, 0, 320*sizeof(int16_t));
        memset(mfcc_float, 0, 13*sizeof(float));
        //读取一帧
        config->read_mic(signal, 320, &bytes, 0);
        if (ret != ESP_OK || bytes != 320 * sizeof(int16_t)){
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        //计算一帧mfcc特征
        wake_mfcc.process_frame(signal, 320, mfcc_float, prev);
        prev = signal[319];
        // 量化到int8
        for (int i = 0; i < 13; i++) {
            int32_t quantized = (int32_t)lroundf(mfcc_float[i] * MFCC_SCALE);
            // 钳位到int8范围
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            mfcc_int8[i] = (int8_t)quantized;
        }
        res = write_one_frame_mfcc_to_buffer(mfcc_int8);
        if(shared_counter<64)   shared_counter++;
        if(res != ESP_OK) {
            printf("Failed to send item\n");
            continue;
        }
        if(shared_counter == 64){
            xTaskNotifyGive(wake_word_task_handle);
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
        
    }
    vTaskDelete(NULL);
}


void detect_task(void* param){

    //硬件接口
    const wakeWord_detection_config_t* config = (wakeWord_detection_config_t*)param;

    //初始化关键词识别模型
    dl::Model *model =
        new dl::Model(config->model_path, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;
    std::vector<int> input_shape = model_input->get_shape();
    int8_t* data_ptr;
    size_t item_size = 0;
    while(1){
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        // 1. 获取数据（但不从ringbuf删除）
        item_size = 0;
        data_ptr = read_whole_mfcc_buffer();
        xTaskNotifyGive(record_audio_task_handle);

        // 2. 此时数据仍然在ringbuf中，可以安全地读取和拷贝
        if (data_ptr != NULL) {
            printf("获取到 %d 字节数据（数据仍在ringbuf中）\n", item_size);
            dl::TensorBase mfcc_tensor(input_shape, data_ptr, 0, dl::DATA_TYPE_INT8, true);
            model_input->exponent=0;
            bool res = model_input->assign(&mfcc_tensor);
            model->run();
            float sigmoid = 1 / (1 + exp(-model_output->get_element<int8_t>(0)));
            if(sigmoid>=70) config->callback(WAKE_WORD_DETECTED, (void*)&sigmoid);
        }
        
        vTaskDelay(pdMS_TO_TICKS(1)); 
    }
    vTaskDelete(NULL);
}




esp_err_t wakeWord_detection_open(wakeWord_detection_config_t* wake_config){
    mutex = xSemaphoreCreateMutex();
    xTaskCreate(record_task, "wake_word_recognization", 4 * 1024, wake_config, 4, record_audio_task_handle);
    xTaskCreate(detect_task, "wake_word_recognization", 4 * 1024, wake_config, 3, wake_word_task_handle);
    return ESP_OK;
}

void wakeWord_detection_close(void){
    if(record_audio_task_handle)    vTaskDelete(record_audio_task_handle);
    if(wake_word_task_handle)    vTaskDelete(wake_word_task_handle);
}