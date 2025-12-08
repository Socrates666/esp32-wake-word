/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include "dirent.h"
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_system.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_spiffs.h"
#include "esp_mfcc/mfcc.h"
#include "esp_dsp.h"
#include "esp_wav/esp_wav.hpp"
#include <cmath>
#include "esp_adc/adc_continuous.h"
#include "ring_buffer/ring_buffer.h"
#include "audio/speech_features/dl_mfcc.hpp"
#include "audio/speech_features/dl_speech_features.hpp"
#include "audio/common/dl_audio_common.hpp"

#define EXAMPLE_ADC_UNIT                    ADC_UNIT_1
#define EXAMPLE_ADC_CHANNEL                 ADC_CHANNEL_7
#define _EXAMPLE_ADC_UNIT_STR(unit)         #unit
#define EXAMPLE_ADC_UNIT_STR(unit)          _EXAMPLE_ADC_UNIT_STR(unit)
#define EXAMPLE_ADC_CONV_MODE               ADC_CONV_SINGLE_UNIT_1
#define EXAMPLE_ADC_ATTEN                   ADC_ATTEN_DB_11
#define EXAMPLE_ADC_BIT_WIDTH               ADC_BITWIDTH_12

#define ADC_TEST_OUTPUT_TYPE                ADC_DIGI_OUTPUT_FORMAT_TYPE1
#define EXAMPLE_ADC_GET_CHANNEL(result)     ((result).type2.channel)
#define EXAMPLE_ADC_GET_DATA(result)        ((result).type2.data)

static const char* TAG = "MAIN";

using namespace dl;

#include <math.h>
int8_t data2[63*13]={
    -49,-58,-60,-61,-60,-63,-61,-63,-63,-62,-60,-52,-46,-41,-36,-31,-28,-32,-31,-32,-31,-35,-27,-21,-22,-24,-28,-33,-33,-34,-35,-35,-42,-26,-24,-26,-23,-23,-25,-26,-28,-31,-36,-41,-44,-46,-48,-51,-53,-55,-87,-87,-87,-87,-87,-87,-87,-87,-87,-87,-87,-87,-87,
-1,-8,-8,-8,-8,-9,-9,-9,-8,-9,-8,-7,-15,-19,-23,-24,-28,-28,-27,-27,-26,-18,-8,1,2,2,3,3,4,4,7,6,-1,3,1,1,0,0,0,0,1,1,1,0,-2,-3,-4,-8,-8,-4,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,-1,0,0,2,1,1,3,2,1,1,4,1,-2,-1,1,3,3,2,3,2,0,4,-4,-4,-5,-2,-1,0,0,0,1,-1,-6,-3,-3,-3,-3,-3,-2,-2,-1,3,3,2,2,0,1,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,0,0,1,-1,2,1,1,1,1,3,3,2,2,3,3,5,4,4,3,4,6,3,2,1,0,-1,-2,-1,-1,-4,-4,0,1,0,1,2,4,5,4,5,6,5,5,4,6,3,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,1,2,1,1,1,1,1,1,1,1,1,-1,0,0,-1,-3,-2,-2,-3,-4,-1,-1,-3,-3,-4,-5,-5,-7,-7,-8,-6,-3,-5,-3,-3,-3,-2,-2,-2,-2,-1,-1,-1,2,2,0,1,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,
-1,1,1,0,1,-1,0,2,2,2,0,3,3,0,2,-1,0,3,1,0,0,0,-2,-2,-1,-1,0,0,-2,-1,0,2,1,-1,-1,-1,-2,-2,-2,-3,-4,-4,-4,-2,-2,-2,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,2,0,1,1,1,1,2,1,2,0,1,-1,-1,0,0,1,1,0,1,2,-1,-3,-1,1,2,4,5,7,7,4,3,4,1,0,-1,-1,-1,-2,-2,-2,-3,-2,-1,-1,-2,-2,-1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,
2,1,0,1,2,1,2,1,2,2,0,2,2,4,2,2,2,-1,0,0,1,0,0,-3,-3,-3,-2,-2,-1,-2,-1,-3,-1,-3,-2,-3,-2,-2,-1,-1,-1,0,1,2,2,2,1,2,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,2,1,2,1,3,2,1,2,2,2,1,2,1,1,0,1,1,2,1,2,1,-1,3,1,0,-1,-1,-2,-1,-1,0,0,1,1,2,2,2,2,2,1,2,2,0,2,2,2,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,
-1,3,3,3,1,4,2,1,1,2,2,0,1,1,0,0,0,0,1,1,2,-2,1,0,1,2,2,2,0,0,-1,1,2,1,1,2,1,0,-1,-1,-1,-1,-2,-3,-1,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,2,2,2,3,2,1,0,1,1,1,-3,-2,-1,0,2,1,0,0,3,1,-3,-1,0,2,1,-1,0,0,0,0,0,1,2,0,0,-1,-1,-1,-1,0,-1,-1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,1,2,1,3,1,2,0,2,0,1,0,0,0,1,0,1,0,2,2,2,1,-1,-1,-1,-1,0,0,1,1,2,1,0,-1,0,-1,0,0,0,1,2,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,2,1,1,1,1,2,2,1,1,1,3,2,2,2,1,2,1,2,1,3,2,-1,-4,-4,-2,-2,-2,-1,-3,-2,-3,-4,-3,-2,-2,-1,-1,-1,-1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0

};

int8_t data1[63*13]={
-49,-1,0,1,0,-1,0,2,0,-1,1,0,0,
-58,-8,-1,0,1,1,2,1,2,3,2,1,0,
-60,-8,0,0,2,1,0,0,1,3,2,2,2,
-61,-8,0,1,1,0,1,1,2,3,2,1,1,
-60,-8,2,-1,1,1,1,2,1,1,3,3,1,
-63,-9,1,2,1,-1,1,1,3,4,2,1,1,
-61,-9,1,1,1,0,1,2,2,2,1,2,1,
-63,-9,3,1,1,2,2,1,1,1,0,0,2,
-63,-8,2,1,1,2,1,2,2,1,1,2,2,
-62,-9,1,1,1,2,2,2,2,2,1,0,1,
-60,-8,1,3,1,0,0,0,2,2,1,1,1,
-52,-7,4,3,1,3,1,2,1,0,-3,0,1,
-46,-15,1,2,-1,3,-1,2,2,1,-2,0,3,
-41,-19,-2,2,0,0,-1,4,1,1,-1,0,2,
-36,-23,-1,3,0,2,0,2,1,0,0,1,2,
-31,-24,1,3,-1,-1,0,2,0,0,2,0,2,
-28,-28,3,5,-3,0,1,2,1,0,1,1,1,
-32,-28,3,4,-2,3,1,-1,1,0,0,0,2,
-31,-27,2,4,-2,1,0,0,2,1,0,2,1,
-32,-27,3,3,-3,0,1,0,1,1,3,2,2,
-31,-26,2,4,-4,0,2,1,2,2,1,2,1,
-35,-18,0,6,-1,0,-1,0,1,-2,-3,1,3,
-27,-8,4,3,-1,-2,-3,0,-1,1,-1,-1,2,
-21,1,-4,2,-3,-2,-1,-3,3,0,0,-1,-1,
-22,2,-4,1,-3,-1,1,-3,1,1,2,-1,-4,
-24,2,-5,0,-4,-1,2,-3,0,2,1,-1,-4,
-28,3,-2,-1,-5,0,4,-2,-1,2,-1,0,-2,
-33,3,-1,-2,-5,0,5,-2,-1,2,0,0,-2,
-33,4,0,-1,-7,-2,7,-1,-2,0,0,1,-2,
-34,4,0,-1,-7,-1,7,-2,-1,0,0,1,-1,
-35,7,0,-4,-8,0,4,-1,-1,-1,0,2,-3,
-35,6,1,-4,-6,2,3,-3,0,1,0,1,-2,
-42,-1,-1,0,-3,1,4,-1,0,2,1,0,-3,
-26,3,-6,1,-5,-1,1,-3,1,1,2,-1,-4,
-24,1,-3,0,-3,-1,0,-2,1,1,0,0,-3,
-26,1,-3,1,-3,-1,-1,-3,2,2,0,-1,-2,
-23,0,-3,2,-3,-2,-1,-2,2,1,-1,0,-2,
-23,0,-3,4,-2,-2,-1,-2,2,0,-1,0,-1,
-25,0,-3,5,-2,-2,-2,-1,2,-1,-1,0,-1,
-26,0,-2,4,-2,-3,-2,-1,2,-1,-1,1,-1,
-28,1,-2,5,-2,-4,-2,-1,1,-1,0,2,-1,
-31,1,-1,6,-1,-4,-3,0,2,-1,-1,1,0,
-36,1,3,5,-1,-4,-2,1,2,-2,-1,1,0,
-41,0,3,5,-1,-2,-1,2,0,-3,1,1,0,
-44,-2,2,4,2,-2,-1,2,2,-1,0,1,0,
-46,-3,2,6,2,-2,-2,2,2,0,0,1,0,
-48,-4,0,3,0,-1,-2,1,2,0,1,1,1,
-51,-8,1,3,1,-1,-1,2,1,-1,0,0,1,
-53,-8,0,0,0,1,2,3,2,1,0,0,1,
-55,-4,2,2,3,0,2,3,2,0,0,1,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0,
-87,0,0,0,0,0,0,0,0,0,0,0,0

};

dl::audio::SpeechFeatureConfig wake_config;
void init_dl_mfcc(void){
    // 根据Python配置调整
    wake_config.sample_rate = 16000;
    wake_config.frame_length = 20;      // 20ms = 320 samples (与win_length=320一致)
    wake_config.frame_shift = 16;       // ❗ 重要：改为16ms = 256 samples (与hop_length=256一致)
    wake_config.num_mel_bins = 40;      // n_mels=40
    wake_config.num_ceps = 13;          // n_mfcc=13

    // 音频处理参数
    wake_config.preemphasis = 0.97f;    // 预加重系数一致
    wake_config.cepstral_lifter = 0.0f;

    // ❗ 重要：窗口函数改为 HAMMING
    // 查看 dl_speech_features.hpp 确定正确的枚举名
    wake_config.window_type = dl::audio::WinType::HAMMING;  // 改为汉明窗

    // 频率范围
    wake_config.low_freq = 0.0f;        // 低频
    wake_config.high_freq = 8000.0f;    // 高频

    // 对数处理
    wake_config.log_epsilon = 1e-6f;
    wake_config.use_log_fbank = 2;      // log_mels=True 对应此设置

    // FFT相关参数（需要检查是否支持设置）
    // 可能需要额外的FFT配置
    wake_config.raw_energy = false;
    wake_config.use_power = true;       // 使用功率谱
    wake_config.use_energy = false;
    wake_config.use_int16_fft = true;  // 使用浮点FFT
    wake_config.remove_dc_offset = false;
}

void test_model(void){
    DIR *dir;
    struct dirent *entry;
    char path[512];
    const char* base_path = "/flash";
    if (!(dir = opendir(base_path))) {
        ESP_LOGE(TAG, "无法打开目录: %s", base_path);
        return;
    }
    dl::Model *model =
        new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    model->test();
    model->profile();
    init_dl_mfcc();
    audio::MFCC wake_mfcc(wake_config);
    float c=0;
    float total=0;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        
        snprintf(path, sizeof(path), "%s/%s", base_path, entry->d_name);
        wav::WavHeader audio(path);
        if(audio.getFileStatus()==WAV_OPEND){
            fseek(audio.getFilePtr(), audio.getRawDataPosition(), SEEK_SET);
            int16_t* raw_data = (int16_t*)heap_caps_malloc(audio.getDataLength()/2 * sizeof(int16_t), MALLOC_CAP_32BIT);
            // float* float_raw_data = (float*)heap_caps_malloc(16000* sizeof(float), MALLOC_CAP_32BIT);
            // memset(float_raw_data, 0, 16000);
            size_t read_count = fread(raw_data, sizeof(int16_t), audio.getDataLength(), audio.getFilePtr());
            if (read_count != audio.getDataLength()/2) {
                printf("读取数据不完整，期望 %ld，实际 %zu\n", audio.getDataLength()/2, read_count);
            }
            // for(int j = 0; j < audio.getDataLength()/2; j++){
                
            //     float_raw_data[j] = (float)raw_data[j] / 32768.0f;
            //     //float_raw_data[j] = (float)raw_data[j];
            //     // if(j<100){
            //     //     if(j%10==0){
            //     //         printf("\n");
            //     //     }
            //     //     printf("%.4f ", float_raw_data[j]);
            //     // }
            //     //vTaskDelay(10 / portTICK_PERIOD_MS);
            // }
            float* mfcc = (float*)heap_caps_malloc(13*63* sizeof(float), MALLOC_CAP_32BIT);
            memset(mfcc, 0, 13*63*sizeof(int16_t));
            int mfcc_ptr, raw_ptr;
            mfcc_ptr=0;raw_ptr=0;
            int16_t* data_frame = (int16_t*)heap_caps_malloc(320* sizeof(int16_t), MALLOC_CAP_8BIT);
            while(mfcc_ptr < 63){
                int16_t prev = mfcc_ptr > 0? *(raw_data+raw_ptr-1) : 0;
                memcpy(data_frame, raw_data+raw_ptr, 320*sizeof(int16_t));
                wake_mfcc.process_frame(data_frame, 320, mfcc+mfcc_ptr*13, prev);
                //printf("\n");
                for(int i = 0; i < 13; i++){
                    //printf("%.2f ", *(mfcc+mfcc_ptr*13+i));
                    if(isnanf(*(mfcc+mfcc_ptr*13+i))||isinf(*(mfcc+mfcc_ptr*13+i))){
                        if(i==0) *(mfcc+mfcc_ptr*13+i)=0;
                        else    *(mfcc+mfcc_ptr*13+i)=0;
                    }else if(abs(*(mfcc+mfcc_ptr*13)) >128 ){
                        *(mfcc+mfcc_ptr*13+i)=0;
                    }
                    //printf("%.2f ", *(mfcc+mfcc_ptr*13+i));
                }
                mfcc_ptr++;
                raw_ptr+=256;
                memset(data_frame, 0, 320*sizeof(int16_t));
            }
            free(data_frame);
            // float* my_mfcc = extract_mfcc(float_raw_data, 16000, 16000, 320, 256, 512, 40, 13);
            // for(int i = 0; i < 63*13; i++){
            //     if(i%13==0) printf("\n");
            //     if(isnanf(my_mfcc[i])||my_mfcc[i]>200||my_mfcc[i]<-200){
            //         //my_mfcc[i]=0;
            //     }
            //     //my_mfcc[i]/=2;
            //     printf("%.2f ", my_mfcc[i]);
            // }
            // standardize_mfcc(mfcc, 63, 13);
            std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
            dl::TensorBase *model_input = model_inputs.begin()->second;
            std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
            dl::TensorBase *model_output = model_outputs.begin()->second;
            std::vector<int> input_shape = model_input->get_shape();
            std::vector<int> shape = {1,13,63};
            dl::TensorBase mfcc_tensor(input_shape, mfcc, 0, dl::DATA_TYPE_FLOAT, true);
            model_input->exponent=0;
            bool res = model_input->assign(&mfcc_tensor);
            // int8_t *input_ptr = (int8_t *)model_input->data;
            // input_ptr = right_data;
            //if(res == false)    ESP_LOGE(TAG, "量化失败");
            // for(int i = 0; i < model_input->size; i++){
            //     if(i%13==0) printf("\n");
            //     printf("%d ", (int)*((int8_t*)model_input->data+i));
            //     //printf("%d ", right_data[i]);
            // }
            printf("开始推理%s\n", entry->d_name);
            printf("shape : [%d, %d, %d], input_exponent : %d, output_exponent : %d\n", 
                    input_shape[0], input_shape[1], input_shape[2], model_input->exponent, model_output->exponent);
            model->run();
            int8_t output_value = 0;
            // 正确获取输出值
            if (model_output->get_dtype() == dl::DATA_TYPE_INT8) {
                output_value = model_output->get_element<int8_t>(0);
                float dequant_value = dl::dequantize<int8_t, float>(output_value, DL_SCALE(model_output->exponent));
                //float dequant_value = output_value/100.00f;
                ESP_LOGI(TAG, "Model output: quant=%d, dequant=%f", output_value, dequant_value);
            }
            
            float sigmoid = 1 / (1 + exp(-output_value));
            ESP_LOGI(TAG, "唤醒词概率: %.2f%%", sigmoid*100);
            total++;
            if(sigmoid>0.5) c++;
            free(raw_data);
            // free(float_raw_data);
            free(mfcc);
        }
        ESP_LOGI(TAG, "正样本成功率: %.2f%%", c/total*100);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        
    }
}


static void init_spiffs(void){
    esp_vfs_spiffs_conf_t conf = {
      .base_path = "/flash",
      .partition_label = "audio",
      .max_files = 7,
      .format_if_mount_failed = false
    };
    // Use settings defined above to initialize and mount SPIFFS filesystem.
    // Note: esp_vfs_spiffs_register is an all-in-one convenience function.
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to find SPIFFS partition");
        } else {
            ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }
        return;
    }
}

extern "C" void app_main(void)
{
    init_spiffs();
    //ring_buffer_test_simple();
    //debug_output_quantization();
    //test_wake_word_detection();
    
    //xTaskCreate(waker_rob, "inference_task", 16 * 1024, NULL, 5, NULL);
    // for(int i = 1;i <= 320000; i+=1){
    //     test_mfcc(16000);
    //     vTaskDelay(1000 / portTICK_PERIOD_MS);
    // }
    test_model();
    for (;;) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("Restarting now.\n");
    fflush(stdout);
    esp_restart();
}
