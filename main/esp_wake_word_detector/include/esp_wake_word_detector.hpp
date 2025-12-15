#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

#include "audio/speech_features/dl_mfcc.hpp"
#include "audio/speech_features/dl_speech_features.hpp"
#include "audio/common/dl_audio_common.hpp"

/*
提供一个结构体用于注册模型和音频接口
*/

typedef enum {
    WAKE_WORD_DETECTED,
    WAKE_WORD_TIMEOUT,
    WAKE_WORD_ERROR
} wake_word_event_t;

typedef void (*wake_event_callback_t)(wake_word_event_t event, void* message);
typedef esp_err_t (*read_mic_fn)(void *audio_buffer, size_t len, size_t *bytes_read, uint32_t timeout_ms);


typedef struct {
    const char* model_path;
    QueueHandle_t event_queue;
    wake_event_callback_t callback;
    read_mic_fn read_mic;
} wakeWord_detection_config_t;

esp_err_t wakeWord_detection_open(wakeWord_detection_config_t* wake_config);

void wakeWord_detection_close(void);