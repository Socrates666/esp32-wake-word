#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

#define RINBUF_ERROR -1


typedef struct
{ 
    uint16_t buffer_len;
    int start_p;
    int end_p;
    uint8_t is_full;
    float* buffer;
}ring_buffer_t;


esp_err_t create_rinbuffer(ring_buffer_t* rinbuf, uint16_t buffer_len);

esp_err_t delete_ringbuffer(ring_buffer_t* rinbuf);

esp_err_t write_rinbuffer(ring_buffer_t* rinbuf, const float* data, uint16_t data_len);

esp_err_t read_rinbuffer(const ring_buffer_t* rinbuf, float* data, uint16_t data_len);

void ring_buffer_test_simple(void);

#ifdef __cplusplus
    }
#endif
#endif