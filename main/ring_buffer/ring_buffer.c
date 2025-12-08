#include "ring_buffer.h"
#include  "esp_log.h"
static uint16_t ringbuf_available_space(ring_buffer_t* rinbuf)
{   
    if(rinbuf->start_p <= rinbuf->end_p)
    {
        return rinbuf->buffer_len - (rinbuf->end_p - rinbuf->start_p);
    }
    else
    {
        return rinbuf->start_p - rinbuf->end_p;
    }
}
static uint16_t ringbuf_data_count(const ring_buffer_t* rinbuf)
{
    if (rinbuf->is_full) {
        return rinbuf->buffer_len;
    }
    
    if (rinbuf->end_p >= rinbuf->start_p) {
        return rinbuf->end_p - rinbuf->start_p;
    } else {
        return rinbuf->buffer_len - rinbuf->start_p + rinbuf->end_p;
    }
}
esp_err_t create_rinbuffer(ring_buffer_t* rinbuf, uint16_t length)
{
    //rinbuf = (ring_buffer_t*)heap_caps_malloc(sizeof(ring_buffer_t), MALLOC_CAP_8BIT|MALLOC_CAP_32BIT);
    rinbuf->buffer = (float*)heap_caps_malloc(length*sizeof(float), MALLOC_CAP_32BIT);
    rinbuf->buffer_len = length;
    //printf("len : %d\n", rinbuf->buffer_len);
    rinbuf->start_p = 0;
    rinbuf->end_p = -1;
    rinbuf->is_full = 0;
    memset(rinbuf->buffer, 0, sizeof(float)*length);
    if(rinbuf&&rinbuf->buffer&&rinbuf->buffer_len == length)
    {
        return ESP_OK;
    }else
    {
        return RINBUF_ERROR;
    }
}

esp_err_t delete_ringbuffer(ring_buffer_t* rinbuf){
    if(rinbuf)
    {
        if(rinbuf->buffer)
        {
            free(rinbuf->buffer);
        }
        return ESP_OK;
    }
    else return RINBUF_ERROR;
}

esp_err_t write_rinbuffer(ring_buffer_t* rinbuf, const float* data, uint16_t data_len)
{
    if (rinbuf == NULL || data == NULL || data_len == 0) {
        return RINBUF_ERROR;
    }
    // 如果写入数据超过缓冲区总大小，只保留最后 buffer_len 个数据
    if (data_len > rinbuf->buffer_len) {
        data = data + (data_len - rinbuf->buffer_len);  // 指向最后 buffer_len 个数据
        data_len = rinbuf->buffer_len;
    }
    
    uint16_t space_to_end = rinbuf->buffer_len - rinbuf->end_p;
    if (data_len <= space_to_end)
    {
        // 情况1：不需要环绕
        memcpy(rinbuf->buffer + rinbuf->end_p, data, data_len * sizeof(float));
        rinbuf->end_p += data_len;
    } else {
        // 情况2：需要环绕
        // 第一部分：拷贝到缓冲区末尾
        memcpy(rinbuf->buffer + rinbuf->end_p, data, space_to_end * sizeof(float));
        
        // 第二部分：环绕到缓冲区开头
        uint16_t remaining = data_len - space_to_end;
        memcpy(rinbuf->buffer, data + space_to_end, remaining * sizeof(float));
        
        rinbuf->end_p = remaining;
    }
    
    // 关键：处理覆盖情况，更新 start_p
    if(rinbuf->is_full == 1){
        rinbuf->start_p = rinbuf->end_p+1;
        if(rinbuf->start_p == rinbuf->buffer_len)   rinbuf->start_p = 0;
    }
    uint16_t num = abs(rinbuf->start_p - rinbuf->end_p)+1;
    if(num == rinbuf->buffer_len || num == 1)
    {
        rinbuf->is_full = 1;
    }
    return ESP_OK;
}

esp_err_t read_rinbuffer(const ring_buffer_t* rinbuf, float* data, uint16_t data_len)
{
    if (rinbuf == NULL || data == NULL || data_len == 0){
        return RINBUF_ERROR;
    }
    uint16_t used_space = ringbuf_data_count(rinbuf);
    if(data_len > used_space)
        return RINBUF_ERROR;
    //不存在环绕
    if(rinbuf->start_p < rinbuf->end_p)
        memcpy(data, rinbuf->buffer+rinbuf->start_p, data_len*sizeof(float));
    else{
    //存在环绕
        uint16_t remain = used_space - rinbuf->end_p;
        memcpy(data, rinbuf->buffer+rinbuf->start_p, remain*sizeof(float));
        memcpy(data+remain, rinbuf->buffer+rinbuf->start_p+remain, (data_len-remain)*sizeof(float));
    }
    return ESP_OK;
}

// 简单的环形缓冲区测试函数
void ring_buffer_test_simple(void)
{
    ESP_LOGI("TEST", "=== 开始环形缓冲区测试 ===");
    
    ring_buffer_t ringbuf;
    esp_err_t ret;
    
    // 测试1：创建缓冲区
    ESP_LOGI("TEST", "测试1:创建缓冲区");
    ret = create_rinbuffer(&ringbuf, 10);  // 使用小缓冲区便于调试
    if (ret != ESP_OK) {
        ESP_LOGE("TEST", "创建缓冲区失败");
        return;
    }
    ESP_LOGI("TEST", "缓冲区创建成功，大小: %d", ringbuf.buffer_len);
    
    // 测试2：写入少量数据
    ESP_LOGI("TEST", "测试2:写入少量数据");
    float data1[] = {1.1, 2.2, 3.3};
    ret = write_rinbuffer(&ringbuf, data1, 3);
    if (ret == ESP_OK) {
        ESP_LOGI("TEST", "写入3个数据成功");
        ESP_LOGI("TEST", "当前数据量: %d, 可用空间: %d", 
                 ringbuf_data_count(&ringbuf), ringbuf_available_space(&ringbuf));
    } else {
        ESP_LOGE("TEST", "写入数据失败: %d", ret);
    }
    
    // 测试3：读取数据
    ESP_LOGI("TEST", "测试3:读取数据");
    float read_data1[3] = {0};
    ret = read_rinbuffer(&ringbuf, read_data1, 3);
    if (ret == ESP_OK) {
        ESP_LOGI("TEST", "读取数据成功: [%.1f, %.1f, %.1f]", 
                 read_data1[0], read_data1[1], read_data1[2]);
        ESP_LOGI("TEST", "读取后数据量: %d, 可用空间: %d", 
                 ringbuf_data_count(&ringbuf), ringbuf_available_space(&ringbuf));
    } else {
        ESP_LOGE("TEST", "读取数据失败: %d", ret);
    }
    
    // 测试4：测试环绕写入
    ESP_LOGI("TEST", "测试4:测试环绕写入");
    float data2[] = {4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10};
    ret = write_rinbuffer(&ringbuf, data2, 7);
    if (ret == ESP_OK) {
        ESP_LOGI("TEST", "写入7个数据成功");
        ESP_LOGI("TEST", "当前数据量: %d, 可用空间: %d", 
                 ringbuf_data_count(&ringbuf), ringbuf_available_space(&ringbuf));
        ESP_LOGI("TEST", "start_p: %d, end_p: %d, is_full: %d", 
                 ringbuf.start_p, ringbuf.end_p, ringbuf.is_full);
    } else {
        ESP_LOGE("TEST", "写入数据失败: %d", ret);
    }
    
    // 测试5：读取环绕数据
    ESP_LOGI("TEST", "测试5:读取环绕数据");
    float read_data2[3] = {0};
    ret = read_rinbuffer(&ringbuf, read_data2, 3);
    if (ret == ESP_OK) {
        ESP_LOGI("TEST", "读取数据成功: [%.1f, %.1f, %.1f]", 
                 read_data2[0], read_data2[1], read_data2[2]);
        ESP_LOGI("TEST", "读取后数据量: %d, 可用空间: %d", 
                 ringbuf_data_count(&ringbuf), ringbuf_available_space(&ringbuf));
    } else {
        ESP_LOGE("TEST", "读取数据失败: %d", ret);
    }
    
    // 测试6：清理
    ESP_LOGI("TEST", "测试6:删除缓冲区");
    ret = delete_ringbuffer(&ringbuf);
    if (ret == ESP_OK) {
        ESP_LOGI("TEST", "缓冲区删除成功");
    } else {
        ESP_LOGE("TEST", "缓冲区删除失败: %d", ret);
    }
    
    ESP_LOGI("TEST", "=== 环形缓冲区测试完成 ===");
}