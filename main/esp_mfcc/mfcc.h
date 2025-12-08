#ifndef _MFCC_H_
#define _MFCC_H_

#ifdef __cplusplus
extern "C" {
#endif
float* flow_extract_mfcc(const float* signal, int signal_len, int sampling_rate,
                   int frame_size, int hop_size, int n_fft, int n_filters, int n_mfcc);

float* extract_mfcc(const float* signal, int signal_len, int sampling_rate,
                   int frame_size, int hop_size, int n_fft, int n_filters, int n_mfcc);

void test_mfcc(int signal_len);

void free_mfcc(float* mfcc);

void analyze_mfcc_range(float* mfcc, int size, const char* label);

#ifdef __cplusplus
}
#endif

#endif