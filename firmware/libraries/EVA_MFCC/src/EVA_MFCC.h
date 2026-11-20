#pragma once
#include <stdint.h>
#include <stddef.h>
int eva_mfcc(const int16_t* audio, size_t length, float* output, size_t fft_length = 512, size_t mel_filters = 40, size_t coefficients = 13);
