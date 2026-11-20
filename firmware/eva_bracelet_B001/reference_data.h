#pragma once
#include <stdint.h>
#include <stddef.h>
constexpr size_t reference_fft_length = 512;
constexpr size_t reference_mel_filters = 32;
constexpr float reference_expected_eva_float32 = 1.000000000e+00f;
constexpr float reference_expected_eva_int8 = 9.960937500e-01f;
extern const int16_t reference_audio[];
extern const float reference_mfcc[];
constexpr size_t reference_audio_count = 17600;
constexpr size_t reference_mfcc_count = 715;
