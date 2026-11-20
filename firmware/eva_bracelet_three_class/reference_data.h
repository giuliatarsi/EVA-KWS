#pragma once
#include <stdint.h>
#include <stddef.h>
constexpr size_t reference_fft_length = 512;
constexpr size_t reference_mel_filters = 40;
constexpr float reference_expected_eva_float32 = 9.834386706e-01f;
constexpr float reference_expected_eva_int8 = 9.843750000e-01f;
extern const int16_t reference_audio[];
extern const float reference_mfcc[];
constexpr size_t reference_audio_count = 17600;
constexpr size_t reference_mfcc_count = 715;
constexpr float reference_expected_noise_int8 = 0.000000000e+00f;
constexpr float reference_expected_unknown_int8 = 1.562500000e-02f;
