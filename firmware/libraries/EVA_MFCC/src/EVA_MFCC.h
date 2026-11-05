#pragma once
#include <stdint.h>
#include <stddef.h>
int eva_mfcc(const int16_t* audio, size_t length, float* output);
