#define EVA_MFCC_EXACT_MEL_BINS 1
#include <cmath>
#define EIDSP_QUANTIZE_FILTERBANK 0
#include "EVA_MFCC.h"
#include "edge-impulse-sdk/dsp/speechpy/feature.hpp"
static const int16_t* samples;
static size_t sample_count;
static int read_preemphasized(size_t offset, size_t length, float* out) {
  if (offset+length>sample_count) return -1;
  for (size_t i=0;i<length;++i) {
    size_t k=offset+i;
    out[i]=(float)samples[k]-.98f*(float)samples[k ? k-1 : sample_count-1];
  }
  return 0;
}
int eva_mfcc(const int16_t* audio, size_t length, float* output, size_t fft_length, size_t mel_filters, size_t coefficients) {
  if (length!=17600) return -100;
  if (fft_length!=256 && fft_length!=512) return -101;
  if (mel_filters!=32 && mel_filters!=40) return -102;
  if (coefficients != 13 && coefficients != 20) return -103;
  samples=audio; sample_count=length;
  ei::signal_t signal; signal.total_length=length; signal.get_data=read_preemphasized;
  ei::matrix_t matrix(55,coefficients,output);
  int ret=ei::speechpy::feature::mfcc(&matrix,&signal,16000,.020f,.020f,coefficients,mel_filters,fft_length,0,0,true,4);
  if(ret) return ret;
  return ei::speechpy::processing::cmvnw(&matrix,101,true,false);
}
