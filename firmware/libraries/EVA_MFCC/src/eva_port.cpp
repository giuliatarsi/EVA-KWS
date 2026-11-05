#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#ifdef ARDUINO
#include <Arduino.h>
#endif
void *ei_malloc(size_t n) {return malloc(n);}
void *ei_calloc(size_t n,size_t s) {return calloc(n,s);}
void ei_free(void* p) {free(p);}
void ei_printf(const char* format,...) {
  char buffer[256]; va_list args;va_start(args,format);vsnprintf(buffer,sizeof(buffer),format,args);va_end(args);
#ifdef ARDUINO
  Serial.print(buffer);
#else
  printf("%s",buffer);
#endif
}
void ei_printf_float(float f) {ei_printf("%f",f);}
