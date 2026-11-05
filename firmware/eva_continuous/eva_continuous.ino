// EVA: ascolto continuo con acquisizione PDM in task separato.
// Finestre complete 1100ms, aggiornamento nominale 500ms; entrambi i modelli.
#include <Arduino.h>
#include <ESP_I2S.h>
#include <EVA_MFCC.h>
#include <math.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "eva_float32_model.h"
#include "eva_int8_model.h"
#include "reference_data.h"

constexpr size_t window_samples=17600, hop_samples=8000; // 1100ms / 500ms @16kHz
constexpr int microphone_gain=2;
constexpr size_t arena_size=96*1024;
alignas(16) static uint8_t arena[arena_size];
static int16_t ring_audio[window_samples], window_audio[window_samples];
static float features[715];
static I2SClass microphone;
static tflite::AllOpsResolver resolver;
static portMUX_TYPE ring_lock=portMUX_INITIALIZER_UNLOCKED;
static size_t write_position=0,valid_samples=0;
static uint64_t total_samples=0;
static uint32_t read_errors=0;
static bool listening=true;
static uint64_t last_end=0;
static uint32_t last_errors=0;

void acquire_audio(void*) {
  int16_t chunk[320];
  // Scarica i campioni iniziali prima di considerare valida la prima finestra.
  for(int i=0;i<7;++i) microphone.readBytes((char*)chunk,sizeof(chunk));
  for(;;) {
    size_t got=microphone.readBytes((char*)chunk,sizeof(chunk));
    if(got!=sizeof(chunk)) {
      // Una lettura incompleta invalida la finestra; non unire audio attraverso un buco.
      portENTER_CRITICAL(&ring_lock);
      ++read_errors;valid_samples=0;
      portEXIT_CRITICAL(&ring_lock);
      vTaskDelay(1);continue;
    }
    for(size_t i=0;i<320;++i) {
      int value=(int)chunk[i]*microphone_gain;
      chunk[i]=(int16_t)(value>32767 ? 32767 : (value< -32768 ? -32768 : value));
    }
    portENTER_CRITICAL(&ring_lock);
    for(size_t i=0;i<320;++i) {
      ring_audio[write_position]=chunk[i];
      write_position=(write_position+1)%window_samples;
    }
    total_samples+=320;
    valid_samples=min(window_samples,valid_samples+320);
    portEXIT_CRITICAL(&ring_lock);
  }
}

bool snapshot(uint64_t &end_sample,uint32_t &errors) {
  bool ready=false;
  portENTER_CRITICAL(&ring_lock);
  end_sample=total_samples;errors=read_errors;
  if(valid_samples==window_samples && (last_end==0 || end_sample-last_end>=hop_samples)) {
    // Snapshot coerente: il task audio non puo modificare la finestra durante la copia.
    size_t first=window_samples-write_position;
    memcpy(window_audio,ring_audio+write_position,first*sizeof(int16_t));
    memcpy(window_audio+first,ring_audio,write_position*sizeof(int16_t));
    ready=true;
  }
  portEXIT_CRITICAL(&ring_lock);
  return ready;
}

bool infer(const unsigned char* bytes,float &eva,float &negative,float &duration_ms) {
  const tflite::Model* model=tflite::GetModel(bytes);
  if(model->version()!=TFLITE_SCHEMA_VERSION) return false;
  tflite::MicroInterpreter interpreter(model,resolver,arena,arena_size);
  if(interpreter.AllocateTensors()!=kTfLiteOk) return false;
  TfLiteTensor *input=interpreter.input(0),*output=interpreter.output(0);
  size_t count=1;
  for(int i=0;i<input->dims->size;++i) count*=input->dims->data[i];
  if(count!=715 || output->dims->size!=2 || output->dims->data[1]!=2) return false;
  if(input->type==kTfLiteInt8 && input->params.scale<=0) return false;
  for(size_t i=0;i<715;++i) {
    if(input->type==kTfLiteFloat32) input->data.f[i]=features[i];
    else if(input->type==kTfLiteInt8) {
      int value=(int)roundf(features[i]/input->params.scale)+input->params.zero_point;
      input->data.int8[i]=(int8_t)(value>127 ? 127 : (value< -128 ? -128 : value));
    } else return false;
  }
  uint32_t start=micros();
  if(interpreter.Invoke()!=kTfLiteOk) return false;
  duration_ms=(micros()-start)/1000.f;
  float scores[2];
  for(int i=0;i<2;++i) {
    if(output->type==kTfLiteFloat32) scores[i]=output->data.f[i];
    else if(output->type==kTfLiteInt8) scores[i]=(output->data.int8[i]-output->params.zero_point)*output->params.scale;
    else return false;
  }
  eva=scores[0];negative=scores[1];
  return isfinite(eva)&&isfinite(negative);
}

void analyze(bool reference,uint64_t end_sample,uint32_t errors,uint64_t advance) {
  uint32_t all_start=micros();
  uint32_t dsp_start=micros();
  int ret=eva_mfcc(reference ? reference_audio : window_audio,window_samples,features);
  float dsp_ms=(micros()-dsp_start)/1000.f;
  if(ret) {Serial.printf("ERRORE MFCC: %d\n",ret);return;}
  for(size_t i=0;i<715;++i) if(!isfinite(features[i])) {Serial.println("ERRORE MFCC non finite");return;}
  float f,nf,tf,q,nq,tq;
  if(!infer(eva_float32_model,f,nf,tf) || !infer(eva_int8_model,q,nq,tq)) {
    Serial.println("ERRORE inferenza");return;
  }
  float elapsed=(micros()-all_start)/1000.f;
  if(reference) {
    float maximum=0;
    for(size_t i=0;i<715;++i) maximum=fmaxf(maximum,fabsf(features[i]-reference_mfcc[i]));
    bool ok=maximum<.001f && fabsf(f-.99803442f)<.005f && fabsf(q-.99609375f)<.005f;
    Serial.printf("RIFERIMENTO %s: MFCC maxerr=%.7f, eva float=%.7f int8=%.7f\n",ok?"OK":"NON COINCIDE",maximum,f,q);
    return;
  }
  double energy=0;int clipped=0;
  for(size_t i=0;i<window_samples;++i) {
    energy+=(double)window_audio[i]*window_audio[i];
    if(window_audio[i]==32767 || window_audio[i]==-32768) ++clipped;
  }
  // Intervallo dei campioni (non timestamp del termine di elaborazione).
  Serial.printf("audio %.2f-%.2fs | EVA float=%.4f int8=%.4f | classi=%s/%s | RMS=%.0f clip=%d | MFCC=%.1fms CNN=%.1f/%.1fms totale=%.1fms | passo=%.0fms errori_audio=%lu\n",
    (end_sample-window_samples)/16000.0,end_sample/16000.0,f,q,
    f>nf?"eva":"non_target",q>nq?"eva":"non_target",sqrt(energy/window_samples),clipped,
    dsp_ms,tf,tq,elapsed,advance/16.0,(unsigned long)errors);
  if(advance>hop_samples+320) Serial.println("NOTA: passo oltre 500ms; analizzata l'ultima finestra disponibile.");
}

void setup() {
  Serial.begin(115200);
  uint32_t start=millis();while(!Serial && millis()-start<15000) delay(10);
  delay(300);
  Serial.println("EVA CONTINUOUS v1 - audio 1100ms / passo nominale 500ms; float32 + int8");
  Serial.println("p=pausa analisi, s=riprendi, r=test riferimento. Nessun invio di allarme.");
  analyze(true,window_samples,0,0);
  microphone.setPinsPdmRx(42,41);
  if(!microphone.begin(I2S_MODE_PDM_RX,16000,I2S_DATA_BIT_WIDTH_16BIT,I2S_SLOT_MODE_MONO)) {
    Serial.println("ERRORE microfono PDM");while(true) delay(1000);
  }
  if(xTaskCreatePinnedToCore(acquire_audio,"eva_audio",4096,nullptr,2,nullptr,0)!=pdPASS) {
    Serial.println("ERRORE task audio");microphone.end();while(true) delay(1000);
  }
  Serial.println("ASCOLTO ATTIVO: puoi parlare liberamente.");
}
void loop() {
  while(Serial.available()) {
    char c=Serial.read();
    if(c=='p'||c=='P') {listening=false;Serial.println("Analisi in pausa; acquisizione ancora attiva.");}
    if(c=='s'||c=='S') {listening=true;last_end=0;Serial.println("ASCOLTO ATTIVO");}
    if(c=='r'||c=='R') analyze(true,window_samples,0,0);
  }
  if(!listening) {delay(10);return;}
  uint64_t end_sample;uint32_t errors;
  bool ready=snapshot(end_sample,errors);
  if(errors!=last_errors) {
    Serial.printf("ATTENZIONE: letture audio incomplete=%lu. Attendo una finestra integra.\n",(unsigned long)errors);
    last_errors=errors;
  }
  if(ready) {
    uint64_t advance=last_end ? end_sample-last_end : 0;
    last_end=end_sample;
    analyze(false,end_sample,errors,advance);
  } else delay(5);
}
