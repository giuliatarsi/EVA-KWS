// EVA: acquisizione comandata 1100ms dal microfono -> MFCC -> float32/int8.
// m = nuova registrazione; r = test sull’audio di riferimento. Nessun allarme/BLE.
#include <Arduino.h>
#include <ESP_I2S.h>
static I2SClass microphone;
static int16_t recorded_audio[17600];
// Guadagno 2: coerente con il registratore usato inizialmente per il dataset.
// Se il registratore effettivo usava un guadagno diverso, aggiornare questo valore.
constexpr int microphone_gain = 2;
#include <math.h>
#include <EVA_MFCC.h>
static float computed_mfcc[715];
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "reference_data.h"
#include "eva_float32_model.h"
#include "eva_int8_model.h"

constexpr size_t arena_size = 96 * 1024;
alignas(16) static uint8_t arena[arena_size];
static tflite::AllOpsResolver resolver;

bool run_model(const unsigned char* bytes, const char* name, float expected_eva) {
  Serial.printf("\n--- %s ---\n", name);
  const tflite::Model* model = tflite::GetModel(bytes);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERRORE: versione schema modello incompatibile.");
    return false;
  }
  // Un interprete per volta: i due modelli riutilizzano la stessa arena.
  tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("ERRORE AllocateTensors: leggere anche i messaggi della libreria.");
    return false;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  size_t count = 1;
  for (int i=0; i<input->dims->size; ++i) count *= input->dims->data[i];
  if (count != reference_mfcc_count || output->dims->size != 2 || output->dims->data[1] != 2) {
    Serial.println("ERRORE: dimensioni tensori inattese.");
    return false;
  }
  if (input->type != kTfLiteFloat32 && input->type != kTfLiteInt8) {
    Serial.println("ERRORE: tipo input non supportato."); return false;
  }
  if (input->type == kTfLiteInt8 && input->params.scale <= 0) {
    Serial.println("ERRORE: scala int8 non valida."); return false;
  }
  // TFLite Micro puo riutilizzare la memoria input come scratch.
  // Ricaricare SEMPRE l'input prima di ogni Invoke, anche nei benchmark.
  auto fill_input = [&]() {
    for (size_t i=0; i<count; ++i) {
      if (input->type == kTfLiteFloat32) input->data.f[i] = computed_mfcc[i];
      else {
        int q = (int)roundf(computed_mfcc[i]/input->params.scale) + input->params.zero_point;
        input->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
    }
  };
  Serial.printf("MFCC in ingresso: %.6f, %.6f, %.6f\n", computed_mfcc[0], computed_mfcc[1], computed_mfcc[2]);
  if (input->type == kTfLiteInt8)
    Serial.printf("Input int8: scale=%.9f, zero_point=%d\n", input->params.scale, input->params.zero_point);
  float first_scores[2] = {0,0};
  uint32_t total_us = 0;
  for (int repeat=0; repeat<21; ++repeat) {
    fill_input();
    uint32_t start = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
      Serial.println("ERRORE Invoke."); return false;
    }
    uint32_t duration = micros()-start;
    if (repeat>0) total_us += duration;
    if (repeat==0) {
      for (int j=0; j<2; ++j) {
        first_scores[j] = output->type==kTfLiteFloat32 ? output->data.f[j]
          : (output->data.int8[j]-output->params.zero_point)*output->params.scale;
      }
      Serial.printf("Prima inferenza: eva=%.8f, non_target=%.8f\n", first_scores[0], first_scores[1]);
    }
  }
  float elapsed = total_us/20.0f;
  float scores[2];
  for (int i=0; i<2; ++i) {
    if (output->type == kTfLiteFloat32) scores[i] = output->data.f[i];
    else if (output->type == kTfLiteInt8) scores[i] = (output->data.int8[i]-output->params.zero_point)*output->params.scale;
    else { Serial.println("ERRORE: tipo output non supportato."); return false; }
  }
  Serial.printf("eva = %.8f\nnon_target = %.8f\n", scores[0], scores[1]);
  if(expected_eva>=0) Serial.printf("Atteso sul PC: eva = %.8f; differenza = %.8f\n", expected_eva, fabsf(scores[0]-expected_eva));
  Serial.printf("Tempo medio CNN: %.1f us (esclude MFCC e microfono)\n", elapsed);
  bool consistent = isfinite(scores[0]) && (expected_eva<0 || fabsf(scores[0]-expected_eva)<0.005f)
      && fabsf(scores[0]-first_scores[0])<0.0001f;
  if(expected_eva>=0) Serial.println(consistent ? "Confronto PC: OK" : "Confronto PC: NON COINCIDE, test da diagnosticare");
  Serial.println(scores[0] > scores[1] ? "Classe: EVA" : "Classe: non_target");
  return consistent;
}

void test_both() {
  Serial.println("\nEVA MFCC TEST v1 - audio PCM -> MFCC -> float32/int8");
  uint32_t start=micros();
  int status=eva_mfcc(reference_audio,reference_audio_count,computed_mfcc);
  Serial.printf("MFCC status=%d, tempo=%.3f ms\n",status,(micros()-start)/1000.0f);
  if(status) return;
  float maximum=0; double square=0;
  for(size_t i=0;i<715;++i) {
    float error=fabsf(computed_mfcc[i]-reference_mfcc[i]);
    if(!isfinite(error)) {Serial.println("MFCC NON FINITE");return;}
    maximum=fmaxf(maximum,error);square+=error*error;
  }
  Serial.printf("Confronto MFCC: max errore=%.8f, RMSE=%.8f\n",maximum,sqrt(square/715));
  for(int i=0;i<6;++i) Serial.printf("MFCC[%d] scheda=%.8f PC=%.8f\n",i,computed_mfcc[i],reference_mfcc[i]);
  bool a=run_model(eva_float32_model,"FLOAT32 da MFCC sulla scheda",.99803442f);
  bool b=run_model(eva_int8_model,"INT8 da MFCC sulla scheda",.99609375f);
  Serial.println(a&&b ? "\nConfronto predizioni PC: OK. Invia r per ripetere." : "\nPredizioni diverse dal PC: verificare MFCC e output.");
}

bool capture_bytes(int16_t* destination,size_t samples) {
  size_t total=0;
  uint32_t start=millis();
  while(total<samples*2 && millis()-start<4000) {
    size_t chunk=samples*2-total;
    if(chunk>1024) chunk=1024;
    size_t got=microphone.readBytes((char*)destination+total,chunk);
    if(!got) {Serial.println("ERRORE: nessun dato dal microfono.");return false;}
    total+=got;
  }
  return total==samples*2;
}
void live_test() {
  Serial.println("\nPreparati: pronuncia EVA aiutami subito dopo PARLA ORA.");
  for(int i=3;i>0;--i) {Serial.println(i);delay(1000);}
  microphone.setPinsPdmRx(42,41);
  if(!microphone.begin(I2S_MODE_PDM_RX,16000,I2S_DATA_BIT_WIDTH_16BIT,I2S_SLOT_MODE_MONO)) {
    Serial.println("ERRORE inizializzazione microfono PDM.");return;
  }
  // Avvio e smaltimento dei campioni iniziali PRIMA del segnale di registrazione.
  int16_t warmup[512];
  for(int i=0;i<4;++i) {
    if(!capture_bytes(warmup,512)) {microphone.end();return;}
  }
  Serial.println("PARLA ORA - registrazione 1,1 secondi");
  bool ok=capture_bytes(recorded_audio,17600);
  microphone.end();
  if(!ok) {Serial.println("Registrazione incompleta: nessuna inferenza.");return;}
  Serial.println("Registrazione terminata. Elaborazione...");
  double energy=0,mean=0;int peak=0,clipped=0;
  for(size_t i=0;i<17600;++i) {
    int value=(int)recorded_audio[i]*microphone_gain;
    if(value>32767) {value=32767;++clipped;}
    if(value< -32768) {value=-32768;++clipped;}
    recorded_audio[i]=(int16_t)value;
    peak=max(peak,abs(value));mean+=value;energy+=(double)value*value;
  }
  Serial.printf("Audio: RMS=%.1f, picco=%d, media=%.1f, saturati=%d, gain=%d\n",
    sqrt(energy/17600),peak,mean/17600,clipped,microphone_gain);
  uint32_t start=micros();
  int ret=eva_mfcc(recorded_audio,17600,computed_mfcc);
  float dsp_ms=(micros()-start)/1000.0f;
  Serial.printf("MFCC status=%d, tempo=%.3f ms\n",ret,dsp_ms);
  if(ret) return;
  for(size_t i=0;i<715;++i) if(!isfinite(computed_mfcc[i])) {
    Serial.println("ERRORE: MFCC non finite.");return;
  }
  bool a=run_model(eva_float32_model,"FLOAT32 MICROFONO",-1);
  bool b=run_model(eva_int8_model,"INT8 MICROFONO",-1);
  Serial.println(a&&b ? "\nAcquisizione e inferenze completate." : "\nErrore di inferenza.");
  Serial.println("m = registra ancora; r = riferimento salvato. Questa prova NON ascolta continuamente.");
}
void setup() {
  Serial.begin(115200);
  unsigned long start=millis();
  while(!Serial && millis()-start<15000) delay(10);
  delay(500);
  Serial.println("EVA MICROPHONE TEST v1 - 16kHz PCM16 mono, finestra 1100ms");
  Serial.println("m = registra e confronta float32/int8; r = verifica audio di riferimento");
}
void loop() {
  if(Serial.available()) {
    char c=Serial.read();
    if(c=='m'||c=='M') live_test();
    else if(c=='r'||c=='R') test_both();
  }
  delay(10);
}
