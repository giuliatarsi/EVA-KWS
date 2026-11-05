// EVA: prova dei modelli float32 e int8 sulle stesse MFCC precalcolate.
// Nessuna acquisizione microfono o estrazione MFCC in questa prima prova.
#include <Arduino.h>
#include <math.h>
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
      if (input->type == kTfLiteFloat32) input->data.f[i] = reference_mfcc[i];
      else {
        int q = (int)roundf(reference_mfcc[i]/input->params.scale) + input->params.zero_point;
        input->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
    }
  };
  Serial.printf("MFCC riferimento: %.6f, %.6f, %.6f\n", reference_mfcc[0], reference_mfcc[1], reference_mfcc[2]);
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
  Serial.printf("Atteso sul PC: eva = %.8f; differenza = %.8f\n", expected_eva, fabsf(scores[0]-expected_eva));
  Serial.printf("Tempo medio CNN: %.1f us (esclude MFCC e microfono)\n", elapsed);
  bool consistent = isfinite(scores[0]) && fabsf(scores[0]-expected_eva)<0.005f
      && fabsf(scores[0]-first_scores[0])<0.0001f;
  Serial.println(consistent ? "Confronto PC: OK" : "Confronto PC: NON COINCIDE, test da diagnosticare");
  Serial.println(scores[0] > scores[1] ? "Classe: EVA" : "Classe: non_target");
  return consistent;
}

void test_both() {
  Serial.println("\nEVA TEST v2 - input ricaricato a ogni inferenza - 55 x 13 MFCC");
  bool a = run_model(eva_float32_model, "FLOAT32", 0.99803442f);
  bool b = run_model(eva_int8_model, "INT8", 0.99609375f);
  Serial.println(a && b ? "\nTest completati. Invia r per ripetere." : "\nTest incompleto: verificare gli errori sopra.");
}
void setup() {
  Serial.begin(115200);
  unsigned long start = millis();
  while (!Serial && millis()-start < 15000) delay(10);
  delay(1000);
  test_both();
}
void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c=='r' || c=='R') test_both();
  }
  delay(10);
}
