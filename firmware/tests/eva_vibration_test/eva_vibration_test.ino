// EVA USB test: continuous INT8 keyword spotting and vibration.
#include <Arduino.h>
#include <ESP_I2S.h>
#include <EVA_MFCC.h>
#include <math.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelli neurali e dati di riferimento (da File A)
#include "eva_int8_model.h"
#include "reference_data.h"

// --- Soglia Riconoscimento Vocale ---
#define CONFIDENCE_THRESHOLD 0.80f     // Soglia di attivazione allarme (80%)

// --- Pin Hardware Motore e Pulsante (XIAO ESP32-S3) ---
#define VIBRATION_PIN        D1        // GPIO2 -> IN del driver Adafruit 5648

// --- Pin Microfono PDM Integrato (XIAO Sense) ---
// NOTA RISOLUZIONE CONFLITTO: Entrambi i file usano i pin 42 (CLK) e 41 (DATA).
// È stata mantenuta un'unica istanza della periferica I2S condivisa.
#define I2S_MIC_CLOCK_PIN    42
#define I2S_MIC_DATA_PIN     41

constexpr uint32_t vibration_ms = 3000;
constexpr uint32_t pulse_ms = 200;

// --- Parametri Audio e TensorFlow Lite (da File A) ---
constexpr size_t window_samples = 17600; // 1100ms @ 16kHz
constexpr size_t hop_samples    = 8000;  // 500ms  @ 16kHz
constexpr int microphone_gain   = 2;
constexpr size_t arena_size     = 96 * 1024;

// ==========================================================================
// 2. VARIABILI GLOBALI, BUFFER E MACCHINA A STATI
// ==========================================================================

// --- Buffer e Strutture TFLite / Audio (File A) ---
alignas(16) static uint8_t arena[arena_size];
static int16_t ring_audio[window_samples], window_audio[window_samples];
static float features[715];
static I2SClass microphone; // Unica istanza I2S (sostituisce "I2S" del File B)
static tflite::AllOpsResolver resolver;
static portMUX_TYPE ring_lock = portMUX_INITIALIZER_UNLOCKED;

static size_t write_position = 0, valid_samples = 0;
static uint64_t total_samples = 0;
static uint32_t read_errors = 0;
static uint64_t last_end = 0;

static bool vibrating = false;
static uint32_t vibration_start = 0;

// Task FreeRTOS ancorato al Core 0 per campionamento PDM continuo
void acquire_audio(void*) {
  int16_t chunk[320];
  // Scarica i campioni iniziali prima di considerare valida la prima finestra
  for (int i = 0; i < 7; ++i) {
    microphone.readBytes((char*)chunk, sizeof(chunk));
  }
  for (;;) {
    size_t got = microphone.readBytes((char*)chunk, sizeof(chunk));
    if (got != sizeof(chunk)) {
      // Una lettura incompleta invalida la finestra
      portENTER_CRITICAL(&ring_lock);
      ++read_errors;
      valid_samples = 0;
      portEXIT_CRITICAL(&ring_lock);
      vTaskDelay(1);
      continue;
    }
    for (size_t i = 0; i < 320; ++i) {
      int value = (int)chunk[i] * microphone_gain;
      chunk[i] = (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
    }
    portENTER_CRITICAL(&ring_lock);
    for (size_t i = 0; i < 320; ++i) {
      ring_audio[write_position] = chunk[i];
      write_position = (write_position + 1) % window_samples;
    }
    total_samples += 320;
    valid_samples = min(window_samples, valid_samples + 320);
    portEXIT_CRITICAL(&ring_lock);
  }
}

// Estrae una finestra coerente di 1100ms dal ring buffer ogni 500ms
bool snapshot(uint64_t &end_sample, uint32_t &errors) {
  bool ready = false;
  portENTER_CRITICAL(&ring_lock);
  end_sample = total_samples;
  errors = read_errors;
  if (valid_samples == window_samples && (last_end == 0 || end_sample - last_end >= hop_samples)) {
    size_t first = window_samples - write_position;
    memcpy(window_audio, ring_audio + write_position, first * sizeof(int16_t));
    memcpy(window_audio + first, ring_audio, write_position * sizeof(int16_t));
    ready = true;
  }
  portEXIT_CRITICAL(&ring_lock);
  return ready;
}

// Modello INT8 locale di prova, esportato con MFCC a 256 punti FFT.
bool infer(float &eva, float &negative, float &duration_ms) {
  const tflite::Model* model = tflite::GetModel(eva_int8_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
  if (interpreter.AllocateTensors() != kTfLiteOk) return false;

  TfLiteTensor *input = interpreter.input(0), *output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->params.scale <= 0 || output->params.scale <= 0 ||
      input->dims->size != 3 || input->dims->data[0] != 1 ||
      input->dims->data[1] != 55 || input->dims->data[2] != 13 ||
      output->dims->size != 2 || output->dims->data[0] != 1 ||
      output->dims->data[1] != 2) return false;

  for (size_t i = 0; i < 715; ++i) {
    const int value = (int)roundf(features[i] / input->params.scale) + input->params.zero_point;
    input->data.int8[i] = (int8_t)(value > 127 ? 127 : (value < -128 ? -128 : value));
  }

  const uint32_t start = micros();
  if (interpreter.Invoke() != kTfLiteOk) return false;
  duration_ms = (micros() - start) / 1000.f;

  float scores[2];
  for (int i = 0; i < 2; ++i) {
    scores[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
  }
  eva = scores[0];
  negative = scores[1];
  return isfinite(eva) && isfinite(negative);
}

// Calcola le MFCC, esegue il modello INT8 e restituisce true se la parola chiave supera la soglia
bool analyze(bool reference, uint64_t end_sample, uint32_t errors, uint64_t advance, float &out_confidence) {
  out_confidence = 0.0f;
  uint32_t all_start = micros();
  uint32_t dsp_start = micros();

  int ret = eva_mfcc(reference ? reference_audio : window_audio, window_samples, features, reference_fft_length);
  float dsp_ms = (micros() - dsp_start) / 1000.f;
  if (ret) {
    Serial.printf("ERRORE MFCC: %d\n", ret);
    return false;
  }
  for (size_t i = 0; i < 715; ++i) {
    if (!isfinite(features[i])) {
      Serial.println("ERRORE MFCC non finite");
      return false;
    }
  }

  float q, nq, tq;
  if (!infer(q, nq, tq)) {
    Serial.println("ERRORE inferenza");
    return false;
  }

  float elapsed = (micros() - all_start) / 1000.f;
  if (reference) {
    float maximum = 0;
    for (size_t i = 0; i < 715; ++i) {
      maximum = fmaxf(maximum, fabsf(features[i] - reference_mfcc[i]));
    }
    bool ok = maximum < .001f && fabsf(q - .99609375f) < .005f;
    Serial.printf("RIFERIMENTO %s: MFCC maxerr=%.7f, eva int8=%.7f\n", ok ? "OK" : "NON COINCIDE", maximum, q);
    return false;
  }

  double energy = 0;
  int clipped = 0;
  for (size_t i = 0; i < window_samples; ++i) {
    energy += (double)window_audio[i] * window_audio[i];
    if (window_audio[i] == 32767 || window_audio[i] == -32768) ++clipped;
  }

  Serial.printf("audio %.2f-%.2fs | EVA int8=%.4f | classe=%s | RMS=%.0f clip=%d | MFCC=%.1fms CNN=%.1fms totale=%.1fms | passo=%.0fms errori_audio=%lu\n",
    (end_sample - window_samples) / 16000.0, end_sample / 16000.0, q,
    q > nq ? "eva" : "non_target", sqrt(energy / window_samples), clipped,
    dsp_ms, tq, elapsed, advance / 16.0, (unsigned long)errors);

  if (advance > hop_samples + 320) {
    Serial.println("NOTA: passo oltre 500ms; analizzata l'ultima finestra disponibile.");
  }

  // Valutazione attivazione allarme: verifica se la classe "eva" vince e supera CONFIDENCE_THRESHOLD
  out_confidence = q;
  return q > nq && q >= CONFIDENCE_THRESHOLD;
}

// ==========================================================================
// 5. SETUP UNIFICATO
// ==========================================================================
void setup() {
  // 1. Inizializzazione Seriale (Unica chiamata non bloccante all'avvio)
  Serial.begin(115200);
  uint32_t start = millis();
  while (!Serial && millis() - start < 5000) {
    // Attesa massima 5s per apertura monitor seriale USB nativo
  }

  // 2. Inizializzazione GPIO Motore e Pulsante (da File B)
  pinMode(VIBRATION_PIN, OUTPUT);
  digitalWrite(VIBRATION_PIN, LOW); // Motore spento all'avvio

  // 4. Test di autodiagnostica modelli MFCC + TFLite (da File A)
  Serial.println("EVA VOICE VIBRATION TEST - audio 1100ms / passo 500ms; INT8");
  Serial.println("Di’ EVA aiutami per avviare la vibrazione.");
  float dummy_conf = 0.0f;
  analyze(true, window_samples, 0, 0, dummy_conf);

  // 5. Inizializzazione Microfono PDM e Task su Core 0 (Unificato File A + File B)
  microphone.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  if (!microphone.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("ERRORE CRITICO: Impossibile avviare il microfono PDM!");
    while (true) { vTaskDelay(pdMS_TO_TICKS(1000)); }
  }

  if (xTaskCreatePinnedToCore(acquire_audio, "eva_audio", 4096, nullptr, 2, nullptr, 0) != pdPASS) {
    Serial.println("ERRORE CRITICO: Impossibile avviare il task audio FreeRTOS!");
    microphone.end();
    while (true) { vTaskDelay(pdMS_TO_TICKS(1000)); }
  }

  Serial.println("🎙️ ASCOLTO ATTIVO: Sistema pronto.");
}

// ==========================================================================
// 6. LOOP PRINCIPALE (Macchina a Stati Non Bloccante + Comandi Seriali)
// ==========================================================================

void loop() {
  if (vibrating) {
    const uint32_t elapsed = millis() - vibration_start;
    if (elapsed < vibration_ms) {
      digitalWrite(VIBRATION_PIN, (elapsed / pulse_ms) % 2 == 0 ? HIGH : LOW);
      delay(1);
      return;
    }

    digitalWrite(VIBRATION_PIN, LOW);
    vibrating = false;
    // Require a fresh window so vibration noise cannot trigger the next detection.
    portENTER_CRITICAL(&ring_lock);
    valid_samples = 0;
    last_end = total_samples;
    portEXIT_CRITICAL(&ring_lock);
    Serial.println("Vibrazione terminata. Ascolto attivo.");
    return;
  }

  uint64_t end_sample;
  uint32_t errors;
  if (!snapshot(end_sample, errors)) {
    delay(1);
    return;
  }

  const uint64_t advance = last_end ? end_sample - last_end : 0;
  last_end = end_sample;
  float confidence = 0.0f;
  if (analyze(false, end_sample, errors, advance, confidence)) {
    digitalWrite(VIBRATION_PIN, HIGH);
    vibration_start = millis();
    vibrating = true;
    Serial.printf("EVA aiutami rilevata (%.2f). Vibrazione per 3 s.\n", confidence);
  }
}
