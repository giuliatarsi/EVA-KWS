// Candidate B001; configuration and source run in DEPLOYMENT.json.
// ==========================================================================
// PROGETTO EVA - FIRMWARE UNIFICATO (Ascolto Continuo + TFLite + BLE + FSM)
// Target: Seeed Studio XIAO ESP32-S3 Sense (Core Arduino ESP32 v3.0.x)
// ==========================================================================

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

// Librerie Bluetooth Low Energy (da File B)
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ==========================================================================
// 1. CONFIGURAZIONE PIN, COSTANTI E PARAMETRI
// ==========================================================================

// --- Soglia Riconoscimento Vocale ---
#define CONFIDENCE_THRESHOLD 0.63671875f // EVA F1 maximum on INT8 validation; see DEPLOYMENT.json.

// --- Pin Hardware Motore e Pulsante (XIAO ESP32-S3) ---
#define VIBRATION_PIN        D1        // GPIO2 -> IN del driver Adafruit 5648
#define CANCEL_BUTTON_PIN    D2        // GPIO3 -> Pulsante tra D2 e GND (INPUT_PULLUP)

// --- Pin Microfono PDM Integrato (XIAO Sense) ---
// NOTA RISOLUZIONE CONFLITTO: Entrambi i file usano i pin 42 (CLK) e 41 (DATA).
// È stata mantenuta un'unica istanza della periferica I2S condivisa.
#define I2S_MIC_CLOCK_PIN    42
#define I2S_MIC_DATA_PIN     41

// --- Tempistiche Macchina a Stati (Non bloccanti) ---
#define CANCEL_WINDOW_MS     8000      // 8 secondi per annullare il falso positivo
#define PULSE_INTERVAL_MS    500       // Impulsi: 500ms ON /500ms OFF

// --- UUID Bluetooth Low Energy (Condivisi con App Flutter) ---
#define SERVICE_UUID         "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID  "beb5483e-36e1-4688-b7f5-ea07361b26a8"

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
static bool serial_listening_enabled = true; // Rinominato da 'listening' per chiarezza con FSM
static uint64_t last_end = 0;
static uint32_t last_errors = 0;

// --- Variabili BLE e Gestione Stati (File B) ---
BLECharacteristic *pCharacteristic = nullptr;
bool deviceConnected = false;

enum SystemState {
  STATE_LISTENING,    // Ascolta e classifica le finestre audio
  STATE_PRE_ALARM,    // Vibra a impulsi per 5s e attende eventuale annullamento
  STATE_SENDING_SOS   // Tempo scaduto: invia la notifica BLE "SOS" allo smartphone
};

SystemState currentState = STATE_LISTENING;

// Timer non bloccanti basati su millis()
unsigned long preAlarmStartTime = 0;
unsigned long lastVibrationToggleTime = 0;
bool motorState = false;

// ==========================================================================
// 3. CALLBACK E FUNZIONI BLUETOOTH LOW ENERGY (da File B)
// ==========================================================================
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("📱 Smartphone connesso via BLE!");
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("⚠️ Smartphone disconnesso. Riavvio advertising BLE...");
    BLEDevice::startAdvertising();
  }
};

void triggerEmergency() {
  if (deviceConnected) {
    String alertMsg = "SOS";
    pCharacteristic->setValue(alertMsg.c_str());
    pCharacteristic->notify(); // Invia la notifica immediata all'app Flutter
    Serial.println("🚨 ALLARME SOS INVIATO ALLO SMARTPHONE VIA BLE!");
  } else {
    Serial.println("⚠️ ALLARME SCATTATO, ma nessun telefono connesso via BLE.");
  }
}

// ==========================================================================
// 4. ACQUISIZIONE AUDIO IN BACKGROUND E INFERENZA AI (da File A)
// ==========================================================================

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

// B001 INT8: CNN1D 8-16-32, kernel7; MFCC13, Mel32, FFT512.
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
static bool reference_ok = false;

bool analyze(bool reference, uint64_t end_sample, uint32_t errors, uint64_t advance, float &out_confidence) {
  out_confidence = 0.0f;
  uint32_t all_start = micros();
  uint32_t dsp_start = micros();

  int ret = eva_mfcc(reference ? reference_audio : window_audio, window_samples, features, reference_fft_length, reference_mel_filters, 13);
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
    bool ok = maximum < .001f && fabsf(q - reference_expected_eva_int8) < .005f;
    reference_ok = ok;
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
  pinMode(CANCEL_BUTTON_PIN, INPUT_PULLUP); // LOW = premuto, HIGH = riposo

  // 3. Inizializzazione Stack e Server BLE (da File B)
  BLEDevice::init("SOS_Bracelet_S3");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  BLEDevice::getAdvertising()->start();
  Serial.println("Bluetooth attivo (SOS_Bracelet_S3). In attesa di connessione...");

  // 4. Test di autodiagnostica modelli MFCC + TFLite (da File A)
  Serial.println("EVA B001 INT8 + BLE - audio 1100ms / passo 500ms; INT8");
  Serial.println("Comandi seriali: p=pausa analisi, s=riprendi, r=test riferimento.");
  float dummy_conf = 0.0f;
  analyze(true, window_samples, 0, 0, dummy_conf);
  if (!reference_ok) {
    Serial.println("STOP: riferimento non valido, ascolto non avviato.");
    while (true) delay(1000);
  }

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
  unsigned long currentMillis = millis();

  // --- Gestione Comandi Seriali di Debug (da File A, sempre accessibili) ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'p' || c == 'P') {
      serial_listening_enabled = false;
      Serial.println("⏸️ Analisi in pausa; acquisizione in background ancora attiva.");
    }
    if (c == 's' || c == 'S') {
      serial_listening_enabled = true;
      last_end = 0;
      Serial.println("▶️ ASCOLTO ATTIVO");
    }
    if (c == 'r' || c == 'R') {
      float dummy_conf = 0.0f;
      analyze(true, window_samples, 0, 0, dummy_conf);
  if (!reference_ok) {
    Serial.println("STOP: riferimento non valido, ascolto non avviato.");
    while (true) delay(1000);
  }
    }
  }

  // --- Macchina a Stati Finiti del Braccialetto (Unificazione Flussi A e B) ---
  switch (currentState) {

    // ----------------------------------------------------------------------
    // STATO 1: ASCOLTO E RICONOSCIMENTO VOCALE CONTINUO
    // ----------------------------------------------------------------------
    case STATE_LISTENING: {
      if (!serial_listening_enabled) {
        break; // Se messo in pausa da seriale, non processa finestre (zero delay bloccanti)
      }

      uint64_t end_sample;
      uint32_t errors;
      bool ready = snapshot(end_sample, errors);

      if (errors != last_errors) {
        Serial.printf("ATTENZIONE: letture audio incomplete=%lu. Attendo una finestra integra.\n", (unsigned long)errors);
        last_errors = errors;
      }

      if (ready) {
        uint64_t advance = last_end ? end_sample - last_end : 0;
        last_end = end_sample;

        float confidence = 0.0f;
        bool detected = analyze(false, end_sample, errors, advance, confidence);

        // Interazione Flusso A -> Flusso B: se EVA supera la soglia, avvia il Pre-Allarme
        if (detected) {
          Serial.printf("⚠️ Parola chiave 'EVA' rilevata (Confidenza: %.2f)! Avvio finestra di %d secondi...\n",
                        confidence, CANCEL_WINDOW_MS / 1000);

          currentState = STATE_PRE_ALARM;
          preAlarmStartTime = millis();
          lastVibrationToggleTime = millis();

          // Attiva immediatamente il primo impulso del motore di vibrazione
          motorState = true;
          digitalWrite(VIBRATION_PIN, HIGH);
        }
      }
      break;
    }

    // ----------------------------------------------------------------------
    // STATO 2: PRE-ALLARME (Vibrazione a impulsi e attesa bottone per 5 sec)
    // ----------------------------------------------------------------------
    case STATE_PRE_ALARM: {
      // A. CONTROLLO PULSANTE DI ANNULLAMENTO (Reattività immediata)
      if (digitalRead(CANCEL_BUTTON_PIN) == LOW) {
        Serial.println("✅ Allarme ANNULLATO dall'utente tramite pulsante.");
        digitalWrite(VIBRATION_PIN, LOW);
        motorState = false;

        // Svuota i campioni accumulati durante la vibrazione per evitare falsi positivi
        // dovuti al rumore meccanico del motorino sul microfono PDM
        portENTER_CRITICAL(&ring_lock);
        valid_samples = 0;
        last_end = total_samples;
        portEXIT_CRITICAL(&ring_lock);

        currentState = STATE_LISTENING;
        break;
      }

      // B. GESTIONE VIBRAZIONE A IMPULSI (Non bloccante con millis())
      if (currentMillis - lastVibrationToggleTime >= PULSE_INTERVAL_MS) {
        lastVibrationToggleTime = currentMillis;
        motorState = !motorState;
        digitalWrite(VIBRATION_PIN, motorState ? HIGH : LOW);
      }

      // C. CONTROLLO SCADENZA FINESTRA DI ANNULLAMENTO (5000 ms)
      if (currentMillis - preAlarmStartTime >= CANCEL_WINDOW_MS) {
        digitalWrite(VIBRATION_PIN, LOW);
        motorState = false;
        currentState = STATE_SENDING_SOS;
      }
      break;
    }

    // ----------------------------------------------------------------------
    // STATO 3: INVIO NOTIFICA BLE ALLO SMARTPHONE
    // ----------------------------------------------------------------------
    case STATE_SENDING_SOS: {
      triggerEmergency(); // Notifica la caratteristica BLE ("SOS")

      // Invalida la finestra audio registrata durante la vibrazione prima di tornare in ascolto
      portENTER_CRITICAL(&ring_lock);
      valid_samples = 0;
      last_end = total_samples;
      portEXIT_CRITICAL(&ring_lock);

      currentState = STATE_LISTENING;
      break;
    }
  }
}
