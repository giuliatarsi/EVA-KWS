// ==========================================================================
// 1. LIBRERIE (BLE, Audio I2S per Core v3.x e Edge Impulse)
// ==========================================================================
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ESP_I2S.h> // Nuova libreria nativa per Core Arduino ESP32 v3.0.x

// SOSTITUISCI CON IL NOME DELLA TUA LIBRERIA EDGE IMPULSE QUANDO PRONTA:
#include <ei_keyword_inferencing.h>

// ==========================================================================
// 2. CONFIGURAZIONE PARAMETRI E PIN (Modificabili in 2 secondi)
// ==========================================================================
// Parametri Modello Vocale Edge Impulse
#define KEYWORD_LABEL        "keyword" // Etichetta della parola chiave nel modello
#define CONFIDENCE_THRESHOLD 0.80      // Soglia di attivazione (80%)

// Pin Hardware (Pin consigliati su Seeed Studio XIAO ESP32-S3)
#define VIBRATION_PIN        D0        // GPIO1 -> Collegare al Gate del MOSFET
#define CANCEL_BUTTON_PIN    D1        // GPIO2 -> Collegare tra D1 e GND (usa INPUT_PULLUP)

// Pin interni fissi del microfono PDM sulla scheda di espansione XIAO Sense
#define I2S_MIC_CLOCK_PIN    42
#define I2S_MIC_DATA_PIN     41

// Tempistiche (in millisecondi) gestite senza bloccare il codice
#define CANCEL_WINDOW_MS     5000      // 5 secondi per annullare il falso positivo
#define PULSE_INTERVAL_MS    200       // Velocità impulsi vibrazione (200ms ON / 200ms OFF)

// Codici UUID Bluetooth (Identici a quelli dell'app Flutter)
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// ==========================================================================
// 3. VARIABILI GLOBALI E MACCHINA A STATI
// ==========================================================================
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// Oggetto per gestire il microfono e memoria (buffer) per salvare i campioni audio
I2SClass I2S;
int16_t *sampleBuffer = NULL;

// I 3 stati possibili del nostro braccialetto
enum SystemState {
  STATE_LISTENING,    // Ascolta il microfono con Edge Impulse
  STATE_PRE_ALARM,    // Vibra a impulsi per 5s e aspetta l'eventuale pressione del bottone
  STATE_SENDING_SOS   // Tempo scaduto: invia la notifica BLE al telefono
};

SystemState currentState = STATE_LISTENING;

// Cronometri digitali per gestire il tempo con millis() (senza usare delay)
unsigned long preAlarmStartTime = 0;
unsigned long lastVibrationToggleTime = 0;
bool motorState = false;

// ==========================================================================
// 4. GESTIONE CONNESSIONE BLUETOOTH E INVIO SOS
// ==========================================================================
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Smartphone connesso via BLE!");
    };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Smartphone disconnesso. Riavvio visibilità Bluetooth...");
      BLEDevice::startAdvertising();
    }
};

void triggerEmergency() {
  if (deviceConnected) {
    String alertMsg = "SOS";
    pCharacteristic->setValue(alertMsg.c_str());
    pCharacteristic->notify(); // Suona il campanello sull'app Flutter
    Serial.println("🚨 ALLARME SOS INVIATO ALLO SMARTPHONE!");
  } else {
    Serial.println("⚠️ Impossibile inviare SOS: nessun telefono connesso.");
  }
}

// Funzione "ponte" richiesta da Edge Impulse per leggere i numeri dell'audio
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&sampleBuffer[offset], out_ptr, length);
  return 0;
}

// ==========================================================================
// 5. CONFIGURAZIONE INIZIALE (Eseguita una sola volta all'accensione)
// ==========================================================================
void setup() {
  Serial.begin(115200);

  // Configurazione Pin Motore e Bottone
  pinMode(VIBRATION_PIN, OUTPUT);
  digitalWrite(VIBRATION_PIN, LOW); // Motore spento all'avvio

  // INPUT_PULLUP attiva una resistenza interna: il pin legge HIGH quando il bottone
  // è a riposo, e legge LOW appena il bottone viene premuto (chiudendo verso GND)
  pinMode(CANCEL_BUTTON_PIN, INPUT_PULLUP);

  // --- Avvio Bluetooth Low Energy ---
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
  Serial.println("Bluetooth attivo. In attesa di connessione...");

  // --- Avvio Microfono Digitale PDM (Sintassi Core ESP32 v3.0.x) ---
  I2S.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  // Avvia il microfono alla frequenza richiesta dal tuo modello Edge Impulse (es. 16000 Hz)
  if (!I2S.begin(I2S_MODE_PDM_RX, EI_CLASSIFIER_FREQUENCY, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("ERRORE: Impossibile avviare il microfono PDM!");
    while (1);
  }

  // Riserva lo spazio nella memoria RAM ( o PSRAM) per registrare la finestra audio
  sampleBuffer = (int16_t *)malloc(EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
  if (sampleBuffer == NULL) {
    Serial.println("ERRORE: Memoria insufficiente per il buffer audio!");
    while (1);
  }
}

// ==========================================================================
// 6. CICLO PRINCIPALE (Macchina a Stati Non Bloccante)
// ==========================================================================
void loop() {
  // Leggiamo l'orologio interno del chip (millisecondi trascorsi dall'accensione)
  unsigned long currentMillis = millis();

  switch (currentState) {

    // ----------------------------------------------------------------------
    // STATO 1: ASCOLTO E RICONOSCIMENTO VOCALE (Edge Impulse)
    // ----------------------------------------------------------------------
    case STATE_LISTENING: {
      // 1. Registra uno spezzone audio dal microfono
      size_t bytesToRead = EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t);
      size_t bytesRead = I2S.readBytes((char *)sampleBuffer, bytesToRead);

      if (bytesRead == bytesToRead) {
        // 2. Prepara il pacchetto audio per la rete neurale
        signal_t signal;
        signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
        signal.get_data = &microphone_audio_signal_get_data;
        ei_impulse_result_t result = { 0 };

        // 3. Esegue l'inferenza (run_classifier)
        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        if (err != EI_IMPULSE_OK) return;

        // 4. Controlla i risultati per vedere se ha sentito la parola d'allarme
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
          if (strcmp(result.classification[ix].label, KEYWORD_LABEL) == 0) {

            // Se la sicurezza supera la soglia (es. > 0.80)
            if (result.classification[ix].value > CONFIDENCE_THRESHOLD) {
              Serial.printf("⚠️ Parola rilevata (%s: %.2f)! Avvio finestra di 5 secondi...\n",
                            result.classification[ix].label, result.classification[ix].value);

              // Passa allo stato di PRE-ALLARME e fa partire i cronometri
              currentState = STATE_PRE_ALARM;
              preAlarmStartTime = millis();
              lastVibrationToggleTime = millis();

              // Accende subito il primo impulso di vibrazione
              motorState = true;
              digitalWrite(VIBRATION_PIN, HIGH);
              break;
            }
          }
        }
      }
      break;
    }

    // ----------------------------------------------------------------------
    // STATO 2: PRE-ALLARME (Vibrazione a impulsi e attesa bottone per 5 sec)
    // ----------------------------------------------------------------------
    case STATE_PRE_ALARM: {
      // A. CONTROLLO PULSANTE DI ANNULLAMENTO (Reattività immediata)
      // Essendo in INPUT_PULLUP, quando premi il bottone il valore diventa LOW
      if (digitalRead(CANCEL_BUTTON_PIN) == LOW) {
        Serial.println("✅ Allarme ANNULLATO dall'utente tramite pulsante.");
        digitalWrite(VIBRATION_PIN, LOW); // Spegne subito il motore
        motorState = false;

        // Torna ad ascoltare il microfono
        currentState = STATE_LISTENING;
        break;
      }

      // B. GESTIONE VIBRAZIONE A IMPULSI (Senza usare delay!)
      // Controlla se sono passati 200 millisecondi dall'ultimo cambio di stato del motore
      if (currentMillis - lastVibrationToggleTime >= PULSE_INTERVAL_MS) {
        lastVibrationToggleTime = currentMillis; // Azzera il cronometro dell'impulso
        motorState = !motorState;                // Inverte: se era acceso spegne, se spento accende
        digitalWrite(VIBRATION_PIN, motorState ? HIGH : LOW);
      }

      // C. CONTROLLO SCADENZA DEI 5 SECONDI
      // Controlla se sono passati 5000 millisecondi da quando è scattato il pre-allarme
      if (currentMillis - preAlarmStartTime >= CANCEL_WINDOW_MS) {
        digitalWrite(VIBRATION_PIN, LOW); // Spegne il motore
        motorState = false;

        // Nessuno ha premuto il bottone in tempo: passa all'invio dell'SOS!
        currentState = STATE_SENDING_SOS;
      }
      break;
    }

    // ----------------------------------------------------------------------
    // STATO 3: INVIO NOTIFICA BLE ALLO SMARTPHONE
    // ----------------------------------------------------------------------
    case STATE_SENDING_SOS: {
      triggerEmergency(); // Invia il pacchetto "SOS" all'app Flutter

      // Una volta inviato l'allarme, riporta il braccialetto in ascolto
      currentState = STATE_LISTENING;
      break;
    }
  }
}
