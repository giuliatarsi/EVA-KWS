// EVA application scaffold. Full integration is not implemented yet.
#include <Arduino.h>
#include "config.h"
void setup() {
  Serial.begin(eva::serial_baud);
  delay(500);
  Serial.println("EVA bracelet scaffold: microphone, detector, alarm and BLE integration pending.");
}
void loop() { delay(10); }
