# Voice-triggered vibration test

Power the XIAO ESP32-S3 Sense via USB-C. Connect D1 to IN on the Adafruit 5648, 3V3_OUT to V+, and GND to GND. Connect the motor to OUT+ and OUT-. Leave the battery and capacitor disconnected for this test.

Use the EVA_MFCC and Chirale_TensorFLowLite libraries. Upload as XIAO_ESP32S3 with USB CDC On Boot enabled. Open the Serial Monitor at 115200 baud and say **EVA aiutami** once listening starts. The INT8 model uses the existing 256-point FFT configuration. A detection starts three seconds of vibration (200 ms on/off pulses), followed by a fresh audio window and resumed classification. This sketch uses no BLE.

The generated model and reference arrays are local files. Copy the matching `eva_int8_model.cpp` and `reference_data.cpp` into this folder if they are absent.
