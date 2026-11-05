# Hardware and development environment

Target: Seeed Studio XIAO ESP32-S3 Sense with microphone expansion board attached. PDM microphone CLK=GPIO42, DATA=GPIO41. SD chip select GPIO21; the recorder uses the board-default SPI pins, FAT32 microSD and OPI PSRAM. Audio is 16kHz PCM16 mono. Current acquisition gain is 2, matching the initial dataset recorder; verify it if data collection settings change.

Arduino: esp32 core3.3.12, Chirale_TensorFLowLite2.0.0, repository EVA_MFCC1.0.0. Select XIAO_ESP32S3 and USB CDC On Boot Enabled. Serial Monitor uses115200 baud. Ports vary by computer; do not hard-code the current Mac's USB path. OPI PSRAM must be enabled for the ten-second recorder.

Vibration driver, motor, cancellation input and BLE phone workflow require final wiring/API decisions. No pin assignments for these unimplemented components are claimed here. Document the agreed wiring before enabling alarm actions.
