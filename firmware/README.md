# Arduino firmware

## Dependencies

Install esp32 by Espressif Systems3.3.12 using Boards Manager and Chirale_TensorFLowLite2.0.0 using Library Manager. EVA_MFCC1.0.0 is included in `libraries/EVA_MFCC/` with its original notices and PROVENANCE.txt. It packages selected official Edge Impulse DSP sources at revision4ec5868a78b0455e8148489548d3de12dd8e6b96 plus the project's wrapper. Public KissFFT symbols are prefixed to avoid collisions with TensorFlow Lite Micro. No second inference engine is bundled.

To install EVA_MFCC, copy that directory into the sketchbook `libraries/` directory shown in Arduino IDE Preferences. The resulting path is `<sketchbook>/libraries/EVA_MFCC/library.properties`; avoid an extra nested folder. Restart Arduino IDE if needed. Alternatively create a local directory symlink to the repo copy so edits stay synchronized. On the development Mac this link is already installed; collaborators should create their own link or copy. Third-party source code is distributed with applicable file-level license notices; see the library's LICENSES.md.

Select XIAO_ESP32S3, the connected port and USB CDC On Boot Enabled. The recorder needs OPI PSRAM. Open the .ino with the same name as its sketch directory. Generated model/audio arrays are local and must be supplied before compiling inference examples: see [deployment](../deployment/README.md).

## Programs

- `eva_bracelet/`: future full application scaffold; no detection/alarm/BLE implementation yet.
- `eva_continuous/`: working float32/int8 continuous comparison, 1100ms windows, nominal500ms stride. Commands p=pause analysis, s=resume, r=saved reference. Capture runs in its own task. Read-error/update-gap diagnostics must be monitored.
- `eva_dataset_recorder/`: full filename then rec; WAV10s,16kHz mono PCM16; see its README.
- `tests/eva_model_test/`: precomputed MFCC -> two CNNs. Reload input before every invocation because the micro runtime can reuse input storage.
- `tests/eva_mfcc_test/`: saved PCM -> C++ MFCC -> CNN. Send r to repeat.
- `tests/eva_microphone_test/`: m records1100ms after countdown; r runs the saved reference check.

All serial monitors use115200 baud. No current example sends an alarm or BLE event. Reference comparisons validate one clip, not recognition performance on all speakers or false alarms per hour.

MFCC settings:16kHz,1100ms, preemphasis0.98 with the Python pipeline's circular previous sample, version4,20ms frame/hop,13coefficients,40unquantized Mel filters,FFT256,0..Nyquist and CMVN101. Native C++ reference check before deployment: max absolute error2.02656e-05, RMSE4.93818e-06. Keep preprocessing and model configuration paired when changing models.
