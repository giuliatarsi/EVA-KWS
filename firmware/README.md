# Arduino firmware

## Dependencies

Install esp32 by Espressif Systems3.3.12 using Boards Manager and Chirale_TensorFLowLite2.0.0 using Library Manager. EVA_MFCC1.0.0 is included in `libraries/EVA_MFCC/` with its original notices and PROVENANCE.txt. It packages selected official Edge Impulse DSP sources at revision4ec5868a78b0455e8148489548d3de12dd8e6b96 plus the project's wrapper. Public KissFFT symbols are prefixed to avoid collisions with TensorFlow Lite Micro. No second inference engine is bundled.

To install EVA_MFCC, copy that directory into the sketchbook `libraries/` directory shown in Arduino IDE Preferences. The resulting path is `<sketchbook>/libraries/EVA_MFCC/library.properties`; avoid an extra nested folder. Restart Arduino IDE if needed. Alternatively create a local directory symlink to the repo copy so edits stay synchronized. On the development Mac this link is already installed; collaborators should create their own link or copy. Third-party source code is distributed with applicable file-level license notices; see the library's LICENSES.md.

Select XIAO_ESP32S3, the connected port and USB CDC On Boot Enabled. The recorder needs OPI PSRAM. Open the .ino with the same name as its sketch directory. Generated model/audio arrays are local and must be supplied before compiling inference examples: see [deployment](../deployment/README.md).

## Programs

- `eva_bracelet/`: integrated INT8 recognition, eight-second pulsed vibration/cancellation window and BLE SOS notification.
- `tests/eva_continuous/`: working float32/int8 continuous comparison, 1100ms windows, nominal500ms stride. Commands p=pause analysis, s=resume, r=saved reference. Capture runs in its own task. Read-error/update-gap diagnostics must be monitored.
- `eva_dataset_recorder/`: full filename then rec; WAV10s,16kHz mono PCM16; see its README.
- `tests/eva_model_test/`: precomputed MFCC -> two CNNs. Reload input before every invocation because the micro runtime can reuse input storage.
- `tests/eva_mfcc_test/`: saved PCM -> C++ MFCC -> CNN. Send r to repeat.

All serial monitors use115200 baud. The main bracelet sketch sends BLE SOS; the diagnostic continuous and vibration sketches do not. Reference comparisons validate one clip, not recognition performance on all speakers or false alarms per hour.

Historical deployed MFCC settings:16kHz,1100ms, preemphasis0.98 with the Python pipeline's circular previous sample, version4,20ms frame/hop,13coefficients,40unquantized Mel filters,FFT256,0..Nyquist and CMVN101. Native C++ reference check before deployment: max absolute error2.02656e-05, RMSE4.93818e-06. Keep preprocessing and model configuration paired when changing models.

New final training uses V3 and FFT512. EVA_MFCC defaults to FFT512 and also supports historical FFT256 explicitly. Existing sketches pass reference_fft_length from reference_data.h, currently 256 for the old model. Regenerate/copy model and reference artifacts together to switch to 512. The wrapper supports13 or20 coefficients with either32 or40 Mel filters; callers pass reference_mel_filters for newly generated references. Legacy callers default to 40 Mel filters. See docs/implementation.md for current status and validation limits.

## B003 deployment checks

- `tests/eva_b003_mfcc_test/`: B003 float32/int8 on saved PCM, using MFCC13, Mel32 and FFT512. Checks MFCC and predictions against generated PC references; send `r` to repeat.
- `tests/eva_b003_test/`: B003 INT8 microphone recognition with D1 (GPIO2) motor control, three seconds of pulsed vibration, no BLE. Stops at boot if the saved reference fails. Threshold 0.50 is a provisional baseline, not a calibrated operating threshold. Audio windows are 1100 ms with a nominal 500 ms hop; fresh audio is required after vibration.

Both sketches use artifacts from final run `20261120_152323_263633`. Generated model/audio arrays remain excluded from Git. Compilation alone does not establish on-board parity or live recognition performance. The integrated BLE bracelet now uses the same B003 INT8 model and MFCC13/Mel32/FFT512 front-end; it uses the B003 validation-selected threshold0.78125 and retains the eight-second cancellation/BLE procedure.

## Three-class bracelet comparison

`eva_bracelet_three_class/eva_bracelet_three_class.ino` preserves the binary bracelet's pins, 1100 ms windows /500 ms hop, validation-selected threshold0.51171875, eight-second pulsed vibration, cancellation button and BLE event. It uses the three-class candidate (CNN1D 8-16-32, kernel7, training dropout0.50, MFCC13/Mel40/FFT512) from final run `results/final_model_three_class/20261120_161757_451012`. Outputs are ordered `eva`, `noise`, `unknown`; EVA must exceed both negative scores and the threshold to trigger. Boot parity checks all three scores and MFCC against the PC reference. The binary `eva_bracelet` remains separate. Uploading either sketch replaces the firmware currently running on the board; both source versions remain on the computer. Live false-activation performance must be measured, not inferred from compilation.

## Named live comparison sketches

Four independent sketches preserve the same D1 motor / D2 cancellation, BLE and eight-second pre-alert. Each now has its own validation-selected EVA threshold (see below):

- `eva_bracelet_B003`: binary, filters8-16-32, kernel3, MFCC13/Mel32, training dropout0.25.
- `eva_bracelet_B001`: binary, filters8-16-32, kernel7, MFCC13/Mel32, training dropout0.25.
- `eva_bracelet_T001_dropout050`: three classes, filters8-16-32, kernel7, MFCC13/Mel40, training dropout0.50 (tested regularisation follow-up).
- `eva_bracelet_T003_dropout025`: three classes, filters8-16-32, kernel3, MFCC20/Mel40, training dropout0.25 (five-fold verification rejected0.50).

Each directory contains the same-named .ino and a DEPLOYMENT.json describing its exact model run. All use FFT512 and INT8. Open only the desired sketch and upload it; the board runs one firmware at a time. Save scores and false activations for identical utterances/conditions; do not interpret these tests as isolating class count alone because architectures differ. Boot checks compare saved PCM/MFCC and expected EVA scores against the PC; three-class sketches check all three output scores. Model/audio .cpp arrays remain local and Git-ignored.

## Candidate-specific thresholds

Named live sketches now use thresholds maximizing EVA F1 on each exported INT8 model's validation scores (highest threshold among tied F1 maxima). The underlying train/validation sample and group assignments were verified identical across all four exports. These are optimized validation results, not independent test performance. The previous common0.80 setting is superseded:

- B003: threshold0.78125000.
- B001: threshold0.63671875.
- T001_dropout050: threshold0.51171875.
- T003_dropout025: threshold0.87500000.

The original T003_dropout050 sketch has been moved to local results/firmware_archive; the active corrected version is T003_dropout025. Binary eva_bracelet and three-class eva_bracelet_three_class aliases use the B003 and T001 thresholds respectively.
