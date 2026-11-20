# Implementation and validation status

The main application is firmware/eva_bracelet/eva_bracelet.ino. It implements PDM acquisition in a separate task, 1100 ms windows with nominal 500 ms updates, MFCC extraction, INT8 inference and a listening/pre-alarm/sending state machine. Driver D1 and cancellation D2 match the schematic. The binary B003 trigger requires EVA to win and score at least0.78125, selected to maximize EVA F1 on INT8 validation predictions. This optimized validation threshold is not a guarantee of continuous detection performance. The three-class alias uses the T001 threshold0.51171875; named comparison sketches carry their own thresholds in DEPLOYMENT.json.

The pre-alarm lasts eight seconds, with500 ms vibration on/off pulses. A button press cancels it. Otherwise the firmware notifies SOS over BLE. Audio acquired during vibration is invalidated before recognition resumes. With no phone connected the event is logged but is not queued or acknowledged end to end.

The Flutter app in eva_gateway/lib/main.dart receives BLE SOS, retrieves phone location, stores an alert in Firestore and requests push notifications through the Vercel backend/Firebase Cloud Messaging. The root code_app.dart is an earlier Telegram prototype, not the current gateway. Source implementation does not establish measured delivery reliability, background continuity or battery autonomy.

Diagnostics live in firmware/tests/: eva_model_test checks the CNN on saved MFCC; eva_mfcc_test checks saved PCM -> MFCC -> CNN; eva_continuous compares float32/int8 on live audio; eva_vibration_test exercises three seconds of vibration without BLE or cancellation.

The main bracelet now contains B003 INT8 from final run 20261120_152323_263633: CNN1D filters 8-16-32, kernel 3, 2738 parameters, with MFCC13 / Mel32 / FFT512. Its generated reference header supplies both FFT length and Mel count. EVA_MFCC supports FFT256/512 and Mel32/40; model and preprocessing artifacts must be updated together. The B003 MFCC/model reference test passed on the board; the integrated BLE sketch still requires its live hardware checks. Historical diagnostic sketches retain their matching older references.


Measure continuous recognition, thresholds, end-to-end notification latency, RAM and power on the final deployed version. Existing reference parity validates one historical clip, not general recognition accuracy.
