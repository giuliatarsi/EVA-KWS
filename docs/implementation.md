# Firmware implementation status

Working checks: float32/int8 CNN inference; saved-audio C++ MFCC parity; command-triggered microphone acquisition; continuous ring-buffer acquisition with nominal500ms updates; separate WAV recorder.

Future main firmware is `firmware/eva_bracelet/`. Its files currently only mark responsibilities; the .ino prints a scaffold message. Planned flow:

microphone -> audio_capture -> keyword_detector (EVA_MFCC + selected CNN) -> alarm_manager -> ble_service -> smartphone.

Alarm states should represent listening, cancellation interval and sending explicitly. Vibration/cancellation/BLE must not block capture. Threshold and temporal confirmation require validation on continuous audio, including isolated-word false triggers. Continuous firmware's argmax labels are diagnostics, not a final alarm policy. The phone handles onward emergency communication; BLE alone is not an emergency messaging service.

Keep diagnostic sketches in firmware/tests and share the same MFCC library. The main firmware should use one selected model instead of shipping both comparison models. Measure end-to-end latency, capture continuity, RAM and consumption on the target.
