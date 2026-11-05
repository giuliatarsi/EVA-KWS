# EVA WAV recorder — filename then rec

Open `eva_dataset_recorder.ino`. Select XIAO_ESP32S3, USB CDC On Boot Enabled, PSRAM OPI PSRAM and the USB port. Insert the FAT32 microSD while disconnected, then upload. Open Serial Monitor at 115200 baud with New Line selected.

Send the exact complete filename, then send `rec` as a separate line:

```
unknown_P15_S01_R01_solo-eva_esp.wav
rec
```

Speak when `REGISTRAZIONE IN CORSO` appears. There is no countdown: only microphone startup/settling before this message. The recording lasts ten seconds. Wait for `SALVATO`, then send another filename and `rec` again. The program never generates participant IDs, labels, suffixes or progressive numbers. It adds `.wav` only if missing. Names may contain ASCII letters, numbers, underscores, hyphens and dots, without spaces or directory separators; maximum 120 characters. A filename that already exists is rejected, not renamed or overwritten. Successful recording clears the pending name. `h` shows help.

Files are saved under `/recordings/` on the SD. Audio is acquired entirely into PSRAM and written after capture: mono WAV PCM16, 16kHz, ten seconds (320044 bytes including header), gain2 with saturation. Partial reads produce no WAV; incomplete newly-created files are removed on write failure. The recording is not analyzed by a CNN or MFCC. No SD formatting or removal of previous recordings takes place. Filesystem timestamps are not synchronized; use filenames and a collection log.

Use the participant's existing ID. Crop distinct pronunciations to 1100ms later; isolated EVA/aiutami and confusing negative phrases should be labelled unknown. The 50–100 sample target means cropped distinct pronunciations, not ten-second files. Preserve speaker-group separation in training/validation/test.

Upload `../eva_continuous/eva_continuous.ino` to return to live recognition. Uploading the recorder replaces the program on the board, not the recordings on the SD.
