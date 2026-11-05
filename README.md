# EVA — Emergency Voice Activated Bracelet

**EVA (Emergency Voice Activated Bracelet)** is a university project developed for the Digital Systems Design course at Politecnico di Milano. It explores a voice-activated bracelet to support requests for help in dangerous situations, with particular attention to violence against women. The project investigates an **embedded keyword spotting** system that detects the Italian activation phrase **“EVA aiutami”** (“EVA, help me”), allowing the user to initiate an emergency request without manual interaction.

The system is designed to acquire audio from the digital microphone of the **XIAO ESP32-S3 Sense** and run inference locally on its microcontroller. Model selection considers recognition performance alongside network size and computational requirements, with the aim of identifying a classifier suitable for the memory and processing constraints of the ESP32-S3.

This repository contains both the **Python/TensorFlow model-development pipeline** and the **Arduino firmware**. Python experiments compare compact CNNs using MFCC inputs and grouped cross-validation; the firmware performs acquisition and inference locally. The full bracelet interaction (vibration, cancellation and BLE) has a scaffold and is not implemented yet.

## Repository layout

```text
EVA-KWS/
├── training/                 # MFCC, CNNs, training, cross-validation, campaigns
│   ├── train.py              # Spyder entry point: single / custom grid
│   ├── train_final.py        # Final fit and float32/int8 exports
│   └── third_party/          # Python DSP sources, provenance and licenses
├── deployment/               # Model conversion and Arduino data generation
│   ├── tflite_export.py
│   └── prepare_arduino_reference.py
├── firmware/
│   ├── eva_bracelet/          # Future complete application: explicit scaffold
│   ├── eva_continuous/        # Working continuous microphone/CNN comparison
│   ├── eva_dataset_recorder/  # Ten-second WAV recorder, filename then rec
│   ├── tests/                # Model, MFCC parity and triggered microphone tests
│   └── libraries/EVA_MFCC/    # Shared C++ DSP dependency and license notices
├── docs/                     # Setup, dataset, hardware and implementation status
├── requirements.txt
└── .gitignore
```

Local `data/`, `data_v2/`, `cache/`, `results/`, `reports/`, `local_tools/` and development `tests/` remain outside version control. They are not duplicated or moved by the source-code layout. Root paths are resolved from the script location, independent of Spyder's current directory. Generated model arrays and arrays containing recorded audio are also ignored.

## Python setup and entry points

Use the existing working Anaconda environment, or install `requirements.txt` in a separate environment. Do not reinstall TensorFlow into a working environment merely because the files moved. Python modules now use package-qualified imports (`training.*`, `deployment.*`).

In Spyder, open **`training/train.py`** and run the whole file with F5. Configure the first section:

- `MODE='single'`: one feature/network configuration, with grouped cross-validation when enabled.
- `MODE='custom_grid'`, `GRID_SCOPE='full'`: the Cartesian feature/network grid.
- `MODE='custom_grid'`, `GRID_SCOPE='selected'`: only the explicitly listed candidate pairs.
- `MODE='custom_grid'`, `GRID_SCOPE='three_class_campaign'`, `SETTINGS.binary=False`: the matched three-class campaign.

`SETTINGS.binary=True` merges unknown and noise into non_target in memory; original WAV labels are preserved. Training and split seeds are configured separately. Five grouped outer folds keep each speaker/environment out of training once; internal grouped validation selects the checkpoint. Full learning histories are retained across all configured epochs. Grid outputs save numeric metrics without opening hundreds of plot windows.

The complete grid compares four MFCC/Mel choices, two CNN families, four filter sequences and three kernels (96 configurations). The three-class campaign adds matched regularization, batch, learning-rate and feature follow-ups, deduplicating shared configurations (114 unique cases, 570 trainings with five folds/one seed). [Training details](training/README.md).

Equivalent terminal commands from the repository root:

```bash
python -m training.train
python -m training.train_final
python -m deployment.prepare_arduino_reference
```

These commands execute work; importing/opening the modules does not start training. Check settings before launching. For a final fit, open **`training/train_final.py`**: it saves the best Keras checkpoint, inference-only Keras network, float32/int8 TFLite models and conversion comparisons under `results/final_model/<timestamp>/`. Historical train/test data are pooled for a new grouped training/validation split; `test_finale`, if present, stays excluded. These validation checks are not a new independent performance estimate. [Deployment details](deployment/README.md).

## Dataset and Edge Impulse

Recordings are acquired with the XIAO ESP32-S3 Sense microphone: mono PCM16 WAV at 16kHz. Ten-second recordings are annotated and cropped into 1100ms samples, ensuring the complete phrase is present in positive examples. Classes are:

| Label | Content |
|---|---|
| `eva` | Complete “EVA aiutami” phrase |
| `unknown` | Other speech, including isolated EVA/aiutami and confusing negative phrases |
| `noise` | Environmental sounds without speech under the adopted labeling policy |

Speaker IDs (`Pxx`) and environment IDs (`Axx`) preserve group identity across recordings and sessions. Edge Impulse Studio supports annotation and segmentation; model development runs locally. The Python MFCC implementation derives from Edge Impulse sources with provenance and licenses in `training/third_party/`. The C++ wrapper and its dependencies are in `firmware/libraries/EVA_MFCC/`.

Audio and generated reports/results are not distributed. The collaborator must receive authorized data separately or prepare their own compatible dataset. [Dataset structure](docs/dataset.md).

## Arduino setup and current status

Install **esp32 by Espressif Systems 3.3.12** in Boards Manager and **Chirale_TensorFLowLite 2.0.0** in Library Manager. Install the repository's **EVA_MFCC** library using the [firmware setup instructions](firmware/README.md). Select XIAO_ESP32S3, the USB port and USB CDC On Boot Enabled. The SD recorder additionally needs OPI PSRAM and a FAT32 microSD.

Each Arduino sketch lives in a directory with the same name as its `.ino`. Open that `.ino`; Arduino compiles the accompanying `.cpp` files automatically. Model arrays must be generated locally or supplied separately before compiling inference sketches.

| Sketch | Status |
|---|---|
| `firmware/eva_dataset_recorder` | WAV recorder: send full filename, then `rec` |
| `firmware/eva_continuous` | Continuous acquisition, 1100ms windows / nominal 500ms stride; float32/int8 scores |
| `firmware/tests/eva_model_test` | CNN parity using precomputed MFCCs |
| `firmware/tests/eva_mfcc_test` | Saved PCM -> on-device MFCC -> both CNNs |
| `firmware/tests/eva_microphone_test` | Command-triggered microphone acquisition and comparison |
| `firmware/eva_bracelet` | Scaffold only; full alarm/BLE integration pending |

The saved-reference checks established close Python/C++ MFCC agreement and matching predictions for the selected reference clip. Continuous recognition still needs evaluation on complete phrases, isolated words and long negative recordings. Scores printed by test firmware are not a calibrated alarm policy. [Hardware](docs/hardware.md) · [Implementation roadmap](docs/implementation.md).

## Collaborating

Keep `main` usable and work on a branch per change, then review a pull request. Before committing, inspect GitHub Desktop's file list: WAVs, checkpoints, reports, model/audio arrays and credentials must stay local. Preserve source/license notices for bundled DSP dependencies. The README and setup documents explain how to reproduce dependencies; installing a library on one Mac does not install it for collaborators.
